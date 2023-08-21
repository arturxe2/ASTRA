import logging
import os
import time
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.utils import AverageMeter, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.ActionSpotting import evaluate
import json
import zipfile
import wandb
from torch.cuda.amp import GradScaler
from torch import autocast
from eval import pred2vec, compute_mAP
import pandas as pd
from torchvision.io import read_image
from dataset import feats2clip

def trainerAS(train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            patience,
            model_name,
            max_epochs=1000,
            batch_size=32,
            chunk_size=32,
            outputrate=2,
            path_experiments=None):
    """
    Function to train the action spotting model (with validation early stopping)
    """

    logging.info("start training action spotting")

    best_loss = 0
    n_bad_epochs = 0
    
    for epoch in range(max_epochs):
        best_model_path = os.path.join(path_experiments, 'ASmodels', model_name, 'model.pth.tar')

        #Update scheduler
        scheduler.step(epoch+1)
        logging.info(optimizer.param_groups[0]['lr'])

        # train for one epoch
        loss_training = trainAS(train_loader, model, criterion, optimizer, epoch + 1, train=True, 
                                chunk_size=chunk_size, outputrate=outputrate)
        
        # evaluate on validation set
        loss_validation = trainAS(val_loader, model, criterion, optimizer, epoch + 1, train=False,
                                chunk_size=chunk_size, outputrate=outputrate)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join(path_experiments, "ASmodels", model_name), exist_ok=True)

        # remember best loss and save checkpoint
        is_better = loss_validation >= best_loss
        best_loss = max(loss_validation, best_loss)

        # Save the best model based on loss only if validation improves
        if is_better:
            n_bad_epochs = 0
            torch.save(state, best_model_path)
        
        else:
            n_bad_epochs += 1

        #If doesn't improve reduce LR / finish training
        if n_bad_epochs == patience:
            break

    return

def trainerAS_test(train_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            patience,
            model_name,
            max_epochs=1000,
            batch_size=32,
            chunk_size=32,
            outputrate=2,
            path_experiments=None):
    
    """
    Function to train the action spotting model (when no validation early stopping - for using all data and evaluate on challenge)
    """

    logging.info("start training action spotting")

    best_loss = 0
    n_bad_epochs = 0

    
    for epoch in range(max_epochs):
        best_model_path = os.path.join(path_experiments, 'ASmodels', model_name, 'model.pth.tar')

        #Update scheduler
        scheduler.step(epoch+1)
        logging.info(optimizer.param_groups[0]['lr'])

        # train for one epoch
        loss_training = trainAS(train_loader, model, criterion, optimizer, epoch + 1, train=True, 
                                chunk_size=chunk_size, outputrate=outputrate)
        
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join(path_experiments, "ASmodels", model_name), exist_ok=True)

        # remember best loss and save checkpoint
        is_better = True
        best_loss = max(loss_training, best_loss)

        # Save the best model based on loss only if validation improves
        if is_better:
            n_bad_epochs = 0
            torch.save(state, best_model_path)
        
        else:
            n_bad_epochs += 1

        #If doesn't improve reduce LR / finish training
        if n_bad_epochs == patience:
            break

    return

#Define train for AS part
def trainAS(dataloader,
        model,
        criterion,
        optimizer,
        epoch,
        train=False,
        chunk_size=32,
        outputrate=2):
    
    """
    Function to train 1 epoch of the action spotting model
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    lossesC = AverageMeter()
    lossesD = AverageMeter()

    if train:
        model.train()
    else:
        model.eval()
        list_preds = []
        list_labels = []

    end = time.time()

    scaler = GradScaler() #for mixed precision
    
    #Iterate dataloader
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, data in t:
                    
            # measure data loading time
            data_time.update(time.time() - end)
            labels = data['labels'].cuda()
            labelsD = data['labels_displ'].cuda()

            if model.baidu:
                featB = data['featB'].cuda()
            else:
                featB = None
            if model.audio:
                featA = data['featA'].cuda()
            else:
                featA = None
            if model.use_frames:
                featF = data['frames'].cuda()
            else:
                featF = None

            #make predictions
            with autocast(device_type='cuda', dtype=torch.float16):
                if train:
                    output = model(featsB = featB, featsA = featA, featsF = featF, labels = labels, labelsD = labelsD, inference = False)
                    lossC, lossD = criterion(output['labels'], output['preds'], output['labelsD'], output['predsD'])

                else:
                    preds = []
                    with torch.no_grad():
                        output = model(featsB = featB, featsA = featA, featsF = featF, inference = True)
                        lossC, lossD = criterion(labels, output['preds'], labelsD, output['predsD'])

                    if (model.model_cfg['uncertainty']) & (model.model_cfg['uncertainty_mode'] != 'mse'):
                        output['predsD'] = output['predsD'][:, :, :, 0]
                    
                    #To compute MAP (labels + predictions)
                    labs = pred2vec([labels, labelsD], chunk_size = chunk_size, outputrate = outputrate, threshold = 0.2, target = True, window = 4)
                    y = [list_labels.append(lab) for lab in labs]
                    y = [list_preds.append(pred) for pred in pred2vec([output['preds'], output['predsD']], chunk_size = chunk_size, outputrate = outputrate, threshold = 0.2, NMS = True, window = 4)]
                    
                
                loss = lossC + lossD

            if train:
                train_step = (epoch-1) * len(dataloader) + i
                if train_step % 50 == 0:
                    wandb.log({"train/step": train_step, "train/ASloss":loss.item(), "train/ASlossC": lossC.item(), "train/ASlossD": lossD.item()})
            else:
                val_step = (epoch-1) * len(dataloader) + i
                if val_step % 50 == 0:
                    wandb.log({"val/step": val_step, "val/ASloss":loss.item(), "val/ASlossC": lossC.item(), "val/ASlossD": lossD.item()})

            losses.update(loss.item(), labels.size(0))
            lossesC.update(lossC.item(), labels.size(0))
            lossesD.update(lossD.item(), labels.size(0))
                
            #compute gradient and backpropagate
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss:{losses.avg:.3e} '
            desc += f'LossC:{lossesC.avg:.3e} '
            desc += f'LossD:{lossesD.avg:.3e} '

            t.set_description(desc)

    if train:
        wandb.log({"per_epoch/train/epoch": epoch, "per_epoch/train/ASloss":losses.avg, "per_epoch/train/ASlossC":lossesC.avg, "per_epoch/train/ASlossD":lossesD.avg, "per_epoch/train/LR":optimizer.param_groups[0]['lr']})        
        return losses.avg
    
    else:
        #Compute mAP
        amap, amap_class = compute_mAP(list_preds, list_labels)
        dataframe = pd.DataFrame(amap_class).T
        dataframe.columns = ["penalty", "kick-off", "goal", "substitution", "offside", "sh. on targ.", "sh. off targ.", "clearance", "ball oop", "throw in", "foul", "ind. fk", "dir. fk", "corner", "yc", "rc", "2nd yc"]
        table = wandb.Table(dataframe=dataframe)
        wandb.log({'aMAP per class epoch ' + str(epoch+1): table})
        logging.info('aMAP:' + str(amap))

        wandb.log({"per_epoch/val/epoch": epoch, "per_epoch/val/ASloss":losses.avg, "per_epoch/val/ASlossC":lossesC.avg, "per_epoch/val/ASlossD":lossesD.avg, "per_epoch/val/aMAP":amap})        
        return amap


def testSpotting(dataloader, model, model_name, overwrite=True, NMS_window = 8, NMS_threshold=0.5, framerate = 2, outputrate=2, 
                chunk_size=32, stride = 8, path_frames='/data-local/data1-ssd/axesparraguera/SoccerNetFrames', 
                postprocessing = 'SNMS', path_experiments = None):

    """
    Function for inference of the action spotting model (and evaluation)
    """

    #Take split from dataloader
    split = dataloader.dataset.split

    #Output results folder
    output_results = os.path.join(path_experiments, "ASmodels", model_name, 'post_' + str(postprocessing) + 'window_' + str(NMS_window), f"results_spotting_{split}.zip")
    output_folder = f"outputs_{split}"

    if not os.path.exists(output_results) or overwrite:
        batch_time = AverageMeter()
        data_time = AverageMeter()

        spotting_grountruth = list()
        spotting_grountruth_visibility = list()
        spotting_predictions = list()

        model.eval()

        count_visible = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
        count_unshown = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
        count_all = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)

        #nº of frames per clip + nº of repeated frames between consecutive clips
        n_frames = chunk_size * framerate
        n_repeated = max(0, n_frames - stride * framerate)

        end = time.time()
        
        #Iterate over games (dataloader)
        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:

            for i, (game_ID, data) in t:

                data_time.update(time.time() - end)
    
                game_ID = game_ID[0]
                label_half1 = data['label1'].float().squeeze(0)
                label_half2 = data['label2'].float().squeeze(0)

                if model.baidu:
                    featB_half1 = data['featB1'].reshape(-1, data['featB1'].shape[-1])
                    featB_half1 = feats2clip(featB_half1, stride = stride, clip_length = chunk_size)

                    featB_half2 = data['featB2'].reshape(-1, data['featB2'].shape[-1])
                    featB_half2 = feats2clip(featB_half2, stride = stride, clip_length = chunk_size)
                    lenB1 = len(featB_half1)
                    lenB2 = len(featB_half2)
                else:
                    lenB1 = int(data['frames1'])
                    lenB2 = int(data['frames2'])

                if model.audio:
                    featA_half1 = data['featA1'].reshape(-1, data['featA1'].shape[-1])
                    featA_half2 = data['featA2'].reshape(-1, data['featA2'].shape[-1])
                    featA_half1 = feats2clip(featA_half1.T, stride = stride * 100, clip_length = chunk_size * 100)
                    featA_half2 = feats2clip(featA_half2.T, stride = stride * 100, clip_length = chunk_size * 100)

                #batch size for testing
                BS = 4

                json_data = dict()
                json_data["UrlLocal"] = game_ID
                json_data["predictions"] = list()

                #HALF 1 PREDICTIONS
                featV = []
                
                #Initialize half1 preds
                timestamp_long_half_1 = np.zeros((data['frames1'] // 25 * outputrate + 1, 17))
                q = 0
                for b in tqdm(range(min(int(data['frames1'] // 25 // stride)+1, lenB1))):
                    if (b % BS == 0) | (b == len(label_half1)-1):
                        if b != 0:
                            if model.use_frames:
                                featV = torch.stack([torch.stack(feat) for feat in featV]).cuda()
                            else:
                                featV = []
                            if model.baidu:
                                featB = featB_half1[b-q:b].clone().cuda()
                            else:
                                featB = None
                            if model.audio:
                                featA = featA_half1[b-q:b].clone().cuda()
                            else:
                                featA = None
                            q = 0

                            with autocast(device_type='cuda', dtype=torch.float16):
                                with torch.no_grad():
                                    output = model(featsB = featB, featsA = featA, featsF = featV, inference = True)
                            predC = output['preds'].cpu().detach().numpy()
                            if model.model_cfg['uncertainty'] & (model.model_cfg['uncertainty_mode'] != 'mse'):
                                predD = output['predsD'][:, :, :, 0].cpu().detach().numpy()
                            else:
                                predD = output['predsD'].cpu().detach().numpy()
                            batch, nf, nc = predC.shape
                            for l in range(batch):
                                initial_pos = (b - len(predC) + l) * stride * outputrate
                                for j in range(nf):
                                    for k in range(nc-1):
                                        prob = predC[l, j, k+1]
                                        if prob > NMS_threshold:
                                            rel_position = j - predD[l, j, k+1]
                                            position = min(len(timestamp_long_half_1)-1, max(0, int((initial_pos + rel_position).round())))
                                            timestamp_long_half_1[position, k] = max(timestamp_long_half_1[position, k], prob)
                            
                            featV = []
                    
                    #Load frames
                    q += 1
                    if model.use_frames:
                        initial_frame = int(int(b * 25 * stride)) // 4 * 4
                        if b == 0:
                            
                            featV.append([read_image(os.path.join(path_frames, game_ID, 'half1', 'frame ' + str(max(0, min(int(data['frames1']), initial_frame + int(round((j * 25 / framerate) / 4) * 4)))) + '.jpg')) for j in range(n_frames)])
                            if n_repeated > 0:
                                aux_featV = featV[-1][-n_repeated:]

                        else:
                            #featV.append([cv2.imread(os.path.join(path_frames, game_ID, 'half1', 'frame ' + str(max(0, min(int(frames1), int(initial_frame + j * framestride)))) + '.jpg')) for j in range(n_frames)])
                            if n_repeated == 0:
                                featV.append([read_image(os.path.join(path_frames, game_ID, 'half1', 'frame ' + str(max(0, min(int(data['frames1']), initial_frame + int(round((j * 25 / framerate) / 4) * 4)))) + '.jpg')) for j in range(n_frames)])
                            else:
                                
                                s = [aux_featV.append(read_image(os.path.join(path_frames, game_ID, 'half1', 'frame ' + str(max(0, min(int(data['frames1']), initial_frame + int(round((j * 25 / framerate) / 4) * 4)))) + '.jpg'))) for j in range(n_repeated, n_frames)]
                                featV.append(aux_featV)
                                aux_featV = featV[-1][-n_repeated:]

                #HALF 2 PREDICTIONS
                featV = []
                
                #Initialize half2 preds
                timestamp_long_half_2 = np.zeros((data['frames2'] // 25 * outputrate + 1, 17))
                q = 0
                for b in tqdm(range(min(int(data['frames2'] // 25 // stride)+1, lenB2))):
                    if (b % BS == 0) | (b == len(label_half2)-1):
                        if b != 0:
                            if model.use_frames:
                                featV = torch.stack([torch.stack(feat) for feat in featV]).cuda()
                            else:
                                featV = []
                            if model.baidu:
                                featB = featB_half2[b-q:b].clone().cuda()
                            else:
                                featB = None
                            if model.audio:
                                featA = featA_half2[b-q:b].clone().cuda()
                            else:
                                featA = None
                            q = 0

                            with autocast(device_type='cuda', dtype=torch.float16):
                                with torch.no_grad():
                                    output = model(featsB = featB, featsA = featA, featsF = featV, inference = True)
                            predC = output['preds'].cpu().detach().numpy()
                            if model.model_cfg['uncertainty'] & (model.model_cfg['uncertainty_mode'] != 'mse'):
                                predD = output['predsD'][:, :, :, 0].cpu().detach().numpy()
                            else:
                                predD = output['predsD'].cpu().detach().numpy()
                            batch, nf, nc = predC.shape
                            for l in range(batch):
                                initial_pos = (b - len(predC) + l) * stride * outputrate
                                for j in range(nf):
                                    for k in range(nc-1):
                                        prob = predC[l, j, k+1]
                                        if prob > NMS_threshold:
                                            rel_position = j - predD[l, j, k+1]
                                            position = min(len(timestamp_long_half_2)-1, max(0, int((initial_pos + rel_position).round())))
                                            timestamp_long_half_2[position, k] = max(timestamp_long_half_2[position, k], prob)
                            
                            featV = []
                    
                    #Load frames
                    q += 1
                    if model.use_frames:
                        initial_frame = int(int(b * 25 * stride)) // 4 * 4
                        if b == 0:
                            
                            featV.append([read_image(os.path.join(path_frames, game_ID, 'half2', 'frame ' + str(max(0, min(int(data['frames2']), initial_frame + int(round((j * 25 / framerate) / 4) * 4)))) + '.jpg')) for j in range(n_frames)])
                            if n_repeated > 0:
                                aux_featV = featV[-1][-n_repeated:]

                        else:
                            #featV.append([cv2.imread(os.path.join(path_frames, game_ID, 'half1', 'frame ' + str(max(0, min(int(frames1), int(initial_frame + j * framestride)))) + '.jpg')) for j in range(n_frames)])
                            if n_repeated == 0:
                                featV.append([read_image(os.path.join(path_frames, game_ID, 'half2', 'frame ' + str(max(0, min(int(data['frames2']), initial_frame + int(round((j * 25 / framerate) / 4) * 4)))) + '.jpg')) for j in range(n_frames)])
                            else:
                                s = [aux_featV.append(read_image(os.path.join(path_frames, game_ID, 'half2', 'frame ' + str(max(0, min(int(data['frames2']), initial_frame + int(round((j * 25 / framerate) / 4) * 4)))) + '.jpg'))) for j in range(n_repeated, n_frames)]
                                featV.append(aux_featV)
                                aux_featV = featV[-1][-n_repeated:]

                spotting_grountruth.append(torch.abs(label_half1))
                spotting_grountruth.append(torch.abs(label_half2))
                spotting_grountruth_visibility.append(label_half1)
                spotting_grountruth_visibility.append(label_half2)
                spotting_predictions.append(timestamp_long_half_1)
                spotting_predictions.append(timestamp_long_half_2)

                batch_time.update(time.time() - end)
                end = time.time()
        
                desc = f'Test (spot.): '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                t.set_description(desc)
                
        
                
                if postprocessing == 'NMS':
                    get_spot = get_spot_from_NMS
                    nms_window = [NMS_window] * 17

                elif postprocessing == 'SNMS':
                    get_spot = get_spot_from_SNMS
                    nms_window = [5, 7, 9, 12, 10, 14, 14, 5, 8, 8, 8, 8, 13, 5, 6, 6, 6]

                elif postprocessing == 'WSF':
                    get_spot = get_spot_from_WSF
                    nms_window = [NMS_window] * 17
        
                json_data = dict()
                json_data["UrlLocal"] = game_ID
                json_data["predictions"] = list()
                #nms_window = [NMS_window] * 17
                for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                            
                    for l in range(dataloader.dataset.num_classes):
                        spots = get_spot(
                            timestamp[:, l], window=nms_window[l]*outputrate, thresh=NMS_threshold)


                        for spot in spots:
                            # print("spot", int(spot[0]), spot[1], spot)
                            frame_index = int(spot[0])
                            confidence = spot[1]
                            # confidence = predictions_half_1[frame_index, l]
        
                            seconds = int((frame_index//outputrate)%60)
                            minutes = int((frame_index//outputrate)//60)
        
                            prediction_data = dict()
                            prediction_data["gameTime"] = str(half+1) + " - " + str(minutes) + ":" + str(seconds)

                            prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]

                            prediction_data["position"] = str(int((frame_index/outputrate)*1000))
                            prediction_data["half"] = str(half+1)
                            prediction_data["confidence"] = str(confidence)
                            json_data["predictions"].append(prediction_data)
                        
                os.makedirs(os.path.join(path_experiments, "ASmodels", model_name, 'post_' + str(postprocessing) + 'window_' + str(NMS_window), output_folder, game_ID), exist_ok=True)
                with open(os.path.join(path_experiments, "ASmodels", model_name, 'post_' + str(postprocessing) + 'window_' + str(NMS_window), output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)

        

        def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])


        # zip folder
        zipResults(zip_path=output_results,
                target_dir = os.path.join(path_experiments, "ASmodels", model_name, 'post_' + str(postprocessing) + 'window_' + str(NMS_window), output_folder),
                filename="results_spotting.json")

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    labels_path = "/data-net/datasets/SoccerNetv2/ResNET_TF2"
    results_l = evaluate(SoccerNet_path=labels_path, 
                Predictions_path=output_results,
                split="test",
                prediction_file="results_spotting.json", 
                version=2,
                metric="loose")   
    
    results_t = evaluate(SoccerNet_path=labels_path, 
                Predictions_path=output_results,
                split="test",
                prediction_file="results_spotting.json", 
                version=2,
                metric="tight")  

    
    return results_l, results_t


def get_spot_from_NMS(Input, window, thresh=0.0, min_window=0):
    """
    Non-Maximum Suppression
    """
    detections_tmp = np.copy(Input)
    # res = np.empty(np.size(Input), dtype=bool)
    indexes = []
    MaxValues = []
    while(np.max(detections_tmp) >= thresh):
        
        # Get the max remaining index and value
        max_value = np.max(detections_tmp)
        max_index = np.argmax(detections_tmp)
                        
        # detections_NMS[max_index,i] = max_value
        
        nms_from = int(np.maximum(-(window/2)+max_index,0))
        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                            
        if (detections_tmp[nms_from:nms_to] >= thresh).sum() > min_window:
            MaxValues.append(max_value)
            indexes.append(max_index)
        detections_tmp[nms_from:nms_to] = -1
        
    return np.transpose([indexes, MaxValues])

def get_spot_from_SNMS(Input, window, thresh=0.0, decay = 'pow2'):
    """
    Soft Non-Maximum Suppression
    """
    detections_tmp = np.copy(Input)

    indexes = []
    MaxValues = []
    while(np.max(detections_tmp) >= thresh):

        # Get the max remaining index and value
        max_value = np.max(detections_tmp)
        max_index = np.argmax(detections_tmp)

        nms_from = int(np.maximum(-(window/2)+max_index,0))
        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)-1)) + 1

        MaxValues.append(max_value)
        indexes.append(max_index)

        if decay == 'linear':
            weight = np.abs(np.arange(nms_from - max_index, nms_to - max_index)) / (window / 2)
        elif decay == 'sqrt':
            weight = np.sqrt(np.abs(np.arange(nms_from - max_index, nms_to - max_index))) / np.sqrt(window / 2)
        elif decay == 'pow2':
            weight = np.power(np.abs(np.arange(nms_from - max_index, nms_to - max_index)), 2) / np.power(window / 2, 2)

        detections_tmp[nms_from:nms_to] = detections_tmp[nms_from:nms_to] * weight
        detections_tmp[nms_from:nms_to][detections_tmp[nms_from:nms_to] < thresh] = -1


    return np.transpose([indexes, MaxValues])

class LearningRateWarmUP(object):
    """
    Class to implement Learning rate warmup
    """
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.step(1)

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration-self.warmup_iteration)
    
    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)