import numpy as np
from SoccerNet.Evaluation.ActionSpotting import average_mAP


def pred2vec(predictions, chunk_size = 32, outputrate = 2, threshold = 0.01, target = False, NMS = False, window = 4):
    
    if isinstance(predictions, list):
        b, nf, nc = predictions[0].shape
        predsC = predictions[0]
        predsD = predictions[1]

        max_size = (chunk_size) * outputrate + 1

        list_preds = []

        for l in range(b):
            preds = np.zeros((max_size, 17)) -1
            if target:
                preds = preds +1
            predC = predsC[l].cpu().detach().numpy()
            predD = predsD[l].cpu().detach().numpy()
            for f in range(nf):
                for c in range(nc-1):
                    prob = predC[f, c+1]
                    if prob > threshold:
                        position = int(min(max_size-1, max(0, (f - predD[f, c+1]).round())))
                        preds[position, c] = max(preds[position, c], prob)

            if NMS:
                preds = apply_NMS(preds, window = window, thresh = threshold)
            list_preds.append(preds)

    else:
        max_size = (chunk_size) * outputrate + 1
        b, nf, nc = predictions.shape

        list_preds = []
        predictions = predictions.cpu().detach().numpy()

        for l in range(b):
            preds = np.zeros((max_size, 17)) -1
            if target:
                preds = preds +1
            preds[predictions[l, :, 1:] > threshold] = predictions[l, :, 1:][predictions[l, :, 1:] > threshold]

            if NMS:
                preds = apply_NMS(preds, window = window, thresh = threshold)
            list_preds.append(preds)


    return list_preds

def compute_mAP(preds, labs, framerate = 2, metric = 'tight'):
    #preds = pred2vec(predictions, chunk_size = 32, framerate = 2, threshold = 0)
    #labs = targ2vec(targets, chunk_size = 32, framerate = 2)

    closests_numpy = []
    for i in range(len(preds)):
        closest_numpy = np.zeros(labs[i].shape)-1
        for c in np.arange(labs[i].shape[-1]):
            indexes = np.where(labs[i][:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = labs[i][indexes[i],c]
        closests_numpy.append(closest_numpy)

    if metric == "loose":
        deltas=np.arange(12)*5 + 5
    elif metric == "tight":
        deltas=np.arange(5)*1 + 1

    a_mAP, a_mAP_per_class, _, _, _, _ = average_mAP(labs, preds, closests_numpy, framerate, deltas=deltas)

    return a_mAP, a_mAP_per_class

def apply_NMS(predictions, window, thresh=0.0):

    nf, nc = predictions.shape
    for i in range(nc):
        aux = predictions[:,i]
        aux2 = np.zeros(nf) -1
        while(np.max(aux) >= thresh):
            # Get the max remaining index and value
            max_value = np.max(aux)
            max_index = np.argmax(aux)
            # detections_NMS[max_index,i] = max_value

            nms_from = int(np.maximum(-(window/2)+max_index,0))
            nms_to = int(np.minimum(max_index+int(window/2), len(aux)))

            aux[nms_from:nms_to] = -1
            aux2[max_index] = max_value
        predictions[:,i] = aux2

    return predictions