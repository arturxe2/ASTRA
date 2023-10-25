import torch
from torch.utils.data import Dataset
from SoccerNet.Downloader import getListGames
from tqdm import tqdm
import numpy as np
import logging
import os
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
import json
import pickle
import math
from torchvision.io import read_image

class SoccerNetFrames(Dataset):
    """
    Dataset class for SoccerNet
    """
    def __init__(self, 
                path_labels = "ResNET_TF2",
                path_store = "SoccerNetSamples",
                path_baidu = "Baidu_features",
                path_audio = 'SoccerNetAudio',
                features_baidu = "baidu_soccer_embeddings.npy", split=["train"], chunk_size=32, outputrate = 2, 
                stride = 4, rC = 3, rD = 6, store = True, max_games = 1000):

        self.listGames = getListGames(split)
        self.chunk_size = chunk_size
        self.outputrate = outputrate
        self.path_store = os.path.join(path_store, str(chunk_size) + '_Split' + str(split[0]) + str(len(split)) + '_Outputrate' + str(self.outputrate) + '_rC' + str(rC) + '_rD' + str(rD))
        self.split = split[0]
        self.stride = stride

        #Split clips in groups without overlapping
        self.groups = self.chunk_size // self.stride        
        
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"

        logging.info("Pre-compute clips")
        
        self.game_labels = list()
        
        #Check if store clips (or read them)
        if store:
            
            self.path_list = []

            #Label frequency of each class
            self.freq = np.zeros(self.num_classes+1)

            #Auxiliar variables
            ngame = 0
            nclip = 0

            for game in tqdm(self.listGames):
                ngame += 1
                if ngame == max_games:
                    break
                
                #Load Baidu features
                featB_half1 = np.load(os.path.join(path_baidu, game, '1_' + features_baidu))
                featB_half1 = featB_half1.reshape(-1, featB_half1.shape[-1])
                featB_half2 = np.load(os.path.join(path_baidu, game, '2_' + features_baidu))
                featB_half2 = featB_half2.reshape(-1, featB_half2.shape[-1])

                #Game features to clip features
                featB_half1 = feats2clip(torch.from_numpy(featB_half1), stride = self.stride, clip_length = self.chunk_size)
                featB_half2 = feats2clip(torch.from_numpy(featB_half2), stride = self.stride, clip_length = self.chunk_size)

                #Load audio mel-spectrogram
                featA_half1 = np.load(os.path.join(path_audio, game, 'audio1.npy'))
                featA_half1 = feats2clip(torch.from_numpy(featA_half1).T, stride = self.stride * 100, clip_length = self.chunk_size * 100)
                featA_half2 = np.load(os.path.join(path_audio, game, 'audio2.npy'))
                featA_half2 = feats2clip(torch.from_numpy(featA_half2).T, stride = self.stride * 100, clip_length = self.chunk_size * 100)

                #Load labels
                labels = json.load(open(os.path.join(path_labels, game, self.labels)))

                #Labels target
                label_half1 = np.zeros((featB_half1.shape[0], self.chunk_size * outputrate, self.num_classes+1))
                label_half1[:, :, 0] = 1 #BG classes
                label_half2 = np.zeros((featB_half2.shape[0], self.chunk_size * outputrate, self.num_classes+1))
                label_half2[:, :, 0] = 1 #BG classes

                #Displacement targets
                label_half1_displ = np.zeros((math.ceil(featB_half1.shape[0]), self.chunk_size * outputrate, self.num_classes+1)) + 1000
                label_half2_displ = np.zeros((math.ceil(featB_half2.shape[0]), self.chunk_size * outputrate, self.num_classes+1)) + 1000

                #Update labels iterating the annotations
                for annotation in labels["annotations"]:
    
                    time = annotation["gameTime"]
                    event = annotation["label"]
                    visible = annotation["visibility"]
                    visible = int(visible == 'visible')
    
                    half = int(time[0])
                    position = int(int(annotation["position"]) / 1000 * outputrate) #position with outputrate
    
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]
    
                    # if label outside temporal of view
                    if half == 1 and (position / outputrate) // self.stride >= label_half1.shape[0]:
                        continue
                    if half == 2 and (position / outputrate) // self.stride >= label_half2.shape[0]:
                        continue

                    #Range of clips where the action happens
                    a_max = int((position / outputrate) // self.stride)
                    a_min = max(0, int(((position / outputrate) - self.chunk_size) // self.stride) + 1) 

                    #Iterate over the range
                    for l in range(a_min, a_max+1):
                        position_start = self.stride * l * outputrate

                        #Relative positions action
                        min_rel_position = max(0, (position - position_start - rC * outputrate))
                        max_rel_position = min(self.chunk_size * outputrate, (position - position_start + rC * outputrate))

                        #Relative positions displacement
                        min_rel_position_displ = max(0, (position - position_start - rD * outputrate))
                        max_rel_position_displ = min(self.chunk_size * outputrate, (position - position_start + rD * outputrate))

                        if rD != 0:
                            displ = np.arange(-rD * outputrate + max(0, -(position - position_start - rD * outputrate)), (rD * outputrate + 1/outputrate - max(0, -(position_start + self.chunk_size * outputrate - 1 - position - rD * outputrate))))
                        else:
                            displ = 0

                        if half == 1:
                            label_half1[l, min_rel_position:max_rel_position+1, label+1] = 1
                            label_half1[l, min_rel_position:max_rel_position+1, 0] = 0

                            label_half1_displ[l, min_rel_position_displ:max_rel_position_displ+1, label+1] = displ

                        if half == 2:
                            label_half2[l, min_rel_position:max_rel_position+1, label+1] = 1
                            label_half2[l, min_rel_position:max_rel_position+1, 0] = 0 

                            label_half2_displ[l, min_rel_position_displ:max_rel_position_displ+1, label+1] = displ
                            

                #Check storing path exists
                for j in range(self.chunk_size // self.stride):
                    path = self.path_store + '/group' + str(j)
                    exists = os.path.exists(path)
                    if not exists:
                        os.makedirs(path)              

                #Store half 1 clips data
                j = 0
                for i in range(featB_half1.shape[0]):
                    #Store labels
                    np.save(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_labels.npy', label_half1[i, :, :])
                    #Store displacements
                    np.save(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_labels_displ.npy', label_half1_displ[i, :, :])
                    #Store baidu features
                    np.save(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_featB.npy', featB_half1[i, :, :])
                    #Store audio features
                    np.save(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_featA.npy', featA_half1[i, :, :])

                    #Update path list
                    self.path_list.append(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_')
                    
                    #Update group of clips                    
                    j += 1
                    if j == (self.chunk_size // self.stride):
                        j = 0
                        nclip += 1

                #Store half 2 clips data
                for i in range(featB_half2.shape[0]):
                    #Store labels
                    np.save(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_labels.npy', label_half2[i, :, :])
                    #Store displacements
                    np.save(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_labels_displ.npy', label_half2_displ[i, :, :])
                    #Store baidu features
                    np.save(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_featB.npy', featB_half2[i, :, :])
                    #Store audio features
                    np.save(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_featA.npy', featA_half2[i, :, :])

                    #Update path list
                    self.path_list.append(self.path_store + '/group' + str(j) + '/chunk' + str(nclip) + '_')
                    
                    #Update group of clips                    
                    j += 1
                    if j == (self.chunk_size // self.stride):
                        j = 0
                        nclip += 1   

                #Update class frequencies
                self.freq += label_half1.sum(0).sum(0)
                self.freq += label_half2.sum(0).sum(0)

            #Save clips path list & actions frequency
            with open(self.path_store + '/' + split[0] + '_clip_paths.pkl', 'wb') as f:
                pickle.dump(self.path_list, f)
            with open(self.path_store + '/' + split[0] + '_freq.pkl', 'wb') as f:
                pickle.dump(self.freq, f)

        #If already stored, read them
        else:
            
            #Read clips path list, frames path list & actions frequency
            with open(self.path_store + '/' + split[0] + '_clip_paths.pkl', 'rb') as f:
                self.path_list = pickle.load(f)
            with open(self.path_store + '/' + split[0] + '_freq.pkl', 'rb') as f:
                self.freq = pickle.load(f)

        #Total number of clips
        self.n_clips = len(self.path_list)

        self.freq = self.freq.astype('int')

        #Print nº of clips
        logging.info('Nº of clips:' + str(self.n_clips))

    def __getitem__(self, index):

        #randomly select a group of clips (non overlapping groups)
        if self.split != 'valid':
            group = np.random.randint(0, self.groups)
        else:
            group = 0
        
        #Read clip path
        path = self.path_list[index*self.groups + group]

        #Initialize data dictionary
        data = dict()

        data['labels'] = np.load(path + 'labels.npy')
        data['labels_displ'] = np.load(path + 'labels_displ.npy')
        data['featB'] = np.load(path + 'featB.npy')
        data['featA'] = np.load(path + 'featA.npy')

        return data
    
    def __len__(self):

        return(self.n_clips // self.groups)

class SoccerNetFramesTesting(Dataset):
    """
    Dataset class for SoccerNet (inference / testing)
    """

    def __init__(self, path_labels = "ResNET_TF2",
                path_baidu = "Baidu_features",
                path_audio = 'SoccerNetAudio',
                split=["test"], outputrate=2, chunk_size=4, baidu=True,
                features_baidu = "baidu_soccer_embeddings.npy", audio = False):
        
        self.path_labels = path_labels
        self.path_baidu = path_baidu
        self.path_audio = path_audio

        self.split = split
        self.outputrate = outputrate
        self.chunk_size = chunk_size
        self.baidu = baidu
        self.features_baidu = features_baidu
        self.audio = audio

        self.listGames = getListGames(split)
        
        self.stride = 1 / outputrate

        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"

    def __getitem__(self, index):

        data = dict()

        if self.baidu:
            featB_half1 = np.load(os.path.join(self.path_baidu, self.listGames[index], '1_' + self.features_baidu))
            featB_half2 = np.load(os.path.join(self.path_baidu, self.listGames[index], '2_' + self.features_baidu))

            data['featB1'] = featB_half1
            data['featB2'] = featB_half2
            
        if self.audio:
            featA_half1 = np.load(os.path.join(self.path_audio, self.listGames[index], 'audio1.npy'))
            featA_half2 = np.load(os.path.join(self.path_audio, self.listGames[index], 'audio2.npy'))

            data['featA1'] = featA_half1
            data['featA2'] = featA_half2

        return self.listGames[index], data

    def __len__(self):
        return len(self.listGames)


def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    """
    Auxiliar function to split video features into clips
    """
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
    # for i in torch.arange(0, clip_length):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
        # Not replicate last, but take the clip closest to the end of the video
        # idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

    return feats[idx,...]