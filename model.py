import math
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange
from einops.layers.torch import Rearrange
import torchvision.models.video as models


class ASTRA(nn.Module):
    """
    ASTRA model class
    """
    def __init__(
        self,
        chunk_size = 8,
        n_output = 24,
        baidu = True,
        audio = False,
        model_cfg = None
    ):

        super().__init__()

        self.chunk_size = chunk_size
        self.n_output = n_output
        self.baidu = baidu
        self.audio = audio
        self.model_cfg = model_cfg

        #Baidu backbone
        if self.baidu:
            self.Bfeat_dim = [2048, 2048, 384, 2048, 2048]
            self.baidu_LL = nn.ModuleList([Bfeat_module(self.Bfeat_dim[i], model_cfg['dim'], drop = model_cfg['dropout']) for i in range(len(self.Bfeat_dim))])
            self.encTB = nn.Parameter(torch.rand(self.chunk_size, model_cfg['dim']))
            self.encFB = nn.Parameter(torch.rand(len(self.Bfeat_dim), model_cfg['dim']))

        #Audio backbone
        if self.audio:
            self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.vggish_features = self.vggish.features
            self.vggish_embeddings = self.vggish.embeddings
            self.vggish_embeddings[0] = nn.Linear(24576, 4096)
            self.vggish_embeddings[4] = nn.Linear(4096, model_cfg['dim'])
            self.encTA = nn.Parameter(torch.rand(self.chunk_size * 100 // 96, model_cfg['dim']))
            self.encA = nn.Parameter(torch.rand(model_cfg['dim']))
            del self.vggish

        #Features augmentation (for both audio + baidu)
        if model_cfg['feature_augmentation']:
            if self.baidu:
                self.temporal_dropB = temporal_dropM(p = model_cfg['temporal_drop_p'])
            if self.audio:
                self.temporal_dropA = temporal_dropM(p = model_cfg['temporal_drop_p'], dim = 128)
            self.random_switch = random_switchM(p = model_cfg['random_switch_p'])

        #ASTRA model

        #Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model = model_cfg['dim'], nhead = 8, dim_feedforward = model_cfg['dim'] * 4, batch_first = True)
        self.Tencoder = nn.TransformerEncoder(encoder_layer, model_cfg['TE_layers'])

        #Transformer decoder
        self.queries = nn.Parameter(torch.rand((self.n_output, model_cfg['dim'])))
        decoder_layer = nn.TransformerDecoderLayer(d_model = model_cfg['dim'], nhead = 8, dim_feedforward = model_cfg['dim'] * 4, batch_first = True)
        self.Tdecoder = nn.TransformerDecoder(decoder_layer, model_cfg['TD_layers'])

        #Prediction heads
        self.clas_head = pred_head(model_cfg['dim'], model_cfg['num_classes'] + 1, sigmoid = True, drop = model_cfg['dropout'])

        if model_cfg['uncertainty']:
            self.displ_head = uncertainty_head(model_cfg['dim'], model_cfg['num_classes'] + 1, drop = model_cfg['dropout'])
        else:
            self.displ_head = pred_head(model_cfg['dim'], model_cfg['num_classes'] + 1, sigmoid = False, drop = model_cfg['dropout'])


        #Mixup queue system
        if model_cfg['mixup']:
            #Initialize queues (for labels and inputs)
            self.register_buffer('labelQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.n_output, model_cfg['num_classes'] + 1))
            self.labelQ[:, :, :, 0] = 1
            self.register_buffer('labelDQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.n_output, model_cfg['num_classes'] + 1) + 1000)

            self.register_buffer('featBQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.chunk_size, 8576))
            if self.audio:
                self.register_buffer('featAQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.chunk_size * 100, 128))
            else:
                self.featAQ = None

            self.do_mixup = mixupYolo(alpha = model_cfg['mixup_alpha'], beta = model_cfg['mixup_beta'])

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(weights, checkpoint['epoch']))

    def forward(self, featsB = None, featsA = None, labels = None, labelsD = None, inference = False):
        
        #Float type data
        if labels != None:
            labels = labels.float()
        if labelsD != None:
            labelsD = labelsD.float()
        if self.baidu:
            featsB = featsB.float()
            b = len(featsB)

        if self.audio:
            featsA = featsA.float()
            #log mel spectrogram
            featsA = 10.0 * (torch.log10(torch.maximum(torch.tensor(1e-10), featsA)) - torch.log10(torch.tensor(7430.77))) #7430.77 is the maximum value of the log mel spectrogram
            featsA = torch.maximum(featsA, featsA.max() - 80)
            featsA = rearrange(featsA, 'b f h -> b f h')

        #Mixup (not inference)
        if self.model_cfg['mixup'] & (not inference):
            y = labels.clone()
            yD = labelsD.clone()
            if self.baidu:
                xB = featsB.clone()
            if self.audio:
                xA = featsA.clone()

            featsB, featsA, labels, labelsD = self.do_mixup(featsB, self.featBQ, featsA, self.featAQ, labels, self.labelQ, labelsD, self.labelDQ)

            #Update mixup queue
            batch_action = (y[:, :, :] == 1).sum(1).nonzero() #(batch, action) pairs of clip with action
                
            for i in range(self.model_cfg['num_classes']+1):
                aux = batch_action[batch_action[:, 1] == i] #idxs containing action i

                if len(aux) >= self.model_cfg['mixup_nqueue']:
                    idx = aux[:self.model_cfg['mixup_nqueue'], 0] #keep first ones
                    self.labelQ[i, :] = y[idx].clone().detach()
                    self.labelDQ[i, :] = yD[idx].clone().detach()
                    if self.baidu:
                        self.featBQ[i, :] = xB[idx].clone().detach()
                    if self.audio:
                        self.featAQ[i, :] = xA[idx].clone().detach()

                elif len(aux) > 0:
                    idx1 = torch.randint(0, self.model_cfg['mixup_nqueue'], (len(aux),), device = 'cuda:0')
                    idx = aux[:, 0]
                    self.labelQ[i, idx1] = y[idx].clone().detach()
                    self.labelDQ[i, idx1] = yD[idx].clone().detach()
                    if self.baidu:
                        self.featBQ[i, idx1] = xB[idx].clone().detach()
                    if self.audio:
                        self.featAQ[i, idx1] = xA[idx].clone().detach()

        #DATA AUGMENTATIONS + PREPROCESSING BEFORE TE (INCLUDING POSITIONAL ENCODING)

        #Baidu features
        if self.baidu:

            #Feature Augmentation
            if self.model_cfg['feature_augmentation'] & (not inference):
                featsB = self.temporal_dropB(featsB)
                featsB = self.random_switch(featsB)
            #PFFN
            featsB = [self.baidu_LL[i](featsB[:, :, int(torch.tensor(self.Bfeat_dim[:i]).sum()):int(torch.tensor(self.Bfeat_dim[:i+1]).sum())]) for i in range(len(self.Bfeat_dim))]
            featsB = torch.stack(featsB) #5 x b x cs x d
            #Positional Encoding
            l, b, cs, d = featsB.shape
            featsB += self.encTB.expand(l, b, -1, -1)
            featsB = rearrange(featsB, 'l b cs d -> b cs l d') + self.encFB.expand(b, cs, -1, -1)
            featsB = rearrange(featsB, 'b cs l d -> b (cs l) d')

        #Audio features
        if self.audio:

            #Feature Augmentation
            if self.model_cfg['feature_augmentation'] & (not inference):
                featsA = self.temporal_dropA(featsA)
                featsA = self.random_switch(featsA)
            #VGGish backbone
            fA = featsA.shape[1] // 96 #number of 0.96 segments
            featsA = rearrange(featsA[:, :fA * 96, :], 'b (f ns) h -> (b f) 1 ns h', f = fA) #batch*segments x d x 96 x 128
            featsA = self.vggish_features(featsA)
            featsA = featsA.flatten(1)
            featsA = self.vggish_embeddings(featsA)
            #Positional Encoding
            featsA = rearrange(featsA, '(b f) d -> b f d', f = fA) + self.encTA.expand(b, -1, -1) #batch x segments x d
            featsA += self.encA.expand(b, fA, -1) #batch x segments x d

        #Transformer encoder
        nB = featsB.shape[1]
        nA = 0
        if self.audio:
            nA = featsA.shape[1]

        splits = [4, 2, 1, 1, 1] #hierarchical architecture of TE (split in 4 segments at first layer, 2 at second layer...)
        
        for i, mod in enumerate(self.Tencoder.layers):

            #global attention layers (i.e. split == 1 - 1 segment)
            if splits[i] == 1:
                x = featsB
                if self.audio:
                    x = torch.cat((x, featsA), dim = 1)
                x = mod(x)

                featsB = x[:, :(nB)]
                if self.audio:
                    featsA = x[:, (nB):]

            #hierarchical attention layers (i.e. split > 1 - split in segments)
            else: 
                x = rearrange(featsB[:, :(splits[i]-1) * (nB // splits[i])], 'b (ns ls) d -> (b ns) ls d', ns = splits[i] - 1) #split in segments 
                x_aux = featsB[:, (splits[i]-1) * (nB // splits[i]):] #residual segment

                if self.audio:
                    xA = rearrange(featsA[:, :(splits[i]-1) * (nA // splits[i])], 'b (ns ls) d -> (b ns) ls d', ns = splits[i] - 1) #split in segments
                    xA_aux = featsA[:, (splits[i]-1) * (nA // splits[i]):] #residual segment

                    x = torch.cat((x, xA), dim = 1) #merge visual and audio features
                    x_aux = torch.cat((x_aux, xA_aux), dim = 1) #merge residual segments

                #Apply TE layer to segments
                x = mod(x)
                x_aux = mod(x_aux)

                #Restore in original shape and features distinction
                xB = torch.cat((rearrange(x[:, :(nB // splits[i])], '(b ns) ls d -> b (ns ls) d', ns = splits[i] - 1), x_aux[:, :(nB - (nB // splits[i] * (splits[i] - 1)))]), dim = 1)
                featsB = xB

                if self.audio:
                    xA = torch.cat((rearrange(x[:, (nB // splits[i]):(nB // splits[i]) + (nA // splits[i])], '(b ns) ls d -> b (ns ls) d', ns = splits[i] - 1), x_aux[:, (nB - (nB // splits[i] * (splits[i] - 1))):(nB - (nB // splits[i] * (splits[i] - 1))) + (nA - (nA // splits[i] * (splits[i] - 1)))]), dim = 1)
                    featsA = xA

        #Transformer decoder
        queries = self.queries.expand((b, -1, -1)).cuda()
        x = self.Tdecoder(queries, x)

        #Classification head
        y1 = self.clas_head(x)
        y2 = self.displ_head(x)

        output = dict()
        output['preds'] = y1
        output['predsD'] = y2
        output['labels'] = labels
        output['labelsD'] = labelsD

        return output



class pred_head(nn.Module):
    """
    Standard prediction head
    """
    def __init__(self, input_dim, output_dim, sigmoid = True, drop = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(input_dim, output_dim)
        )
        if sigmoid:
            self.head.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        return self.head(x) 
    
class uncertainty_head(nn.Module):
    """
    Uncertainty-aware prediction head
    """
    def __init__(self, input_dim, output_dim, drop = 0.2):
        super().__init__()
        self.shared_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, output_dim)
        )
        self.logvar_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):

        x = self.shared_head(x)
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        x = torch.stack((mean, logvar), dim = 3)
        return x

class Bfeat_module(nn.Module):
    def __init__(self, input_dim, output_dim, drop = 0.2):
        super().__init__()
        self.head = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(input_dim, output_dim),
                nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        return self.head(x)

class mixupYolo(torch.nn.Module):
    """
    Mixup module class
    """
    def __init__(self, alpha = 0.3, beta = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.betaD = torch.distributions.beta.Beta(alpha, beta)
        self.n_queues = 2
            
    def forward(self, featB, featBQ, featA, featAQ, labels, labelsQ, labelsD, labelsDQ):
        #len of batch
        b = len(labels)
        classes = labels.shape[-1]

        #same lambda for all the batch
        lamb = self.betaD.sample()

        #Index of action and nqueue to do mixup
        idxa = torch.randint(0, classes, (b,))
        idxnq = torch.randint(0, self.n_queues, (b,))

        #Mixture
        if featB != None:
            featB = featB * lamb + (1-lamb) * featBQ[idxa, idxnq]
        if featA != None:
            featA = featA * lamb + (1-lamb) * featAQ[idxa, idxnq]
        if labels != None:
            labels = labels * lamb + (1-lamb) * labelsQ[idxa, idxnq]
        if labelsD != None:
            labelsD = ((labelsD == 1000) & (labelsDQ[idxa, idxnq] == 1000)) * 1000 + ((labelsD == 1000) & (labelsDQ[idxa, idxnq] != 1000)) * labelsDQ[idxa, idxnq] + ((labelsD != 1000) & (labelsDQ[idxa, idxnq] == 1000)) * labelsD + ((labelsD != 1000) & (labelsDQ[idxa, idxnq] != 1000)) * (labelsD * lamb + (1-lamb) * labelsDQ[idxa, idxnq])

        return featB, featA, labels, labelsD
        
# Augmentation modules
class temporal_dropM(nn.Module):
    def __init__(self, p = 0.0, dim = 8576):
        super().__init__()
        self.p = p
        self.embedding = nn.Parameter(torch.rand(dim))
        
    def forward(self, x: torch.Tensor):
        x_aux = x.clone()
        mask = torch.rand(x_aux.shape[1]) < self.p
        x_aux[:, mask] = self.embedding
        return x_aux
    
class random_switchM(nn.Module):
    def __init__(self, p = 0.0):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor):
        x_aux = x.clone()
        idxs = torch.arange(x_aux.shape[1]-1)[torch.rand(x_aux.shape[1]-1) < self.p]
        x_aux[:, idxs, :], x_aux[:, idxs+1, :] = x_aux[:, idxs+1, :], x_aux[:, idxs, :]
        return x_aux