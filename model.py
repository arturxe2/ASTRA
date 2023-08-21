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
        n_frames = 24,
        n_output = 24,
        baidu = True,
        audio = False,
        use_frames = False,
        model_cfg = None
    ):

        super().__init__()

        self.chunk_size = chunk_size
        self.n_frames = n_frames
        self.n_output = n_output
        self.baidu = baidu
        self.audio = audio
        self.use_frames = use_frames
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

        #Frames backbone
        if self.use_frames:
            #Normalization
            if model_cfg['frame_normalization'] == 'imagenet':
                means = (0.485, 0.456, 0.406)
                stds = (0.229, 0.224, 0.225)
            elif model_cfg['frame_normalization'] == 'soccernet':
                means = ( 0.3749, 0.4503, 0.2773)
                stds = (0.1699, 0.1880, 0.1866)

            #Preprocessing module (resize + normalize)
            self.preprocessing = nn.Sequential(
                Rearrange('b f c h w -> (b f) c h w'),
                T.Resize((model_cfg['frame_resize'], model_cfg['frame_resize'])),
                T.Normalize(mean=means, std=stds),
                Rearrange('(b f) c h w -> b f c h w', f = self.n_frames)
            )

            #Backbone
            self.backbone = models.__dict__['r3d_18'](weights = 'KINETICS400_V1')
            self.backbone.avgpool = nn.Identity() #avoid avg pooling to keep temporal and spatial dimensions
            self.backbone.fc = nn.Identity() #avoid FC dim reduction
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))

            self.encTF = nn.Parameter(torch.rand(math.ceil(self.n_frames / 8), model_cfg['dim']))
            self.encHF = nn.Parameter(torch.rand(math.ceil(self.model_cfg['frame_resize'] / 16), model_cfg['dim']))
            self.encWF = nn.Parameter(torch.rand(math.ceil(self.model_cfg['frame_resize'] / 16), model_cfg['dim']))
            self.encF = nn.Parameter(torch.rand(model_cfg['dim']))

            #Frames augmentation
            if model_cfg['frame_augmentation']:
                self.augmentationFr = T.Compose([
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
                    T.RandomApply([T.GaussianBlur(5, [.1, 2.])], p=0.5),
                    T.RandomGrayscale(p=0.2),
                    T.RandomHorizontalFlip()
                ])

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


        #Mixup
        if model_cfg['mixup']:
            #Initialize queues
            self.register_buffer('labelQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.n_output, model_cfg['num_classes'] + 1))
            self.labelQ[:, :, :, 0] = 1
            self.register_buffer('labelDQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.n_output, model_cfg['num_classes'] + 1) + 1000)
            if self.baidu:
                self.register_buffer('featBQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.chunk_size, 8576))
            else:
                self.featBQ = None
            if self.audio:
                self.register_buffer('featAQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.chunk_size * 100, 128))
            else:
                self.featAQ = None
            if self.use_frames:
                self.register_buffer('featFQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.n_frames, 3, 224, 398))
            else:
                self.featFQ = None

            self.do_mixup = mixupYolo(alpha = model_cfg['mixup_alpha'], beta = model_cfg['mixup_beta'])

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(weights, checkpoint['epoch']))
            
    def reparametrization_trick(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def forward(self, featsB = None, featsA = None, featsF = None, labels = None, labelsD = None, inference = False):
        
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
            b = len(featsA)
            #log mel spectrogram
            featsA = 10.0 * (torch.log10(torch.maximum(torch.tensor(1e-10), featsA)) - torch.log10(torch.tensor(7430.77))) #7430.77 is the maximum value of the log mel spectrogram
            featsA = torch.maximum(featsA, featsA.max() - 80)
            featsA = rearrange(featsA, 'b f h -> b f h')

        if self.use_frames:
            featsF = featsF.float() / 255. #b f c h w
            b, f, c, h, w = featsF.shape

        #Mixup
        if self.model_cfg['mixup'] & (not inference):
            y = labels.clone()
            yD = labelsD.clone()
            if self.baidu:
                xB = featsB.clone()
            if self.audio:
                xA = featsA.clone()
            if self.use_frames:
                xF = featsF.clone()

            featsB, featsA, featsF, labels, labelsD = self.do_mixup(featsB, self.featBQ, featsA, self.featAQ, featsF, self.featFQ, labels, self.labelQ, labelsD, self.labelDQ)

            #Update mixup queue
            if self.model_cfg['mixup_balanced']:
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
                        if self.use_frames:
                            self.featFQ[i, :] = xF[idx].clone().detach()

                    elif len(aux) > 0:
                        idx1 = torch.randint(0, self.model_cfg['mixup_nqueue'], (len(aux),), device = 'cuda:0')
                        idx = aux[:, 0]
                        self.labelQ[i, idx1] = y[idx].clone().detach()
                        self.labelDQ[i, idx1] = yD[idx].clone().detach()
                        if self.baidu:
                            self.featBQ[i, idx1] = xB[idx].clone().detach()
                        if self.audio:
                            self.featAQ[i, idx1] = xA[idx].clone().detach()
                        if self.use_frames:
                            self.featFQ[i, idx1] = xF[idx].clone().detach()

            #Normal mixup (without balancing with queue)
            else:
                idxs1 = torch.randint(0, self.model_cfg['num_classes']+1, (b,), device = 'cuda:0') #random action to update
                idxs2 = torch.randint(0, self.model_cfg['mixup_nqueue'], (b,), device = 'cuda:0') #random queue to update
                self.labelQ[idxs1, idxs2] = y.clone().detach()
                self.labelDQ[idxs1, idxs2] = yD.clone().detach()
                if self.baidu:
                    self.featBQ[idxs1, idxs2] = xB.clone().detach()
                if self.audio:
                    self.featAQ[idxs1, idxs2] = xA.clone().detach()
                if self.use_frames:
                    self.featFQ[idxs1, idxs2] = xF.clone().detach()
            
            del y
            del yD
            if self.baidu:
                del xB
            if self.audio:
                del xA
            if self.use_frames:
                del xF

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
                
            #VGGish
            fA = featsA.shape[1] // 96 #number of 0.96 segments
            featsA = rearrange(featsA[:, :fA * 96, :], 'b (f ns) h -> (b f) 1 ns h', f = fA) #batch*segments x d x 96 x 128
            featsA = self.vggish_features(featsA)
            featsA = featsA.flatten(1)
            featsA = self.vggish_embeddings(featsA)

            #Positional Encoding
            featsA = rearrange(featsA, '(b f) d -> b f d', f = fA) + self.encTA.expand(b, -1, -1) #batch x segments x d
            featsA += self.encA.expand(b, fA, -1) #batch x segments x d


        #Frames
        if self.use_frames:
            #Frame Augmentation
            if self.model_cfg['frame_augmentation'] & (not inference):
                for i in range(b):
                    featsF[i] = self.augmentationFr(featsF[i])
            featsF = self.preprocessing(featsF) #b f c h w
            b, f, c, h, w = featsF.shape

            #ResNet3d
            ns = f // 16 #number of 16-frame segments
            featsF_aux1 = rearrange(featsF[:, :(ns-1) * 16], 'b (ns ls) c h w -> (b ns) c ls h w', ns = ns - 1, ls = 16) #b*segments x c ls h w
            featsF_aux2 = rearrange(featsF[:, (ns-1) * 16:], 'b ls c h w -> b c ls h w') #b x c ls h w
            featsF_aux1 = rearrange(self.backbone(featsF_aux1), '(b ns) (d f h w) -> b d (ns f) h w', b = b, d = self.model_cfg['dim'], f = int(16 / 8), h = math.ceil(h / 16), w = math.ceil(w / 16)) #b*segments x (d f' h' w')
            featsF_aux2 = rearrange(self.backbone(featsF_aux2), 'b (d f h w) -> b d f h w', d = self.model_cfg['dim'], f = math.ceil((f - (ns-1) * 16) / 8), h = math.ceil(h / 16), w = math.ceil(w / 16))
            featsF = torch.cat((featsF_aux1, featsF_aux2), dim = 2) #b x d f h w
            
            #Positional Encoding
            b, d, f, h, w = featsF.shape
            featsF = rearrange(featsF, 'b d f h w -> b h w f d') + self.encTF.expand(b, h, w, -1, -1) #b h w f d #Temporal
            featsF = rearrange(featsF, 'b h w f d -> b f h w d') + self.encWF.expand(b, f, w, -1, -1) #Spatial1
            featsF = rearrange(featsF, 'b f h w d -> b f w h d') + self.encHF.expand(b, f, h, -1, -1) #Spatial2
            featsF += self.encF.expand(b, f, w, h, -1) #Feature encoding
            featsF = rearrange(featsF, 'b f w h d -> b (f h w) d')

        #Transformer encoder
        x = None
        if not self.model_cfg['TE_hierarchical']:
            if self.baidu:
                x = featsB
            if self.audio:
                if x != None:
                    x = torch.cat((x, featsA), dim = 1)
                else:
                    x = featsA
            if self.use_frames:
                if x != None:
                    x = torch.cat((x, featsF), dim = 1)
                else:
                    x = featsF
            
            x = self.Tencoder(x)
        
        else:
            if self.baidu:
                nB = featsB.shape[1]
            else:
                nB = 0
            if self.audio:
                nA = featsA.shape[1]
            else:
                nA = 0
            if self.use_frames:
                nF = featsF.shape[1]
            else:
                nF = 0

            splits = [4, 2, 1, 1, 1]
            for i, mod in enumerate(self.Tencoder.layers):
                x = None
                if splits[i] != 1:
                    #Split into segments
                    if self.baidu:
                        xB = rearrange(featsB[:, :(splits[i]-1) * (nB // splits[i])], 'b (ns ls) d -> (b ns) ls d', ns = splits[i] - 1)
                        xB_aux = featsB[:, (splits[i]-1) * (nB // splits[i]):]

                        x = xB
                        x_aux = xB_aux
                    if self.audio:
                        xA = rearrange(featsA[:, :(splits[i]-1) * (nA // splits[i])], 'b (ns ls) d -> (b ns) ls d', ns = splits[i] - 1)
                        xA_aux = featsA[:, (splits[i]-1) * (nA // splits[i]):]

                        if x != None:
                            x = torch.cat((x, xA), dim = 1)
                            x_aux = torch.cat((x_aux, xA_aux), dim = 1)
                        else:
                            x = xA
                            x_aux = xA_aux
                    if self.use_frames:
                        xF = rearrange(featsF[:, :(splits[i]-1) * (nF // splits[i])], 'b (ns ls) d -> (b ns) ls d', ns = splits[i] - 1)
                        xF_aux = featsF[:, (splits[i]-1) * (nF // splits[i]):]

                        if x != None:
                            x = torch.cat((x, xF), dim = 1)
                            x_aux = torch.cat((x_aux, xF_aux), dim = 1)
                        else:
                            x = xF
                            x_aux = xF_aux

                    #Apply TE layer to segments
                    x = mod(x)
                    x_aux = mod(x_aux)

                    #Concatenate segments
                    if self.baidu:
                        xB = torch.cat((rearrange(x[:, :(nB // splits[i])], '(b ns) ls d -> b (ns ls) d', ns = splits[i] - 1), x_aux[:, :(nB - (nB // splits[i] * (splits[i] - 1)))]), dim = 1)
                        featsB = xB

                    if self.audio:
                        xA = torch.cat((rearrange(x[:, (nB // splits[i]):(nB // splits[i]) + (nA // splits[i])], '(b ns) ls d -> b (ns ls) d', ns = splits[i] - 1), x_aux[:, (nB - (nB // splits[i] * (splits[i] - 1))):(nB - (nB // splits[i] * (splits[i] - 1))) + (nA - (nA // splits[i] * (splits[i] - 1)))]), dim = 1)
                        featsA = xA

                    if self.use_frames:
                        xF = torch.cat((rearrange(x[:, (nB // splits[i]) + (nA // splits[i]):], '(b ns) ls d -> b (ns ls) d', ns = splits[i] - 1), x_aux[:, (nB - (nB // splits[i] * (splits[i] - 1))) + (nA - (nA // splits[i] * (splits[i] - 1))):]), dim = 1)
                        featsF = xF

                else:
                    #Concatenate all features
                    if self.baidu:
                        x = featsB
                    if self.audio:
                        if x != None:
                            x = torch.cat((x, featsA), dim = 1)
                        else:
                            x = featsA
                    if self.use_frames:
                        if x != None:
                            x = torch.cat((x, featsF), dim = 1)
                        else:
                            x = featsF
                    
                    #Apply TE layer
                    x = mod(x)

                    #Split into features
                    if self.baidu:
                        featsB = x[:, :(nB)]
                    if self.audio:
                        featsA = x[:, (nB):(nB) + (nA)]
                    if self.use_frames:
                        featsF = x[:, (nB) + (nA):]
            
            x = None
            if self.baidu:
                x = featsB
            if self.audio:
                if x != None:
                    x = torch.cat((x, featsA), dim = 1)
                else:
                    x = featsA
            if self.use_frames:
                if x != None:
                    x = torch.cat((x, featsF), dim = 1)
                else:
                    x = featsF
        
        #Transformer decoder
        queries = self.queries.expand((b, -1, -1)).cuda()
        x = self.Tdecoder(queries, x)

        #Classification head
        y1 = self.clas_head(x)
        y2 = self.displ_head(x)

        if self.model_cfg['uncertainty'] & (self.model_cfg['uncertainty_mode'] == 'mse') & (not inference):
            y2 = self.reparametrization_trick(y2[:, :, :, 0], y2[:, :, :, 1])
        elif (self.model_cfg['uncertainty_mode'] == 'mse') & inference:
            y2 = y2[:, :, :, 0]

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
            
    def forward(self, featB, featBQ, featA, featAQ, featF, featFQ, labels, labelsQ, labelsD, labelsDQ):
        #len of batch
        b = len(labels)

        #same lambda for all the batch
        lamb = self.betaD.sample()

        
        #Index of action and nqueue to do mixup
        idxa = torch.randint(0, 18, (b,))
        idxnq = torch.randint(0, 2, (b,))

        #Mixture

        if featB != None:
            featB = featB * lamb + (1-lamb) * featBQ[idxa, idxnq]
        if featA != None:
            featA = featA * lamb + (1-lamb) * featAQ[idxa, idxnq]
        if featF != None:
            featF = featF * lamb + (1-lamb) * featFQ[idxa, idxnq]
        if labels != None:
            labels = labels * lamb + (1-lamb) * labelsQ[idxa, idxnq]
        if labelsD != None:
            labelsD = ((labelsD == 1000) & (labelsDQ[idxa, idxnq] == 1000)) * 1000 + ((labelsD == 1000) & (labelsDQ[idxa, idxnq] != 1000)) * labelsDQ[idxa, idxnq] + ((labelsD != 1000) & (labelsDQ[idxa, idxnq] == 1000)) * labelsD + ((labelsD != 1000) & (labelsDQ[idxa, idxnq] != 1000)) * (labelsD * lamb + (1-lamb) * labelsDQ[idxa, idxnq])

        return featB, featA, featF, labels, labelsD
        
# Augmentation modules

class random_noiseM(nn.Module):
    def __init__(self, mean = 0, std = 0.05, norm = False, dim = 8576):
        super().__init__()
        self.mean = mean
        self.std = std
        self.norm = norm
        if self.norm:
            self.bnorm = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor):
        if self.norm:
            x = self.bnorm(x.transpose(1, 2)).transpose(1, 2)
        noise = (torch.randn(x.shape) * self.std + self.mean).cuda()
        return x + noise
    
class temporal_shiftM(nn.Module):
    def __init__(self, max_shift = 5):
        super().__init__()
        self.max_shift = max_shift
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, yD: torch.Tensor):
        shift = torch.randint(high = self.max_shift, size = (1, ))
        x = torch.roll(x, int(shift), 1)
        y = torch.roll(y, int(shift), 1)
        yD = torch.roll(yD, int(shift), 1)
        return x, y, yD
    
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