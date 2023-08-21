import torch
import torch.nn.functional as F
from torch import nn


class LogLikelihoodLoss(torch.nn.Module):
    """
    Class to compute the loglikelihood loss (displacement loss uncertaint-aware)
    """
    def __init__(self, alpha = 0.6, beta = 0.4):
        super(LogLikelihoodLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha

    def forward(self, labelsD, predictionsD):
        '''
        labelsD: b x 18
        predictionsD: b x 2 (with mean in 0 and logvar in 1)
        '''

        rec_loss = self.alpha * (labelsD - predictionsD[:, 0])**2 / (predictionsD[:, 1].exp())
        sup_loss = self.beta * predictionsD[:, 1]
        return rec_loss+sup_loss

class ASTRALoss(torch.nn.Module):
    """
    Class to compute the ASTRA loss (classification + displacement)
    """
    def __init__(self, wC = 1, wD = 1, focal = False, nw = 7, uncertainty = False, uncertainty_mode = 'mse'):
        super(ASTRALoss, self).__init__()

        self.wC = wC
        self.wD = wD
        self.maxpool = nn.MaxPool2d((nw, 1), stride = (1, 1), padding = (nw//2, 0))
        self.uncertainty = uncertainty
        self.uncertainty_mode = uncertainty_mode
        if self.uncertainty & (self.uncertainty_mode == 'loglikelihood'):
            self.loglikelihood = LogLikelihoodLoss(alpha = 0.3)
        self.focal = focal

    def forward(self, labels, predictions, labelsD = None, predictionsD = None):

        b, nf, nc = labels.shape #b x cs*fr+1 x 18

        #Classification loss
        if self.focal:
            lossC = - torch.log(predictions + 7e-05) * labels * (labels - predictions).abs()**1 - torch.log(1 - predictions + 7e-05) * (1 - labels) * (labels - predictions).abs()**1
        else:
            lossC = - torch.log(predictions + 7e-05) * labels - torch.log(1 - predictions + 7e-05) * (1 - labels)
        lossC = lossC.mean() * self.wC

        #Displacement loss
        if len(labelsD[labelsD != 1000]) == 0:
            lossD = torch.tensor(0, device = 'cuda:0')
        else:
            labels_aux = self.maxpool(labels) #auxiliar labels to weight displacement loss by the probability of the class
            if self.uncertainty & (self.uncertainty_mode == 'loglikelihood'):
                lossD = (self.loglikelihood(labelsD[labelsD != 1000], predictionsD[labelsD != 1000]) * labels_aux[labelsD != 1000]).sum() / (b * nf)
            else:
                lossD = ((labelsD[labelsD != 1000] - predictionsD[labelsD != 1000]).pow(2) * labels_aux[labelsD != 1000]).sum() / (b * nf)
            lossD = lossD * self.wD

        return lossC, lossD
