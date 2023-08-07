import torch.nn as nn
import torch

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, l=0.5, softmax = True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.l = l
        self.softmaxloss = nn.CrossEntropyLoss()
 
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        self.softmax = softmax
 
    def forward(self, x, y_pred, labels: torch.Tensor, device="cuda"):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
 
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(labels.device)
        labels_ex = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_ex.eq(classes.expand(batch_size, self.num_classes))
 
        dist = distmat * mask.float()
        centerloss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        if self.softmax:
            softmaxloss = self.softmaxloss(y_pred, labels)
            loss = softmaxloss + self.l * centerloss
        else:
            loss = centerloss
        
        return loss