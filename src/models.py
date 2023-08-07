import torch.nn as nn
import torch
from mm.MFN import MFN
from mm.Centerloss import CenterLoss
from mm.AttriNet import Attri_classify_net, Attri_classify_net_22
from datasets import MT_CODE22, ATTRI_CODE22, MT_CODE, ATTRI_CODE

class MobileAnoNet_23(nn.Module):
    def __init__(self, in_channels, num_class) -> None:
        super().__init__()
        self.featurenet = MFN(in_channels=in_channels)
        self.attrinet = Attri_classify_net(self.featurenet.out_dim, mtcode=MT_CODE, attricode=ATTRI_CODE)
        self.machinenet = nn.Linear(in_features = self.featurenet.out_dim, out_features = num_class)
        self.closs = CenterLoss(num_class, self.featurenet.out_dim, l = 0.2)

        print("MobileAnoNet Built")

    def forward(self, x, machineid = None, attri = None):

        x = self.featurenet(x)
        y_pred = self.machinenet(x)

        if machineid is None or attri is None:
            return x, y_pred
        else:

            loss = self.closs(x, y_pred, machineid)

            loss += self.attrinet(x, machineid, attri)

            return loss

class MobileAnoNet_22(nn.Module):
    def __init__(self, in_channels, num_class, machines) -> None:
        super().__init__()
        self.featurenet = MFN(in_channels=in_channels)
        self.attrinet = Attri_classify_net_22(self.featurenet.out_dim, mtcode=MT_CODE22, attricode=ATTRI_CODE22)
        self.machinenet = nn.Linear(in_features = self.featurenet.out_dim, out_features = num_class)
        self.closs = CenterLoss(num_class, self.featurenet.out_dim, l = 0.2)

        print("MobileAnoNet Built")

    def forward(self, x, machineid = None, attri = None):

        x = self.featurenet(x)
        y_pred = self.machinenet(x)

        if machineid is None or attri is None:
            return x, y_pred
        else:

            loss = self.closs(x, y_pred, machineid)

            loss += self.attrinet(x, machineid, attri)

            return loss

def build_model(in_channels, n_class, t = "DCASE23"):
    if t == "DCASE23":
        return MobileAnoNet_23(in_channels, n_class)
    elif t == "DCASE22":
        return MobileAnoNet_22(in_channels)

if __name__ == '__main__':
    bs = 3
    rand_image_batch = torch.rand((bs, 1, 1025, 64), dtype=torch.float32, requires_grad=False)
    net = MFN(in_channels=1)
    device = "cuda:7"
    rand_image_batch = rand_image_batch.to(device)
    net.to(device)
    net.eval()
    o = net(rand_image_batch)
    print(o.size())

