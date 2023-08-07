import torch.nn as nn
import torch

class LNlossNet(nn.Module):
    def __init__(self, fdim: int, n_class: int) -> None:
        super().__init__()
        self.linear = nn.Linear(fdim, n_class, bias=False)
        self.bn = nn.BatchNorm1d(n_class)
        self.lossfunc = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.linear(x)
        x = self.bn(x)
        loss = self.lossfunc(x, y)
        return loss

class Attri_classify_net(nn.Module):
    def __init__(self, fdim: int, mtcode, attricode) -> None:
        super().__init__()

        self.mtcode = mtcode
        self.attricode = attricode

        self.bearing_vel = LNlossNet(fdim, len(self.attricode['bearing']['vel']))
        self.bearing_loc = LNlossNet(fdim, len(self.attricode['bearing']['loc']))

        self.fan_mn = LNlossNet(fdim, len(self.attricode['fan']['m-n']))

        self.gearbox_volt = LNlossNet(fdim, len(self.attricode['gearbox']['volt']))
        self.gearbox_wt = LNlossNet(fdim, len(self.attricode['gearbox']['wt']))

        self.toycar_car = LNlossNet(fdim, len(self.attricode['ToyCar']['car']))
        self.toycar_spd = LNlossNet(fdim, len(self.attricode['ToyCar']['spd']))
        # self.toycar_mic = LNlossNet(fdim, len(self.attricode['ToyCar']['mic']))

        self.toytrain_car = LNlossNet(fdim, len(self.attricode['ToyTrain']['car']))
        self.toytrain_spd = LNlossNet(fdim, len(self.attricode['ToyTrain']['spd']))
        # self.toytrain_mic = LNlossNet(fdim, len(self.attricode['ToyTrain']['mic']))

        self.slider_vel = LNlossNet(fdim, len(self.attricode['slider']['vel']))
        self.slider_ac = LNlossNet(fdim, len(self.attricode['slider']['ac']))

        self.valve_pat = LNlossNet(fdim, len(self.attricode['valve']['pat']))

        self.bandsaw_vel = LNlossNet(fdim, len(self.attricode['bandsaw']['vel']))

        self.grinder_grdstone = LNlossNet(fdim, len(self.attricode['grinder']['grindstone']))
        self.grinder_plate = LNlossNet(fdim, len(self.attricode['grinder']['plate']))

        self.shaker_spd = LNlossNet(fdim, len(self.attricode['shaker']['speed']))

        self.ToyDrone_car = LNlossNet(fdim, len(self.attricode['ToyDrone']['car']))
        self.ToyDrone_spd = LNlossNet(fdim, len(self.attricode['ToyDrone']['spd']))

        self.ToyNscale_car = LNlossNet(fdim, len(self.attricode['ToyNscale']['car']))
        self.ToyNscale_spd = LNlossNet(fdim, len(self.attricode['ToyNscale']['spd']))

        self.ToyTank_car = LNlossNet(fdim, len(self.attricode['ToyTank']['car']))
        self.ToyTank_spd = LNlossNet(fdim, len(self.attricode['ToyTank']['spd']))

        self.Vacuum_car = LNlossNet(fdim, len(self.attricode['Vacuum']['car']))
        self.Vacuum_spd = LNlossNet(fdim, len(self.attricode['Vacuum']['spd']))

        self.num_attr = 24

    def forward(self, x, machineid, attri):
        loss = 0.

        index_bearing = machineid == self.mtcode['bearing']
        if index_bearing.size(0) > 1:
            loss += self.bearing_vel(x[index_bearing], attri[0][index_bearing])
            loss += self.bearing_loc(x[index_bearing], attri[1][index_bearing])

        index_fan = machineid == self.mtcode['fan']
        if index_fan.size(0) > 1:
            loss += self.fan_mn(x[index_fan], attri[0][index_fan])

        index_gearbox = machineid == self.mtcode['gearbox']
        if index_gearbox.size(0) > 1:
            loss += self.gearbox_volt(x[index_gearbox], attri[0][index_gearbox])
            loss += self.gearbox_wt(x[index_gearbox], attri[1][index_gearbox])

        index_slider = machineid == self.mtcode['slider']
        if index_slider.size(0) > 1:
            loss += self.slider_vel(x[index_slider], attri[0][index_slider])
            loss += self.slider_ac(x[index_slider], attri[1][index_slider])

        index_toycar = machineid == self.mtcode['ToyCar']
        if index_toycar.size(0) > 1:
            loss += self.toycar_car(x[index_toycar], attri[0][index_toycar])
            loss += self.toycar_spd(x[index_toycar], attri[1][index_toycar])
            # loss += self.toycar_mic(x[index_toycar], attri[2][index_toycar])

        index_toytrain = machineid == self.mtcode['ToyTrain']
        if index_toytrain.size(0) > 1:
            loss += self.toytrain_car(x[index_toytrain], attri[0][index_toytrain])
            loss += self.toytrain_spd(x[index_toytrain], attri[1][index_toytrain])
            # loss += self.toytrain_mic(x[index_toytrain], attri[2][index_toytrain])

        index_valve = machineid == self.mtcode['valve']
        if index_valve.size(0) > 1:
            loss += self.valve_pat(x[index_valve], attri[0][index_valve])

        index_bandsaw = machineid == self.mtcode['bandsaw']
        if index_bandsaw.size(0) > 1:
            loss += self.bandsaw_vel(x[index_bandsaw], attri[0][index_bandsaw])

        index_grinder = machineid == self.mtcode['grinder']
        if index_grinder.size(0) > 1:
            loss += self.grinder_grdstone(x[index_grinder], attri[0][index_grinder])
            loss += self.grinder_plate(x[index_grinder], attri[1][index_grinder])

        index_shaker = machineid == self.mtcode['shaker']
        if index_shaker.size(0) > 1:
            loss += self.shaker_spd(x[index_shaker], attri[0][index_shaker])

        index_ToyDrone = machineid == self.mtcode['ToyDrone']
        if index_ToyDrone.size(0) > 1:
            loss += self.ToyDrone_car(x[index_ToyDrone], attri[0][index_ToyDrone])
            loss += self.ToyDrone_spd(x[index_ToyDrone], attri[1][index_ToyDrone])

        index_ToyNscale = machineid == self.mtcode['ToyNscale']
        if index_ToyNscale.size(0) > 1:
            loss += self.ToyNscale_car(x[index_ToyNscale], attri[0][index_ToyNscale])
            loss += self.ToyNscale_spd(x[index_ToyNscale], attri[1][index_ToyNscale])

        index_ToyTank = machineid == self.mtcode['ToyTank']
        if index_ToyTank.size(0) > 1:
            loss += self.ToyTank_car(x[index_ToyTank], attri[0][index_ToyTank])
            loss += self.ToyTank_spd(x[index_ToyTank], attri[1][index_ToyTank])

        index_Vacuum = machineid == self.mtcode['Vacuum']
        if index_Vacuum.size(0) > 1:
            loss += self.Vacuum_car(x[index_Vacuum], attri[0][index_Vacuum])
            loss += self.Vacuum_spd(x[index_Vacuum], attri[1][index_Vacuum])
        

        return loss / self.num_attr
    
class Attri_classify_net_22(nn.Module):
    def __init__(self, fdim: int, mtcode, attricode) -> None:
        super().__init__()
        
        self.mtcode = mtcode
        self.attricode = attricode
        
        self.aLN0 = LNlossNet(fdim, len(self.attricode['bearing']['vel']))
        self.aLN1 = LNlossNet(fdim, len(self.attricode['bearing']['loc']))
        self.aLN2 = LNlossNet(fdim, len(self.attricode['bearing']['f-n']))
        
        self.bLN0 = LNlossNet(fdim, len(self.attricode['fan']['m-n']))
        self.bLN1 = LNlossNet(fdim, len(self.attricode['fan']['f-n']))
        self.bLN2 = LNlossNet(fdim, len(self.attricode['fan']['n-lv']))
        
        self.cLN0 = LNlossNet(fdim, len(self.attricode['gearbox']['volt']))
        self.cLN1 = LNlossNet(fdim, len(self.attricode['gearbox']['wt']))
        self.cLN2 = LNlossNet(fdim, len(self.attricode['gearbox']['id']))
        
        self.dLN0 = LNlossNet(fdim, len(self.attricode['slider']['vel']))
        self.dLN1 = LNlossNet(fdim, len(self.attricode['slider']['ac']))
        self.dLN2 = LNlossNet(fdim, len(self.attricode['slider']['f-n']))
        
        self.eLN0 = LNlossNet(fdim, len(self.attricode['ToyCar']['car']))
        self.eLN1 = LNlossNet(fdim, len(self.attricode['ToyCar']['speed']))
        self.eLN2 = LNlossNet(fdim, len(self.attricode['ToyCar']['mic']))
        
        self.fLN0 = LNlossNet(fdim, len(self.attricode['ToyTrain']['car']))
        self.fLN1 = LNlossNet(fdim, len(self.attricode['ToyTrain']['speed']))
        self.fLN2 = LNlossNet(fdim, len(self.attricode['ToyTrain']['mic']))
        
        self.gLN0 = LNlossNet(fdim, len(self.attricode['valve']['pat']))
        self.gLN1 = LNlossNet(fdim, len(self.attricode['valve']['panel']))
        self.gLN2 = LNlossNet(fdim, len(self.attricode['valve']['v1pat']))
            
        self.mapp = {0: self.aLN0, 1: self.aLN1, 2: self.aLN2, 
                     3: self.bLN0, 4: self.bLN1, 5: self.bLN2,
                     6: self.cLN0, 7: self.cLN1, 8: self.cLN2, 
                     9: self.dLN0, 10: self.dLN1, 11: self.dLN2, 
                     12: self.eLN0, 13: self.eLN1, 14: self.eLN2,
                     15: self.fLN0, 16: self.fLN1, 17: self.fLN2, 
                     18: self.gLN0, 19: self.gLN1, 20: self.gLN2
                     }
            
    def forward(self, x, sectionid, attri):
        loss = 0.
        for i in range(21):
            index = sectionid == i
            if index.size(0) > 2:
                loss += self.mapp[i](x[index], attri[0][index])
        
        loss /= 21
        
        return loss