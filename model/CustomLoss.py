import torch.nn.functional as F
import torch

class WeightedL1Loss:
    def __init__(self):
        pass

    def __call__(self, output, target):
        loss = F.l1_loss(output, target, reduction='none') 
        size = target.size()
        center_x, center_y = int(size[2] / 2), int(size[3] / 2)
        x, y = torch.meshgrid(torch.arange(size[2]), torch.arange(size[3]))
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2) 
        weight = torch.zeros(size)
        weight[distance <= 50] = 1 
        weight = weight / weight.mean() 
        weighted_loss = (loss * weight).mean() 
        return weighted_loss
