import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet38d import Net

from tools.torch_utils import flatten, global_average_pooling_2d

class Classifier(Net):
    def __init__(self, classes):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, classes, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]

    def forward(self, x, with_cam=False):
        x = super().forward(x)
        x = self.dropout7(x)

        if not with_cam:
            x = global_average_pooling_2d(x)
            x = self.fc8(x)
            return flatten(x)
        else:
            # CAM
            x = self.fc8(x)
            # features = F.relu(x) # I have to ablation study 
            features = x
            
            # GAP
            logits = global_average_pooling_2d(x, with_flatten=True)
            return logits, features

    def forward_for_cam(self, x):
        x = super().forward(x)

        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
