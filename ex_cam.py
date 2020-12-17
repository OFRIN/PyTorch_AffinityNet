import torch

from core.resnet38_cls import Classifier

pascal_voc_model = Classifier(20)

# images = torch.randn([1, 3, 448, 448])
images = torch.randn([1, 3, 224, 224])
cams = pascal_voc_model.forward_for_cam(images)

print(cams.size()) # [1, 20, 56, 56]