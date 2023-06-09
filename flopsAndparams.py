from MixNetModel import Dilated_UNET
from thop import profile
import torch

model = Dilated_UNET()
input = torch.randn(1, 3, 512, 1024)  # 模型输入的形状,batch_size=1
flops, params = profile(model, inputs=(input,))
print(flops / 1e9, params / 1e6)  # flops单位G，para单位M
