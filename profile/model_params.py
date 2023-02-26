# import torchvision.models as models
# import torch
# from torch.utils.tensorboard import SummaryWriter
# # torchinfo 打印模型结构 输出shape 参数量
#
# #
# # # 如果将pretrained设置为True, 意味着直接得到一个加载了预训练参数的vgg19模型。
# # # 就会自动下载vgg19的参数文件并放在本地缓存中。所以不用提供本地参数文件的路径。
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device",device)
# vgg_model_4096 = models.vgg16(pretrained=False)
# vgg_model_4096.load_state_dict(torch.load("../models/vgg16-397923af.pth", map_location=device))
# print('模型各部分名称', vgg_model_4096._modules.keys())
# # 直接print可以看到层次信息，比较好看 ，但是看不到tensor大小
# print(vgg_model_4096)
# # 其实这个取出分类器本意是为了去掉分类器最后几层，获得4096个特征数，例如，
# # new_classifier = torch.nn.Sequential(*list(vgg_model_4096.children())[-1][:6])
# # 但米良最后还是没有修改分类器
# new_classifier = torch.nn.Sequential(*list(vgg_model_4096.children())[-1])
# vgg_model_4096.classifier = new_classifier
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# # # torchinfo 的summary
# # # NCHW
# # # 返回ModelStatistics实例
# # # ModelStatistics中的summary_list
#
# from torchinfo import summary
# results=summary(vgg_model_4096, (1,3, 720, 1080))
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# VGG                                      [1, 1000]                 --
# ├─Sequential: 1-1                        [1, 512, 22, 33]          --
# │    └─Conv2d: 2-1                       [1, 64, 720, 1080]        1,792
# │    └─ReLU: 2-2                         [1, 64, 720, 1080]        --
# │    └─Conv2d: 2-3                       [1, 64, 720, 1080]        36,928
# │    └─ReLU: 2-4                         [1, 64, 720, 1080]        --
# │    └─MaxPool2d: 2-5                    [1, 64, 360, 540]         --
# │    └─Conv2d: 2-6                       [1, 128, 360, 540]        73,856
# │    └─ReLU: 2-7                         [1, 128, 360, 540]        --
# │    └─Conv2d: 2-8                       [1, 128, 360, 540]        147,584
# │    └─ReLU: 2-9                         [1, 128, 360, 540]        --
# │    └─MaxPool2d: 2-10                   [1, 128, 180, 270]        --
# │    └─Conv2d: 2-11                      [1, 256, 180, 270]        295,168
# │    └─ReLU: 2-12                        [1, 256, 180, 270]        --
# │    └─Conv2d: 2-13                      [1, 256, 180, 270]        590,080
# │    └─ReLU: 2-14                        [1, 256, 180, 270]        --
# │    └─Conv2d: 2-15                      [1, 256, 180, 270]        590,080
# │    └─ReLU: 2-16                        [1, 256, 180, 270]        --
# │    └─MaxPool2d: 2-17                   [1, 256, 90, 135]         --
# │    └─Conv2d: 2-18                      [1, 512, 90, 135]         1,180,160
# │    └─ReLU: 2-19                        [1, 512, 90, 135]         --
# │    └─Conv2d: 2-20                      [1, 512, 90, 135]         2,359,808
# │    └─ReLU: 2-21                        [1, 512, 90, 135]         --
# │    └─Conv2d: 2-22                      [1, 512, 90, 135]         2,359,808
# │    └─ReLU: 2-23                        [1, 512, 90, 135]         --
# │    └─MaxPool2d: 2-24                   [1, 512, 45, 67]          --
# │    └─Conv2d: 2-25                      [1, 512, 45, 67]          2,359,808
# │    └─ReLU: 2-26                        [1, 512, 45, 67]          --
# │    └─Conv2d: 2-27                      [1, 512, 45, 67]          2,359,808
# │    └─ReLU: 2-28                        [1, 512, 45, 67]          --
# │    └─Conv2d: 2-29                      [1, 512, 45, 67]          2,359,808
# │    └─ReLU: 2-30                        [1, 512, 45, 67]          --
# │    └─MaxPool2d: 2-31                   [1, 512, 22, 33]          --
# ├─AdaptiveAvgPool2d: 1-2                 [1, 512, 7, 7]            --
# ├─Sequential: 1-3                        [1, 1000]                 --
# │    └─Linear: 2-32                      [1, 4096]                 102,764,544
# │    └─ReLU: 2-33                        [1, 4096]                 --
# │    └─Dropout: 2-34                     [1, 4096]                 --
# │    └─Linear: 2-35                      [1, 4096]                 16,781,312
# │    └─ReLU: 2-36                        [1, 4096]                 --
# │    └─Dropout: 2-37                     [1, 4096]                 --
# │    └─Linear: 2-38                      [1, 1000]                 4,097,000
# ==========================================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# Total mult-adds (G): 238.01
# ==========================================================================================
# Input size (MB): 9.33
# Forward/backward pass size (MB): 1679.41
# Params size (MB): 553.43
# Estimated Total Size (MB): 2242.17
# ==========================================================================================
#


# from torchsummary import summary
# summary(vgg_model_4096, (3,720, 1080))

# device cpu
# 模型各部分名称 odict_keys(['features', 'avgpool', 'classifier'])
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1        [-1, 64, 720, 1080]           1,792
#               ReLU-2        [-1, 64, 720, 1080]               0
#             Conv2d-3        [-1, 64, 720, 1080]          36,928
#               ReLU-4        [-1, 64, 720, 1080]               0
#          MaxPool2d-5         [-1, 64, 360, 540]               0
#             Conv2d-6        [-1, 128, 360, 540]          73,856
#               ReLU-7        [-1, 128, 360, 540]               0
#             Conv2d-8        [-1, 128, 360, 540]         147,584
#               ReLU-9        [-1, 128, 360, 540]               0
#         MaxPool2d-10        [-1, 128, 180, 270]               0
#            Conv2d-11        [-1, 256, 180, 270]         295,168
#              ReLU-12        [-1, 256, 180, 270]               0
#            Conv2d-13        [-1, 256, 180, 270]         590,080
#              ReLU-14        [-1, 256, 180, 270]               0
#            Conv2d-15        [-1, 256, 180, 270]         590,080
#              ReLU-16        [-1, 256, 180, 270]               0
#         MaxPool2d-17         [-1, 256, 90, 135]               0
#            Conv2d-18         [-1, 512, 90, 135]       1,180,160
#              ReLU-19         [-1, 512, 90, 135]               0
#            Conv2d-20         [-1, 512, 90, 135]       2,359,808
#              ReLU-21         [-1, 512, 90, 135]               0
#            Conv2d-22         [-1, 512, 90, 135]       2,359,808
#              ReLU-23         [-1, 512, 90, 135]               0
#         MaxPool2d-24          [-1, 512, 45, 67]               0
#            Conv2d-25          [-1, 512, 45, 67]       2,359,808
#              ReLU-26          [-1, 512, 45, 67]               0
#            Conv2d-27          [-1, 512, 45, 67]       2,359,808
#              ReLU-28          [-1, 512, 45, 67]               0
#            Conv2d-29          [-1, 512, 45, 67]       2,359,808
#              ReLU-30          [-1, 512, 45, 67]               0
#         MaxPool2d-31          [-1, 512, 22, 33]               0
# AdaptiveAvgPool2d-32            [-1, 512, 7, 7]               0
#            Linear-33                 [-1, 4096]     102,764,544
#              ReLU-34                 [-1, 4096]               0
#           Dropout-35                 [-1, 4096]               0
#            Linear-36                 [-1, 4096]      16,781,312
#              ReLU-37                 [-1, 4096]               0
#           Dropout-38                 [-1, 4096]               0
#            Linear-39                 [-1, 1000]       4,097,000
# ================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.90
# Forward/backward pass size (MB): 3384.20
# Params size (MB): 527.79
# Estimated Total Size (MB): 3920.89
# ----------------------------------------------------------------



# print(results)

# vgg_model_4096._get_distribution_strategy = lambda: None
# writer = SummaryWriter(log_dir="./exp1")
# lr=torch.randn((1,3,270,480))
# writer.add_graph(vgg_model_4096, lr)
# writer.close()

#
# ###############################下面是fasterd rcnn模型的profile
import time

import torch
import torchvision.models as models
# from torch.utils.tensorboard import SummaryWriter
from gpu_mem_track import MemTracker
from torchinfo import summary
# torchinfo 答应模型结构 输出shape 参数量

#
# # 如果将pretrained设置为True, 意味着直接得到一个加载了预训练参数的vgg19模型。
# # 就会自动下载vgg19的参数文件并放在本地缓存中。所以不用提供本地参数文件的路径。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_detect = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
print("is model on gpu",next(model_detect.parameters()).is_cuda)
print(model_detect)
t0=time.time()
model_detect.load_state_dict(torch.load("../DRL_continuous/fasterrcnn_resnet50_fpn_coco.pth", map_location="cpu"))
print("is model on gpu",next(model_detect.parameters()).is_cuda,"load 参数 to device ",time.time()-t0)

# 不管参数load到哪里，都还不是真正的cuda  tensor  只有to("cuda")之后才是cuda张量  但是如果前面参数是load到CPU上的，模型to device时间很长
t1=time.time()
# gpu_tracker = MemTracker()
# gpu_tracker.track()
model_detect.to(device)
# gpu_tracker.track()
# 报错！！！！！！！！！
# 服务器上生成txt  发gpu_tracker估计的memory小了很多
# torchinfo 和torchsummary基本是准确的

print("is model on gpu",next(model_detect.parameters()).is_cuda,"model to device",time.time()-t1,"total time ",time.time()-t0)

print('模型各部分名称', model_detect._modules.keys())

results=summary(model_detect, (1,3, 270, 480))

# 生成文件，为了后面可以命令行进行可视化,出现以下错误未解决
# RuntimeError: Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions
# Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions
# Error occurs, No graph saved
# writer = SummaryWriter(log_dir="./exp1")
# lr=torch.randn((1,3,270,480))
# writer.add_graph(model_detect, lr)
# writer.close()


# input=torch.rand(1, 3, 224, 224).to(device)
#
# gpu_tracker = MemTracker()
# # output=model_detect.forward(input)
# gpu_tracker = MemTracker()

