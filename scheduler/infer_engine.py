"""
整体代码逻辑：
1.先将几个模型加载到CPU，并放入队列中
2.scheduler会安排GPU index，
3.然后多进程传入参数，参数中包含模型 输入 GPU_id参数，target函数是推理函数，可以是不同模型不一样的



# 可能的不合理之处：每个模型可能到来的时间不一样，真实的模拟应该一个进程或线程安排模型入队，并同时开启多进程对每个模型输入数据做一些数据初始化，
另一个单独的进程或者线程在判断是否有GPU空闲，空闲才安排模型执行，但是对于实现难度上来说，将会面临复杂的进程逻辑（锁）。且很像贪心或启发式算法

因此，我现在是假设一段时间内同时安排多个模型，暂时不考虑输入在CPU上的多进程预处理
不是针对每一个到来的模型进行安排，而是默认这三个模型一起到来，这样安排和预测也比较有意思

# TODO：
1.多线程读取数据（可能没有必要，如果数据量比较少的话，对于实时视频分析来说，可以设置interval，一段时间的数据集就不是很大了）、处理数据，处理不同模型输入数据
2.加入结果的返回，考虑到多进程或 线程消息返回或同步可能带来额外的开销，因此我不太敢一开始就进行结果返回，如果不是best practice后面将很麻烦
"""

from query import QueryContext

import torch
import torchvision.models as models
from collections import deque
import os

def scheduler(queries):
    gpu_ids=deque([0,1,0])
    return gpu_ids
def prediction(query,id):

    device=f"cuda:{id}"
    print(f"进程号{os.getpid()}   设备号:{device}")
    model=query.model.to(device)
    model.eval()
    #将每一个input进行推理
    for input in query.input_list:
        input=input.to(device)
        output=model.forward(input)
        # print(output.shape)

if __name__=="__main__":
    n_models=3
    host_device="cpu"
    net1 = models.vgg16(pretrained=False)
    net1.load_state_dict(torch.load("../models/vgg16-397923af.pth", map_location=host_device))

    new_classifier = torch.nn.Sequential(*list(net1.children())[-1])
    net1.classifier = new_classifier

    net2 = models.quantization.mobilenet_v3_large(pretrained=False, quantize=False)
    net2.load_state_dict(
        torch.load("../models/mobilenet_v3_large-8738ca79.pth", map_location=host_device))


    net3 = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    net3.load_state_dict(torch.load("../models/fasterrcnn_resnet50_fpn_coco.pth", map_location=host_device))

    # 输入应该是data list
    input_list = [torch.rand(1, 3, 224, 224)]
    queries = deque()
    new_query=QueryContext(0,net1,input_list)
    queries.append(new_query)
    new_query = QueryContext(0, net2, input_list)
    queries.append(new_query)
    new_query = QueryContext(0, net3, input_list)
    queries.append(new_query)

    gpu_ids=scheduler(queries)
    print("schedule device numbers:",gpu_ids)

    torch.multiprocessing.set_sharing_strategy('file_system')
    mp = torch.multiprocessing.get_context('spawn')
    pool=[]
    for i in range(len(queries)):
        query=queries.popleft()
        gpu_id=gpu_ids.popleft()
        p=mp.Process(target=prediction,args=(query,gpu_id))
        pool.append(p)
        p.start()


    for p in pool:
        p.join()