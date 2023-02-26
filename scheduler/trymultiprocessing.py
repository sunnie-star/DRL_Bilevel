import os

import torch
import torchvision.models as models
import time
import psutil
import argparse



class Timer:
    def __init__(self):
        self.begin=0
        self.end=0
        self.diff=0
    def tic(self):
        self.begin=time.time()
    def toc(self,tip=None):
        self.end=time.time()
        self.diff=self.end-self.begin
        if self.diff>0.001:
            print(self.diff)

        elif self.diff<0.001 and self.diff>0:
            # 忽略不计时
            print("ignore")
        else:
            print("Error!!!!!!")

def forward_(input,net):
    t1=time.time()
    print(os.getpid(),"begin subprocess function:",time.time(),"--------------------")
    for i in range(100):
        output=net.forward(input)
    print(os.getpid(),"end subprocess function:",time.time(),time.time()-t1,"---------------------")

def forward(input,net):
    t1 = time.time()
    print("begin subprocess function", os.getpid(), time.time(), "--------------------")

    output = net.forward(input)
    t2=time.time()
    print("-----该进程第一次执行模型", t2 - t1)
    output = net.forward(input)
    print("-----该进程第二次执行模型", time.time() - t2)
    t3=time.time()
    count=400
    # for i in range(count):
    #     output=net.forward(input)
    # print(f"-----该进程执行了{count}次该模型", time.time() - t3)
    print("end subprocess function:", os.getpid(), time.time(), time.time() - t1, "---------------------")

mem = psutil.virtual_memory()
print(os.getpid(), "进程start", f"cpu利用率{psutil.cpu_percent()}、总内存{mem.total}、已使用{mem.used}、未使用{mem.free} ",time.time())
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='params')

    parser.add_argument('--n_process', type=int, default=1,
                        help='number of processes for model execution')
    args = parser.parse_args()



    n_process =4
    print("进程数量:",n_process,"++++++++++++++++++++++++++++++++++++++++++++++")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device="cuda:0"
    print(device)
    start=time.time()
    net1 = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).to(device)
    # net1 = models.vgg16(pretrained=False)
    # net1.load_state_dict(torch.load("../models/vgg16-397923af.pth", map_location=device))
    # print('模型各部分名称', net1._modules.keys())
    # net1.to(device)
    #
    # net3 = models.vgg16(pretrained=False)
    # net3.load_state_dict(torch.load("../models/vgg16-397923af.pth", map_location=device))
    # print('模型各部分名称', net1._modules.keys())
    # net3.to(device)
    #
    # net2 = models.vgg16(pretrained=False)
    # net2.load_state_dict(torch.load("../models/vgg16-397923af.pth", map_location=device))
    # print('模型各部分名称', net1._modules.keys())
    # net2.to(device)
    #
    # net4 = models.vgg16(pretrained=False)
    # net4.load_state_dict(torch.load("../models/vgg16-397923af.pth", map_location=device))
    # print('模型各部分名称', net1._modules.keys())
    # net4.to(device)

    net2 = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).to(device)

    net3 = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).to(device)
    net4 = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).to(device)

    end=time.time()
    print("prepare model:",end-start)

    inputs = torch.rand(1, 3, 224, 224).to(device)
    input1 = torch.rand(1, 3, 224, 224).to(device)
    input2 = torch.rand(1, 3, 224, 224).to(device)
    input3 = torch.rand(1, 3, 224, 224).to(device)
    input4 = torch.rand(1, 3, 224, 224).to(device)
    # # 这个先执行一次是因为cuda上第一次初始化cuda上下文会很耗时间，测出来的 时间不是真正执行 的时间
    # start = time.time()
    # outputs = forward(inputs,net1)
    # end = time.time()
    # print('Time taken for （第一次执行）forward prop on 1 stream: (sequentially)', end - start)

    start = time.time()
    # 以下这也是为了看计算繁重的时候对比
    outputs = forward(inputs,net1)
    end = time.time()
    print('Time taken for forward prop on 1 stream: (sequentially)', end - start)
    print("=======================================================")

    # 下面是用自己写得forward函数（计算比较繁重）来对比
    t=Timer()
    start = time.time()

    t.tic()
    outputs = forward(input1, net1)
    t.toc()
    t.tic()
    outputs = forward(input2, net2)
    t.toc()
    t.tic()
    outputs = forward(input3, net3)
    t.toc()
    t.tic()
    outputs = forward(input4, net4)
    t.toc()


    end = time.time()
    print('Time taken for forward prop on all stream: (sequentially)', end - start)



    print("=======================================================")
    torch.multiprocessing.set_sharing_strategy('file_system')
    mp = torch.multiprocessing.get_context('spawn')

    start = time.time()

    p1 = mp.Process(target=forward, args=(input1, net1))
    p2 = mp.Process(target=forward, args=(input2, net2))
    p3 = mp.Process(target=forward, args=(input3, net3))
    p4 = mp.Process(target=forward, args=(input4, net4))

    start = time.time()
    p1.start()
    p2.start()
    p3.start()
    p4.start()


    p1.join()
    p2.join()
    p3.join()
    p4.join()

    end = time.time()

    print('Time taken for forward prop on all stream: (parallel)', end - start)