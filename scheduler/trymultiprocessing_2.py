import os

import torch
import torchvision.models as models
import time

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
print("进程号：",os.getpid())

def forward(input,net):
    output=net.forward(input)
    output = net.forward(input)

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    start=time.time()
    net1 = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).to(device)
    net2 = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).to(device)
    net3 = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).to(device)
    net4 = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False).to(device)
    end=time.time()
    print("prepare model:",end-start)

    inputs = torch.rand(1, 3, 224, 224).to(device)
    # 这个先执行一次是因为cuda上第一次初始化cuda上下文会很耗时间，测出来的 时间不是真正执行 的时间
    outputs = net1.forward(inputs)
    start = time.time()
    outputs = net1.forward(inputs)
    end = time.time()
    print('Time taken for forward prop on 1 stream: (sequentially)', end - start)

    t=Timer()
    start = time.time()
    t.tic()
    outputs = net1.forward(inputs)
    t.toc()
    t.tic()
    outputs = net2.forward(inputs)
    t.toc()
    t.tic()
    outputs = net3.forward(inputs)
    t.toc()
    t.tic()
    outputs = net4.forward(inputs)
    t.toc()
    end = time.time()
    print('Time taken for forward prop on all stream: (sequentially)', end - start)



    print("=======================================================")
    torch.multiprocessing.set_sharing_strategy('file_system')
    mp = torch.multiprocessing.get_context('spawn')


    p1 = mp.Process(target=net1.forward, args=(inputs,))
    p2 = mp.Process(target=net2.forward, args=(inputs,))
    p3 = mp.Process(target=net3.forward, args=(inputs,))
    p4 = mp.Process(target=net4.forward, args=(inputs,))
    p5 = mp.Process(target=net1.forward, args=(inputs,))
    p6 = mp.Process(target=net2.forward, args=(inputs,))
    p7 = mp.Process(target=net3.forward, args=(inputs,))
    p8 = mp.Process(target=net4.forward, args=(inputs,))


    # p1 = mp.Process(target=forward, args=(inputs,net1))
    # p2 = mp.Process(target=forward, args=(inputs,net2))
    # p3 = mp.Process(target=forward, args=(inputs,net3))
    # p4 = mp.Process(target=forward, args=(inputs,net4))
    # p1 = mp.Process(target=forward, args=(inputs,net1))
    # p2 = mp.Process(target=forward, args=(inputs,net2))
    # p3 = mp.Process(target=forward, args=(inputs,net3))
    # p4 = mp.Process(target=forward, args=(inputs,net4))



    start = time.time()

    t1=time.time()
    print(t1,"t1")
    p1.start()
    t2 = time.time()
    print(t2,"t2")
    p2.start()
    t3 = time.time()
    print(t3,"t3")
    p3.start()

    t4 = time.time()
    print(t4,"t4")
    p4.start()
    t5 = time.time()
    print(t5,"t5")

    p5.start()
    p6.start()
    p7.start()
    p8.start()



    p1.join()

    p2.join()

    p3.join()

    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    end = time.time()
    print('Time taken for forward prop on all stream: (parallel)', end - start)










