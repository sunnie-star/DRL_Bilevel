"""
效果不好，学得action一个值
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import pickle
import torchvision.models as models
from torch.autograd import Variable
import diff_processor
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record,set_init_LSTM
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
# from envs import Envs,Envs1,Envs2,Envs2mv,Envs3
from envs import Envs3,Envs4
import os
# from dds_utils import merge_boxes_in_results,Region,Results,read_results_dict
# from ..backend.object_detector import Detector
import cv2 as cv
import time as T
from multicate import multi_categorical_maker
import numpy as np
import matplotlib.pyplot as plt
import time
os.environ["OMP_NUM_THREADS"] = "20"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

# env = gym.make('CartPole-v0')
# N_S = env.observation_space.shape[0]
# N_A = env.action_space.n
# print(N_S,N_A)
#raw_images_path='D:\\shiyan\\server\\server\\my_dds_sr_619\\dataset\\trafficcam_1\\src\\'

N_S=30+128+30+8+8+1+1

N_A=75*10


print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        #self.pi0=nn.LSTM(s_dim, 128, batch_first = False)
        self.pi1 = nn.Linear(self.s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        #self.pi3 = nn.Linear(128, a_dim)
        #self.v0=nn.LSTM(s_dim, 128, batch_first = False)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
#        set_init_LSTM([self.pi0,self.v0])
        #self.vgg16 = models.vgg16(pretrained=True)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical
        #self.distribution=multi_categorical_maker([15, 15])
        # self.a_hx = None
        # self.a_cx = None
        # self.c_hx = None
        # self.c_cx = None
        self.a_hidden=None
        self.c_hidden = None

        # self.a_hidden = self.init_hidden()
        # self.c_hidden = self.init_hidden()

    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return (torch.zeros(128).unsqueeze(0).unsqueeze(0),
                torch.zeros(128).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # print(x.size())
        #print("x",x)
        #x = x.view(len(x), 1, -1)
        #print(x)
        #print('a',(x))

        ###
        #pi0,self.a_hidden=self.pi0(x,self.a_hidden)
        #print(self.a_hidden)
        #self.a_hidden=(self.a_hidden[0].data,self.a_hidden[1].data)
        #pi0,_=self.pi0(x)
        #print("tuple",pi0)
        #pi0,_=self.pi0(x)
        #print("notuple",pi0)
        #pi0=torch.tanh(pi0)
        pi1 = torch.tanh(self.pi1(x))
        # torch.tanh 激活函数
        #print("pi1",pi1)
        logits = self.pi2(pi1)
        # logits1 = self.pi3(pi1)
        # logits=torch.cat([logits0, logits1], dim=1)
        #print('log',logits,logits.shape)
        #logits1 = self.pi3(pi1)
        #logits=
        #v0,self.c_hidden=self.v0(x,self.c_hidden)
        #self.c_hidden=(self.c_hidden[0].data,self.c_hidden[1].data)
        #self.c_hidden = self.c_hidden.data
        #v0,_=self.v0(x)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        #logits0,logits1=logits.split([50, 25], dim=1)
        #print("logits,]",logits)
        #print(logits)
        #prob = F.softmax(logits[0], dim=1).data
        # 首先，我们知道概率有两个性质：1）预测的概率为非负数；2）各种预测结果概率之和等于1。
        # softmax就是将在负无穷到正无穷上的预测结果按照这两步转换为概率的。
        prob = F.softmax(logits, dim=1).data
        # print('prob',prob)
        # print(torch.rand(2, 75).shape)
        # print(prob.shape)
        # torch.distributions.Categorical其作用是创建以参数probs为标准的类别分布，样本是来自“0，...，K-1”的整数，
        # 也就是按照probs的概率，在相应的位置进行采样，采样返回的是该位置的整数索引。
        m = self.distribution(prob)
        # print((m.sample().numpy()[0],m.sample().numpy()),"sample  m")
        # (20, array([3]))
        #这个输出的采样竟然不是概率最大的对应的索引额？？？？？？？？？？？？？？？
        # print(np.argmax(prob.numpy()[0]))

        #print(m.sample().numpy()[0])
        # prob1 = F.softmax(logits1, dim=1).data
        # m1 = self.distribution(prob1)
        #print("prob",prob)
        #print((m0.sample().numpy()[0],m1.sample().numpy()[0]))
        #return int(torch.argmax(prob))
        # m.sample()从序列a中随机抽取n个元素,并将n个元素生以list形式返回。
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    # 继承mp.Process这个基类
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name,env):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        #self.env = gym.make('CartPole-v0').unwrapped
        self.env=env
    def run(self):
        total_step = 1

        print(self.g_ep.value, ' start')
        while self.g_ep.value < MAX_EP:
            # print(self.name, ' into1')
            # s是第一个gop的状态，  30（edge difference)+128(vgg feature)+30(res)
            s = self.env.reset()
            # self.lnet.a_hx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.a_cx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.c_hx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            # self.lnet.c_cx = torch.zeros(128).unsqueeze(0).unsqueeze(0)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                #print(self.name, ' into2')
                # if self.name == 'w00':
                #     self.env.render()
                #print(s,"v_wrap(s)",v_wrap(s))
                # v_wrap将numpy 变为torch tensor, s[None,:]关键字None则表示维度扩充
                a = self.lnet.choose_action(v_wrap(s[None,:]))
                # a是[0，75）之间的数字
                s_, r, done, _ = self.env.step(a)
                print(self.g_ep.value, ' epoch--------------------------------')
                print("reward done action",r,done,a,'\n')

                #if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync

                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        print(self.name, ' over')
        self.res_queue.put(None)


# def get_chunk_sizes():
#     dict={
#         10:10919 / 1024,
#         15:6051/1024,
#         20:3192/1024,
#         25:1621/1024,
#         30:817/1024,
#         35:442/1024,
#         40:264/1024,
#         45:172/1024,
#         50: 118 / 1024
#
#
#     }
#
#     return dict

# bitrate_list=[200,300,450,750,1200,1850,2850,4300,6000,8000]
def get_chunk_sizes():
    dict={
        200:200 / 1024,
        300:300/1024,
        450:450/1024,
        750:750/1024,
        1200:1200/1024,
        1850:1850/1024,
        2850:2850/1024,
        4300:4300/1024,
        6000:6000 / 1024,
        8000:8000/1024


    }

    return dict
if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    # mp.Value 进程之间交流数据。
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()


    times = []
    result = []
    # times是infer的时间，我根据后面猜测的
    # with open("times.txt", "rb") as get_myprofile:
    #     times = pickle.load(get_myprofile)

    # with open("features.txt", "rb") as get_myprofile:
    #     features = pickle.load(get_myprofile)
    with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/constbit_features_dict_long.txt", "rb") as get_myprofile:
        features_dict = pickle.load(get_myprofile)

    # DiffProcessor.str2class('edge')返回了一个类，（0）实例化该类
    d_pro = diff_processor.DiffProcessor.str2class('edge')(0)


    # 其实有bug
    with open("res.txt", "rb") as get_myprofile:
        res = pickle.load(get_myprofile)
    # 所有帧的edge差值(不包括I帧），以一个gop一组，每组29个；每个gop第一帧的diff_value；每个gop第一帧的原图
    bitrate_list=[200,300,450,750,1200,1850,2850,4300,6000,8000]
    states_dict={}
    diff_gop_dict={}
    for bitrate in bitrate_list:
        video_path = f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_bit/video_test540_{bitrate}/video_test540_{bitrate}.mp4"
        # 所有帧的edge差值(不包括I帧），以一个gop一组，每组29个；每个gop第一帧的diff_value；每个gop第一帧的原图
        states,diff_gop,_= d_pro.get_all_diff_vector(video_path, 30)
        states_dict[bitrate]=states
        diff_gop_dict[bitrate]=diff_gop
    # with open("dds_results540.txt", "rb") as get_myprofile:
    #     result = pickle.load(get_myprofile)
    #     for id in result.regions_dict:
    #         for r in result.regions_dict[id]:
    #             r.y = (r.y - 0.077) / 0.84583333
    # #sun 疑问：r.y = (r.y - 0.077) / 0.84583333
    # with open("dds_results.txt", "rb") as get_myprofile:
    #     h_result = pickle.load(get_myprofile)

    result_dict={}
    h_result_dict = {}

    # dds_results540_50.txt
    # dds_results540_50_plusSR.txt
    for bitrate in bitrate_list:
        with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/constbit_dds_results540_{bitrate}.txt","rb") as myfile:
            result=pickle.load(myfile)
        result_dict[bitrate]=result.regions_dict
        with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/constbit_dds_results540_{bitrate}_plusSR.txt", "rb") as myfile:
            h_result = pickle.load(myfile)
        h_result_dict[bitrate] = h_result.regions_dict
    # 16个线程worker
    chunksizes=get_chunk_sizes()
    print(chunksizes)
    f=open(f"../trace2.txt","r")
    # /home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/trace2.txt
    band=[]
    for line in f:
        band.append(float(line.split(' ')[1]))
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, Envs4(720, 1280, 2000,
                                                                             states_dict,diff_gop_dict,
                                                                             times, result_dict,
                                                                            h_result_dict,
                                                                             res,features_dict,chunksizes,band)) for i in range(16)]

    start = T.time()
    [w.start() for w in workers]

    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    end = T.time()
    print('t',(end-start))
    t=int(T.time())
    torch.save(gnet, f"./models/constbit_resnew_{t}.pth")
    import matplotlib
    # matplotlib.use('TkAgg')

    with open(f"constbit_res_record_{t}.txt","wb") as myfile:
        pickle.dump(res,myfile)

    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig(f"constbit_training_res_{t}.png")
    plt.show()

