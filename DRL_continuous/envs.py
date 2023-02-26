# 改变了reward中的time惩罚项  resnew_1673276406.pth  表现很差  又改回来了
"""

Envs4是   discrete_A3C_训练_恒定比特率.py对应的envs,用的是10个bitrate的动作空间，即bitrate_thres
Envs5 加上transmit time 并且吧qp单位变为mbps
Envs6是为连续动作空间
Envs7 在Envs5基础上减少动作空间，并且将action/。。。统一化  然后直接可以用Envs7代替Envs5
我看一下视频上每一段的acc和内容的关系，即全部设置qp=30
Envs9  进行公平的带宽分配
关于归一化，之前state都没有做归一化啊。。。但是考虑到qp放进去很突兀
"""
from pro_mv import find_refer,find_re_refer,move_bbox,move_re_bbox
from sklearn.decomposition import PCA
from PIL import Image
import diff_processor
#from ..backend.object_detector import Detector
import pickle
import torchvision.transforms as transforms
import numpy as np
from gym import spaces
from gym.utils import seeding
from dds_utils import (Results, read_results_dict, filter_bbox_group,evaluate, cleanup, Region,
                       compute_regions_size, merge_boxes_in_results, extract_images_from_video,calc_iou)
import time as T
import time
import math
import random
import matplotlib.pyplot as plt
from numpy import *
# mean函数头文件

N_thres=15
N_rthres=5
N_qp=6
ground_path='./video_test_gt'
#ground_path='video_test_gt'
raw_images_path='./video_test/src'
thres=range(50,800,int(750/N_thres))
thres=[i/1000 for i in thres]
# print(len(thres))
#r_thres=range(0,2200,75)
r_thres=range(0,3000,int(3000/N_rthres))
r_thres=[i/1000 for i in r_thres]
N_CHUNK=8

qp_thres=[20,25,30,35,40,45]
# qp_thres=[20,25,30,35,40]
# qp_thres=[30,35,40]
bitrate_thres=[200,300,450,750,1200,1850,2850,4300,6000,8000]

# qp_thres=[10,20,30,40,50]   间隔太大了
def Normalize(array):
    '''
    Normalize the array
    '''
    mx = np.nanmax(array)
    mn = np.nanmin(array)
    t = (array-mn)/(mx-mn)
    return t

class Envs4:

    def __init__(self, height, width, length, states_dict, diff_gop_dict, times, result,h_result,res, features_dict,chunksizes,bandwith):
        # self.args=create_args()
        self.last_bitrate=1200
        with open('./I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
        #print(self.I_frame)
        self.environment_title='video_V0'
        #self.action_space=spaces.Discrete(75)
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )
        high=np.zeros(128+60,dtype=np.float32)
        high=np.array([np.finfo(np.float32).max for _ in high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states_dict=states_dict
        self.states = states_dict[self.last_bitrate]
        self.diff_gop_dict = diff_gop_dict
        self.diff_gop = diff_gop_dict[self.last_bitrate]

        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = None
        self._max_episode_steps=100
        self.t_reward=[]
        #self.model = Detector()
        # with open("dds_results.txt", "rb") as get_myprofile:
        #     h_result = pickle.load(get_myprofile)
        # self.ground_truth_dict = read_results_dict(ground_path)
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     1000, self.ground_truth_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.4, 0.4)
        # print(f1)
        #self.ground_truth_dict =h_result.regions_dict
        self.times = times
        self.result_dict = result
        self.show_results = Results()
        self.h_result_dict=h_result
        # gt是质量最好的540p  每帧SR之后 infer的结果
        self.ground_truth_dict=h_result[8000]

        self.res=res
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features_dict = features_dict
        self.features=features_dict[self.last_bitrate]


        with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/models/pca.pkl","rb") as get_myprofile:
            self.pca=pickle.load(get_myprofile)
        # pca = PCA(n_components=128)  # 实例化
        # self.pca = pca.fit(np.array(self.features))  # 拟合模型
        self.srl=[]
        self.dtl=[]
        self.s_a=0
        self.d_a=0
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)
        self.bitrate_bitrate = chunksizes
        self.chunk_times=np.zeros((N_CHUNK))
        self.chunk_sizes=np.zeros((N_CHUNK))

        self.bandwith=bandwith

        self.bitrate_trace=[]
        self.last_compute_time=0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.idx = 30
        self.queue = 0
        # self.states[0] 取一个gop的state  29张（不算第一帧）
        state_ = copy.deepcopy(self.states[0])
        res_ = copy.deepcopy(self.res[1:30])
        self.last_sr_frame=0
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        res_.insert(0,0)
        #print(len(res_),self.idx)
        #
        self.state = np.array(state_)
        # pca输入是两维
        random_integer = random.randint(0, 10)
        self.last_bitrate=bitrate_thres[random_integer % len(bitrate_thres)]
        print("reset random bitrate", self.last_bitrate)
        self.features=self.features_dict[self.last_bitrate]
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        state_+=res_
        # # 这个后面需要改时间和size，现在只是简单模拟！！！！！！！！！！！！！！！！！！！！！！！！！！！
        for i in range(N_CHUNK):
            self.chunk_sizes[i]=self.bandwith[N_CHUNK-1-i]


        self.chunk_times = [self.bitrate_bitrate[self.last_bitrate] / i for i in self.chunk_sizes]

        self.bitrate_trace.append(self.last_bitrate)
        state_.extend(self.chunk_sizes)
        state_.extend(self.chunk_times)
        state_.append(self.last_bitrate)
        state_.append(self.last_compute_time)
        # state_ list:205
        # print(state_)
        return np.array(state_)
    # def delete(self):
    #     for r in self.l_result:
    #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist
    def move(self,frame_idx):
        if frame_idx + 1 in self.I_frame:
            for r in self.l_result:
                label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                #final_results.append(r)
        else:
            _d_result = Results()
            refer = find_refer(frame_idx + 1) - 1
            if refer + 1 == frame_idx:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    _x, _y = _x / 4, _y / 4
                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                    if _x != 0 and _y != 0:
                        _x = _x - __x
                        _y = _y - __y
                    _x, _y = _x / 4, _y / 4

                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            else:

                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                    _x, _y = _x / 4, _y / 4

                    # _x, _y =0,0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            self.l_result = _d_result.regions_dict[frame_idx]

    def resize(self,r):
        if (r.x+r.w)>1:
            r.w=1-r.x
        if (r.y + r.h) > 1:
            r.w = 1 - r.y
        r.x=max(0,r.x)
        r.y=max(0,r.y)
        r.h=max(0,r.h)
        r.w=max(0,r.w)
        r.x = min(1, r.x)
        r.y = min(1, r.y)
        return r

    def isexist(self,region,result):
        for r in result:
            if calc_iou(r,region)>0.1:
                return True
        return False
    def isdelete(self,r):

        if r.w==0 or r.h==0:
            return False
        return True

    def clear_sr(self,frame_idx):
        res=[]
        for region in self.l_result:
            flag=False
            for h_region in self.result[frame_idx]:
                if filter_bbox_group(region, h_region, 0.1) or h_region.w*h_region.h>0.005:
                    flag=True
                    break
            if not flag:
                res.append(region)
        self.l_result=res+self.result[frame_idx]
    def step(self, action):
        # 先通过a映射出edge difference阈值选出select_frames，再通过映射出的res阈值再selected_frames中选出SR list
        print("action",action,'\n','idx',self.idx,'\n','goplen',self.goplen)
        print("s sr bitrate", int(action/50),   int(action%50/10),action%10,self.last_bitrate,bitrate_thres[action%10])
        # self.idx  30 60 90...(下一个gop的第一帧帧号，用来索引本gop的末尾帧号,self.goplen 恒定30
        s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(int(action/50))], self.state, (self.idx - self.goplen),
                                                    self.goplen)
        #self.select.append(s_frames)r_thres[int(action/15)]
        SR_list=self.get_SRlist(s_frames,r_thres[int(action%50/10)])
        self.result=self.result_dict[self.last_bitrate]
        # 对应的bitrate  SR之后infer的结果
        self.h_result=self.h_result_dict[self.last_bitrate]
        self.srl+=SR_list
        self.dtl+=s_frames
        # s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres[self.d_a]), self.state, (self.idx - self.goplen),                                             self.goplen)
        # SR_list = self.get_SRlist(s_frames, r_thres[self.s_a])
        # s_frames = [0, 16, 29, 39, 50, 58, 67, 78, 87, 101, 115, 116, 135, 145, 162, 174, 203, 232, 261, 290, 294, 297, 302, 307,
        #            310, 318, 319, 330, 345, 348, 351, 354, 356, 365, 377, 382, 386, 394, 406, 412, 428, 435, 441, 446, 450, 454,
        #            457, 462, 464, 480, 493, 497, 500, 505, 511, 518, 522, 527, 532, 540, 549, 551, 567, 580, 588, 600, 604, 608,
        #            609, 629, 638, 661, 667, 671, 674, 679, 684, 689, 694, 696, 713, 725, 746, 754, 766, 780, 783, 796, 809, 812,
        #            821, 831, 841, 854, 869, 870, 882, 895, 899, 901, 903, 905, 907, 909, 912, 915, 918, 921, 925, 928, 932, 936,
        #            941, 948, 956, 957, 977, 986, 1003, 1015, 1044, 1052, 1058, 1065, 1072, 1073, 1101, 1102, 1111, 1122, 1131,
        #            1147, 1160, 1177, 1189, 1214, 1218, 1246, 1247, 1276, 1304, 1305, 1333, 1334, 1352, 1363, 1385, 1392, 1410,
        #            1421, 1440, 1450, 1468, 1479, 1496, 1508, 1526, 1537, 1557, 1566, 1587, 1595, 1614, 1624, 1644, 1653, 1674,
        #            1682, 1697, 1711, 1720, 1731, 1740, 1751, 1762, 1769, 1787, 1798, 1815, 1827, 1843, 1856, 1871, 1885, 1900,
        #            1914, 1929, 1943, 1958, 1972, 1988, 2001, 2017, 2030, 2046, 2059, 2075, 2088, 2103, 2117, 2118, 2120, 2122,
        #            2123, 2124, 2125, 2126, 2127, 2129, 2130, 2132, 2134, 2136, 2138, 2140, 2142, 2144, 2146, 2164, 2175, 2193,
        #            2204, 2214, 2225, 2233, 2252, 2262, 2268, 2274, 2280, 2284, 2288, 2291, 2306, 2320, 2333, 2347, 2349, 2361,
        #            2376, 2378, 2390, 2403, 2407, 2422, 2436, 2451, 2465, 2475, 2490, 2494, 2511, 2523, 2538, 2552, 2568, 2581,
        #            2597, 2610, 2627, 2639, 2655, 2668, 2681, 2696, 2697, 2708, 2726, 2745, 2755, 2770, 2784, 2803, 2813, 2830,
        # #            2842, 2845, 2848, 2850, 2854, 2859, 2862, 2867, 2871, 2893, 2900, 2922, 2929, 2946]
        # s_frames=[0, 29, 41, 55, 58, 63, 68, 73, 77, 82, 87, 96, 100, 105, 113, 116, 145, 174, 203, 232, 261, 290, 319, 348, 358, 376, 377, 406, 435, 444, 450, 458, 464, 474, 483, 493, 505, 517, 522, 531, 543, 551, 560, 570, 580, 590, 600, 609, 619, 629, 638, 648, 658, 667, 678, 689, 695, 696, 701, 710, 718, 725, 732, 741, 749, 750, 754, 762, 769, 776, 783, 807, 812, 823, 836, 841, 852, 866, 870, 877, 887, 896, 899, 902, 911, 919, 927, 928, 937, 947, 957, 968, 979, 986, 995, 1005, 1015, 1029, 1041, 1044, 1055, 1063, 1070, 1073, 1093, 1102, 1119, 1131, 1147, 1160, 1183, 1189, 1210, 1218, 1239, 1247, 1270, 1276, 1298, 1305, 1327, 1334, 1353, 1363, 1392, 1416, 1421, 1449, 1450, 1479, 1505, 1508, 1537, 1566, 1595, 1624, 1653, 1680, 1682, 1705, 1711, 1740, 1765, 1769, 1782, 1794, 1798, 1816, 1827, 1851, 1856, 1881, 1885, 1906, 1914, 1927, 1940, 1943, 1963, 1972, 1990, 2001, 2022, 2030, 2051, 2059, 2069, 2077, 2085, 2088, 2111, 2117, 2123, 2129, 2135, 2141, 2146, 2158, 2167, 2175, 2187, 2197, 2204, 2211, 2218, 2225, 2232, 2233, 2248, 2262, 2272, 2281, 2291, 2298, 2305, 2312, 2319, 2320, 2329, 2338, 2348, 2349, 2357, 2366, 2374, 2378, 2388, 2396, 2407, 2419, 2430, 2436, 2447, 2459, 2465, 2471, 2476, 2481, 2486, 2490, 2494, 2510, 2521, 2523, 2533, 2548, 2552, 2564, 2577, 2581, 2587, 2597, 2606, 2610, 2620, 2627, 2635, 2639, 2649, 2658, 2663, 2667, 2668, 2674, 2681, 2689, 2697, 2713, 2726, 2734, 2741, 2749, 2754, 2755, 2784, 2807, 2813, 2842, 2850, 2859, 2866, 2871, 2877, 2884, 2891, 2897, 2900, 2929, 2934, 2939, 2945, 2950, 2956, 2958, 2986]
        #
        # SR_list =[]
        # s=range(180)
        # s_frames=np.array(s).tolist()
        #s_frames=[0,30,60,90,120,150]
        #SR_list=s_frames
        # print("s_a",int(action/15))
        print('s_frames', s_frames)
        print('SR',SR_list)
        final_results = Results()
        # 取s_frames中帧做检测的f1score
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        #all_time = 0
        # print(self.idx-self.goplen)
        # [0，30） [30,60)
        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                if frame_idx in SR_list:

                    self.l_result = self.h_result[frame_idx]
                    #self.l_result = self.result[frame_idx]
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                    # sr   1080 infer
                    time += 0.1724+0.0924+0.02
                else:
                    #for r in self.self.result[frame_idx]:
                    _d_result = Results()
                    #self.move(frame_idx)
                    # self.clear_sr(frame_idx)
                    # 就用lr infer 代替transfer+infer
                    self.l_result = self.result[frame_idx]

                    time += 0.0924+0.02
                    # for r in self.l_result:
                    #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    #     r = Region(frame_idx, x, y, w, h, conf, label,
                    #                0, origin="mpeg")
                    #     r = self.resize(r)
                    #     if self.isdelete(r):
                    #         _d_result.append(r)
                    # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]

                    for r in self.l_result:
                        label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                        #print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                #all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         # print(y)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         r = self.resize(r)
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         if self.isdelete(r):
            #             final_results.append(r)
            #     continue
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         final_results.append(r)
            #     all_time += self.times[frame_idx]
            #     continue
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    r = self.resize(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    if self.isdelete(r):
                        final_results.append(r)
#0.7868061224489794


            else:
                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            #all_time += self.times[frame_idx]
            # print('result',results)

        #final_results = merge_boxes_in_results(final_results.regions_dict, 0.5, 0.5)
        #self.show_results.combine_results(final_results)
        try:
            tp, fp, fn, _, _, _, f1 = evaluate(
                self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
                0.5, 0.5, 0.4, 0.4)
        except Exception:
            f1=0
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.04, 0.04)
        if SR_list!=s_frames:
            self.t_reward.append(f1)
        # final_results = Results()
        # # 取s_frames中帧做检测的f1score
        # #print('f1:',f1)
        # # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        # print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
        self.last_compute_time=time
        time_trans=self.bitrate_bitrate[self.last_bitrate]/self.bandwith[int(self.idx / 30-1)+N_CHUNK-1]
        time+=time_trans
        # reward = self.a1 * (f1)  - self.a2 * (1 if time>1 else 0)
        reward = self.a1 * (f1) - self.a2 * (1 if time > 1 else 0)
        # 排队时间
        # self.queue =max(0,(self.queue+time-1))
        ###
        # 计算前一个gop剩余feature差值
        new_bitrate = bitrate_thres[int(action % len(bitrate_thres))]
        self.diff_gop=self.diff_gop_dict[new_bitrate]
        # state_要拷贝新的chunk选的bitrate
        state_ = copy.deepcopy(self.states_dict[new_bitrate][int(self.idx / self.goplen)])
        # res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        # sun  31  60
        res_ = copy.deepcopy(self.res[self.idx+ 1:self.idx+self.goplen])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('余下的',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            # res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
            # [20 31)
            res_.insert(0, sum(self.res[self.last_sr_frame + 1:self.idx + 1]))

        else:
            res_.insert(0,0)
            state_.insert(0, 0)


        ###
        self.state = np.array(state_)
        # idx变成60
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == 2970:
            done = True
        else:
            done = False
        #print(len(res_), self.idx)

        # 下一个chunk

        self.features=self.features_dict[new_bitrate]
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        a=np.array(self.chunk_sizes)
        a=np.roll(a,1)
        a[0]=self.bitrate_bitrate[self.last_bitrate]
        self.chunk_sizes=a.tolist()
        a=np.array(self.chunk_times)
        a=np.roll(a,1)
        a[0]=self.chunk_sizes[0]/self.bandwith[int(self.idx/30-2)+N_CHUNK-1]
        # print(time_trans,a[0],"is equal")  相等
        self.chunk_times=a.tolist()
        state_.extend(self.chunk_sizes)
        state_.extend(self.chunk_times)
        state_.append(self.last_bitrate)
        state_.append(self.last_compute_time)
        print(self.idx)
        print('time ',time_trans,self.last_compute_time,time)
        print('f1',f1)
        # print('chunk_sizes',self.chunk_sizes)
        # print('chunk_times',self.chunk_times)
        # if self.idx>=2940:
        #     print("bitrate list:",self.bitrate_trace)
        # 这里的bitrate决定的是下一个chunk的bitrate，但是这一个chunk已经编码好了（并且取出了编码器信息），所以不太好修改
        # 根据action更新bitrate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.last_bitrate=new_bitrate
        self.bitrate_trace.append(self.last_bitrate)
        return np.array(state_), reward, done, None



class Envs3:

    def __init__(self, height, width, length, states_dict, diff_gop_dict, times, result,h_result,res, features_dict,chunksizes,bandwith):
        # self.args=create_args()
        self.last_qp=30
        with open('./I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
        #print(self.I_frame)
        self.environment_title='video_V0'
        #self.action_space=spaces.Discrete(75)
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )
        high=np.zeros(128+60,dtype=np.float32)
        high=np.array([np.finfo(np.float32).max for _ in high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states_dict=states_dict
        self.states = states_dict[self.last_qp]
        self.diff_gop_dict = diff_gop_dict
        self.diff_gop = diff_gop_dict[self.last_qp]

        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = None
        self._max_episode_steps=100
        self.t_reward=[]
        #self.model = Detector()
        # with open("dds_results.txt", "rb") as get_myprofile:
        #     h_result = pickle.load(get_myprofile)
        # self.ground_truth_dict = read_results_dict(ground_path)
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     1000, self.ground_truth_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.4, 0.4)
        # print(f1)
        #self.ground_truth_dict =h_result.regions_dict
        self.times = times
        self.result_dict = result
        self.show_results = Results()
        self.h_result_dict=h_result
        # gt是质量最好的540p  每帧SR之后 infer的结果
        self.ground_truth_dict=h_result[10]

        self.res=res
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features_dict = features_dict
        self.features=features_dict[self.last_qp]


        with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/models/pca.pkl","rb") as get_myprofile:
            self.pca=pickle.load(get_myprofile)
        # pca = PCA(n_components=128)  # 实例化
        # self.pca = pca.fit(np.array(self.features))  # 拟合模型
        self.srl=[]
        self.dtl=[]
        self.s_a=0
        self.d_a=0
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)
        self.qp_bitrate = chunksizes
        self.chunk_times=np.zeros((N_CHUNK))
        self.chunk_sizes=np.zeros((N_CHUNK))
        # self.chunk_sizes=[ 1.6027964834173045, 1.6027964834173045,
        # 1.6027964834173045,
        # 1.6027964834173045,
        # 1.5982735510189188,
        # 1.5982735510189188,
        #
        # 1.5982735510189188,
        # 1.5982735510189188
        # ]
        #
        # self.chunk_times=[self.qp_bitrate[self.last_qp]/i for i in self.chunk_sizes]

        self.bandwith=bandwith

        self.qp_trace=[]
        self.last_compute_time=0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.idx = 30
        self.queue = 0
        # self.states[0] 取一个gop的state  29张（不算第一帧）
        state_ = copy.deepcopy(self.states[0])
        res_ = copy.deepcopy(self.res[1:30])
        self.last_sr_frame=0
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        res_.insert(0,0)
        #print(len(res_),self.idx)
        #
        self.state = np.array(state_)
        # pca输入是两维
        random_integer = random.randint(0, 10)
        self.last_qp=qp_thres[random_integer % 5]
        print("reset random qp", self.last_qp)
        self.features=self.features_dict[self.last_qp]
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        # print(len(self.states[0]))
        # print(len(state_))

        # self.chunk_sizes[0]=self.qp_bitrate[self.last_qp]
        # # 这个后面需要改时间和size，现在只是简单模拟！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # self.chunk_times[0]=self.chunk_sizes[0]/self.bandwith[0]
        for i in range(N_CHUNK):
            self.chunk_sizes[i]=self.bandwith[N_CHUNK-1-i]

        # self.chunk_sizes = [1.6027964834173045, 1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.5982735510189188,
        #                     1.5982735510189188,
        #
        #                     1.5982735510189188,
        #                     1.5982735510189188
        #                     ]

        self.chunk_times = [self.qp_bitrate[self.last_qp] / i for i in self.chunk_sizes]

        self.qp_trace.append(self.last_qp)
        state_.extend(self.chunk_sizes)
        state_.extend(self.chunk_times)
        state_.append(self.last_qp)
        state_.append(self.last_compute_time)
        # state_ list:205
        # print(state_)
        return np.array(state_)
    # def delete(self):
    #     for r in self.l_result:
    #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist
    def move(self,frame_idx):
        if frame_idx + 1 in self.I_frame:
            for r in self.l_result:
                label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                #final_results.append(r)
        else:
            _d_result = Results()
            refer = find_refer(frame_idx + 1) - 1
            if refer + 1 == frame_idx:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    _x, _y = _x / 4, _y / 4
                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                    if _x != 0 and _y != 0:
                        _x = _x - __x
                        _y = _y - __y
                    _x, _y = _x / 4, _y / 4

                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            else:

                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                    _x, _y = _x / 4, _y / 4

                    # _x, _y =0,0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            self.l_result = _d_result.regions_dict[frame_idx]

    def resize(self,r):
        if (r.x+r.w)>1:
            r.w=1-r.x
        if (r.y + r.h) > 1:
            r.w = 1 - r.y
        r.x=max(0,r.x)
        r.y=max(0,r.y)
        r.h=max(0,r.h)
        r.w=max(0,r.w)
        r.x = min(1, r.x)
        r.y = min(1, r.y)
        return r

    def isexist(self,region,result):
        for r in result:
            if calc_iou(r,region)>0.1:
                return True
        return False
    def isdelete(self,r):

        if r.w==0 or r.h==0:
            return False
        return True

    def clear_sr(self,frame_idx):
        res=[]
        for region in self.l_result:
            flag=False
            for h_region in self.result[frame_idx]:
                if filter_bbox_group(region, h_region, 0.1) or h_region.w*h_region.h>0.005:
                    flag=True
                    break
            if not flag:
                res.append(region)
        self.l_result=res+self.result[frame_idx]
    def step(self, action):
        # 先通过a映射出edge difference阈值选出select_frames，再通过映射出的res阈值再selected_frames中选出SR list
        print("action",action,'\n','idx',self.idx,'\n','goplen',self.goplen)
        print("s sr qp", int(action/25),   int(action%25/5),action%5,self.last_qp,qp_thres[action%5])
        # self.idx  30 60 90...(下一个gop的第一帧帧号，用来索引本gop的末尾帧号,self.goplen 恒定30
        s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(int(action/25))], self.state, (self.idx - self.goplen),
                                                    self.goplen)
        #self.select.append(s_frames)r_thres[int(action/15)]
        SR_list=self.get_SRlist(s_frames,r_thres[int(action%25/5)])
        self.result=self.result_dict[self.last_qp]
        # 对应的qp  SR之后infer的结果
        self.h_result=self.h_result_dict[self.last_qp]
        # self.states=self.states_dict[self.last_qp]
        # self.features=self.features_dict[self.last_qp]
        # self.diff_gop=self.diff_gop_dict[self.last_qp]
        #SR_list=[]
        #s_frames=SR_list
        # s_frames = self.d_pro.baQtch_diff_noobj_last((action%0.1)*10, self.state, (self.idx - self.goplen),self.goplen)
        # s_frames = self.d_pro.batch_diff_noobj_last(action[0], self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        # SR_list=self.get_SRlist(s_frames,r_thres[int(action[1])])
        # SR_list = self.get_SRlist(s_frames,(action-action%0.1)*3)
        #print(action,(action%0.1)*10,(action-action%0.1)*3)
        self.srl+=SR_list
        self.dtl+=s_frames
        # s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres[self.d_a]), self.state, (self.idx - self.goplen),                                             self.goplen)
        # SR_list = self.get_SRlist(s_frames, r_thres[self.s_a])
        # s_frames = [0, 16, 29, 39, 50, 58, 67, 78, 87, 101, 115, 116, 135, 145, 162, 174, 203, 232, 261, 290, 294, 297, 302, 307,
        #            310, 318, 319, 330, 345, 348, 351, 354, 356, 365, 377, 382, 386, 394, 406, 412, 428, 435, 441, 446, 450, 454,
        #            457, 462, 464, 480, 493, 497, 500, 505, 511, 518, 522, 527, 532, 540, 549, 551, 567, 580, 588, 600, 604, 608,
        #            609, 629, 638, 661, 667, 671, 674, 679, 684, 689, 694, 696, 713, 725, 746, 754, 766, 780, 783, 796, 809, 812,
        #            821, 831, 841, 854, 869, 870, 882, 895, 899, 901, 903, 905, 907, 909, 912, 915, 918, 921, 925, 928, 932, 936,
        #            941, 948, 956, 957, 977, 986, 1003, 1015, 1044, 1052, 1058, 1065, 1072, 1073, 1101, 1102, 1111, 1122, 1131,
        #            1147, 1160, 1177, 1189, 1214, 1218, 1246, 1247, 1276, 1304, 1305, 1333, 1334, 1352, 1363, 1385, 1392, 1410,
        #            1421, 1440, 1450, 1468, 1479, 1496, 1508, 1526, 1537, 1557, 1566, 1587, 1595, 1614, 1624, 1644, 1653, 1674,
        #            1682, 1697, 1711, 1720, 1731, 1740, 1751, 1762, 1769, 1787, 1798, 1815, 1827, 1843, 1856, 1871, 1885, 1900,
        #            1914, 1929, 1943, 1958, 1972, 1988, 2001, 2017, 2030, 2046, 2059, 2075, 2088, 2103, 2117, 2118, 2120, 2122,
        #            2123, 2124, 2125, 2126, 2127, 2129, 2130, 2132, 2134, 2136, 2138, 2140, 2142, 2144, 2146, 2164, 2175, 2193,
        #            2204, 2214, 2225, 2233, 2252, 2262, 2268, 2274, 2280, 2284, 2288, 2291, 2306, 2320, 2333, 2347, 2349, 2361,
        #            2376, 2378, 2390, 2403, 2407, 2422, 2436, 2451, 2465, 2475, 2490, 2494, 2511, 2523, 2538, 2552, 2568, 2581,
        #            2597, 2610, 2627, 2639, 2655, 2668, 2681, 2696, 2697, 2708, 2726, 2745, 2755, 2770, 2784, 2803, 2813, 2830,
        # #            2842, 2845, 2848, 2850, 2854, 2859, 2862, 2867, 2871, 2893, 2900, 2922, 2929, 2946]
        # s_frames=[0, 29, 41, 55, 58, 63, 68, 73, 77, 82, 87, 96, 100, 105, 113, 116, 145, 174, 203, 232, 261, 290, 319, 348, 358, 376, 377, 406, 435, 444, 450, 458, 464, 474, 483, 493, 505, 517, 522, 531, 543, 551, 560, 570, 580, 590, 600, 609, 619, 629, 638, 648, 658, 667, 678, 689, 695, 696, 701, 710, 718, 725, 732, 741, 749, 750, 754, 762, 769, 776, 783, 807, 812, 823, 836, 841, 852, 866, 870, 877, 887, 896, 899, 902, 911, 919, 927, 928, 937, 947, 957, 968, 979, 986, 995, 1005, 1015, 1029, 1041, 1044, 1055, 1063, 1070, 1073, 1093, 1102, 1119, 1131, 1147, 1160, 1183, 1189, 1210, 1218, 1239, 1247, 1270, 1276, 1298, 1305, 1327, 1334, 1353, 1363, 1392, 1416, 1421, 1449, 1450, 1479, 1505, 1508, 1537, 1566, 1595, 1624, 1653, 1680, 1682, 1705, 1711, 1740, 1765, 1769, 1782, 1794, 1798, 1816, 1827, 1851, 1856, 1881, 1885, 1906, 1914, 1927, 1940, 1943, 1963, 1972, 1990, 2001, 2022, 2030, 2051, 2059, 2069, 2077, 2085, 2088, 2111, 2117, 2123, 2129, 2135, 2141, 2146, 2158, 2167, 2175, 2187, 2197, 2204, 2211, 2218, 2225, 2232, 2233, 2248, 2262, 2272, 2281, 2291, 2298, 2305, 2312, 2319, 2320, 2329, 2338, 2348, 2349, 2357, 2366, 2374, 2378, 2388, 2396, 2407, 2419, 2430, 2436, 2447, 2459, 2465, 2471, 2476, 2481, 2486, 2490, 2494, 2510, 2521, 2523, 2533, 2548, 2552, 2564, 2577, 2581, 2587, 2597, 2606, 2610, 2620, 2627, 2635, 2639, 2649, 2658, 2663, 2667, 2668, 2674, 2681, 2689, 2697, 2713, 2726, 2734, 2741, 2749, 2754, 2755, 2784, 2807, 2813, 2842, 2850, 2859, 2866, 2871, 2877, 2884, 2891, 2897, 2900, 2929, 2934, 2939, 2945, 2950, 2956, 2958, 2986]
        #
        # SR_list =[]
        # s=range(180)
        # s_frames=np.array(s).tolist()
        #s_frames=[0,30,60,90,120,150]
        #SR_list=s_frames
        # print("s_a",int(action/15))
        print('s_frames', s_frames)
        print('SR',SR_list)
        final_results = Results()
        # 取s_frames中帧做检测的f1score
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        #all_time = 0
        # print(self.idx-self.goplen)
        # [0，30） [30,60)
        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                if frame_idx in SR_list:

                    self.l_result = self.h_result[frame_idx]
                    #self.l_result = self.result[frame_idx]
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                    # sr   1080 infer
                    time += 0.1724+0.0924+0.02
                else:
                    #for r in self.self.result[frame_idx]:
                    _d_result = Results()
                    #self.move(frame_idx)
                    # self.clear_sr(frame_idx)
                    # 就用lr infer 代替transfer+infer
                    self.l_result = self.result[frame_idx]

                    time += 0.0924+0.02
                    # for r in self.l_result:
                    #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    #     r = Region(frame_idx, x, y, w, h, conf, label,
                    #                0, origin="mpeg")
                    #     r = self.resize(r)
                    #     if self.isdelete(r):
                    #         _d_result.append(r)
                    # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]

                    for r in self.l_result:
                        label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                        #print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                #all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         # print(y)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         r = self.resize(r)
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         if self.isdelete(r):
            #             final_results.append(r)
            #     continue
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         final_results.append(r)
            #     all_time += self.times[frame_idx]
            #     continue
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    r = self.resize(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    if self.isdelete(r):
                        final_results.append(r)
#0.7868061224489794


            else:
                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            #all_time += self.times[frame_idx]
            # print('result',results)

        #final_results = merge_boxes_in_results(final_results.regions_dict, 0.5, 0.5)
        #self.show_results.combine_results(final_results)
        try:
            tp, fp, fn, _, _, _, f1 = evaluate(
                self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
                0.5, 0.5, 0.4, 0.4)
        except Exception:
            f1=0
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.04, 0.04)
        if SR_list!=s_frames:
            self.t_reward.append(f1)
        # final_results = Results()
        # # 取s_frames中帧做检测的f1score
        # #print('f1:',f1)
        # # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        # print(name,'f1',f1,"all_f1",s_f1,'Select',s_frames,len(s_frames),'time',time)
        self.last_compute_time=time
        time_trans=self.qp_bitrate[self.last_qp]/self.bandwith[int(self.idx / 30-1)+N_CHUNK-1]
        time+=time_trans
        # reward = self.a1 * (f1)  - self.a2 * (1 if time>1 else 0)
        reward = self.a1 * (f1) - self.a2 * (1 if time > 1 else 0)
        # 排队时间
        # self.queue =max(0,(self.queue+time-1))
        ###
        # 计算前一个gop剩余feature差值
        new_qp = qp_thres[int(action % 5)]
        self.diff_gop=self.diff_gop_dict[new_qp]
        # state_要拷贝新的chunk选的qp
        state_ = copy.deepcopy(self.states_dict[new_qp][int(self.idx / self.goplen)])
        # print("state_.shape", np.array(state_).shape)
        # res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        # sun  31  60
        res_ = copy.deepcopy(self.res[self.idx+ 1:self.idx+self.goplen])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('余下的',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            # res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
            # [20 31)
            res_.insert(0, sum(self.res[self.last_sr_frame + 1:self.idx + 1]))

        else:
            res_.insert(0,0)
            state_.insert(0, 0)
        # print("state_.shape",np.array(state_).shape)
        ###
        self.state = np.array(state_)
        # idx变成60
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == 2970:
            done = True
        else:
            done = False
        #print(len(res_), self.idx)

        # 下一个chunk

        self.features=self.features_dict[new_qp]
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        a=np.array(self.chunk_sizes)
        a=np.roll(a,1)
        a[0]=self.qp_bitrate[self.last_qp]
        self.chunk_sizes=a.tolist()
        a=np.array(self.chunk_times)
        a=np.roll(a,1)
        a[0]=self.chunk_sizes[0]/self.bandwith[int(self.idx/30-2)+N_CHUNK-1]
        # print(time_trans,a[0],"is equal")  相等
        self.chunk_times=a.tolist()
        state_.extend(self.chunk_sizes)
        state_.extend(self.chunk_times)
        state_.append(self.last_qp)
        state_.append(self.last_compute_time)
        print(self.idx)
        print('time ',time_trans,self.last_compute_time,time)
        print('f1',f1)
        # print('chunk_sizes',self.chunk_sizes)
        # print('chunk_times',self.chunk_times)
        # if self.idx>=2940:
        #     print("qp list:",self.qp_trace)
        # 这里的qp决定的是下一个chunk的qp，但是这一个chunk已经编码好了（并且取出了编码器信息），所以不太好修改
        # 根据action更新qp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.last_qp=new_qp
        self.qp_trace.append(self.last_qp)
        return np.array(state_), reward, done, None



class Envs5:

    def __init__(self, height, width, length, states_dict, diff_gop_dict, times, result,h_result,res, features_dict,chunksizes,bandwith):
        # self.args=create_args()
        self.last_qp=30
        with open('./I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
        #print(self.I_frame)
        self.environment_title='video_V0'
        #self.action_space=spaces.Discrete(75)
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )
        high=np.zeros(128+60,dtype=np.float32)
        high=np.array([np.finfo(np.float32).max for _ in high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states_dict=states_dict
        self.states = states_dict[self.last_qp]
        self.diff_gop_dict = diff_gop_dict
        self.diff_gop = diff_gop_dict[self.last_qp]

        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = None
        self._max_episode_steps=100
        self.t_reward=[]
        #self.model = Detector()
        # with open("dds_results.txt", "rb") as get_myprofile:
        #     h_result = pickle.load(get_myprofile)
        # self.ground_truth_dict = read_results_dict(ground_path)
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     1000, self.ground_truth_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.4, 0.4)
        # print(f1)
        #self.ground_truth_dict =h_result.regions_dict
        self.times = times
        self.result_dict = result
        self.show_results = Results()
        self.h_result_dict=h_result
        # gt是质量最好的540p  每帧SR之后 infer的结果
        self.ground_truth_dict=h_result[10]

        self.res=res
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features_dict = features_dict
        self.features=features_dict[self.last_qp]


        with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/models/pca.pkl","rb") as get_myprofile:
            self.pca=pickle.load(get_myprofile)
        # pca = PCA(n_components=128)  # 实例化
        # self.pca = pca.fit(np.array(self.features))  # 拟合模型
        self.srl=[]
        self.dtl=[]
        self.s_a=0
        self.d_a=0
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)
        self.qp_bitrate = chunksizes
        self.chunk_times=np.zeros((N_CHUNK))
        self.chunk_sizes=np.zeros((N_CHUNK))
        # self.chunk_sizes=[ 1.6027964834173045, 1.6027964834173045,
        # 1.6027964834173045,
        # 1.6027964834173045,
        # 1.5982735510189188,
        # 1.5982735510189188,
        #
        # 1.5982735510189188,
        # 1.5982735510189188
        # ]
        #
        # self.chunk_times=[self.qp_bitrate[self.last_qp]/i for i in self.chunk_sizes]

        self.bandwith=bandwith
        self.qp_trace=[]
        self.last_compute_time=0
        self.f1_list=[]
        self.reward_list=[]
        self.reward_sum = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.idx = 30
        self.queue = 0
        # self.states[0] 取一个gop的state  29张（不算第一帧）
        import copy
        state_ = copy.deepcopy(self.states[0])
        res_ = copy.deepcopy(self.res[1:30])
        self.last_sr_frame=0
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        res_.insert(0,0)
        #print(len(res_),self.idx)
        #
        self.state = np.array(state_)
        # pca输入是两维
        random_integer = random.randint(0, 10)
        self.last_qp=qp_thres[random_integer % 5]
        print("reset random qp", self.last_qp)
        self.features=self.features_dict[self.last_qp]
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        # print(len(self.states[0]))
        # print(len(state_))

        # self.chunk_sizes[0]=self.qp_bitrate[self.last_qp]
        # # 这个后面需要改时间和size，现在只是简单模拟！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # self.chunk_times[0]=self.chunk_sizes[0]/self.bandwith[0]
        for i in range(N_CHUNK):
            self.chunk_sizes[i]=self.bandwith[N_CHUNK-1-i]

        # self.chunk_sizes = [1.6027964834173045, 1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.5982735510189188,
        #                     1.5982735510189188,
        #
        #                     1.5982735510189188,
        #                     1.5982735510189188
        #                     ]

        self.chunk_times = [self.qp_bitrate[self.last_qp] / i for i in self.chunk_sizes]

        self.qp_trace.append(self.last_qp)
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)

        state_.append(self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(self.qp_bitrate[self.last_qp]/self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.last_compute_time)
        # state_ list:205
        # print(state_)
        return np.array(state_)
    # def delete(self):
    #     for r in self.l_result:
    #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist
    def move(self,frame_idx):
        if frame_idx + 1 in self.I_frame:
            for r in self.l_result:
                label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                #final_results.append(r)
        else:
            _d_result = Results()
            refer = find_refer(frame_idx + 1) - 1
            if refer + 1 == frame_idx:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    _x, _y = _x / 4, _y / 4
                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                    if _x != 0 and _y != 0:
                        _x = _x - __x
                        _y = _y - __y
                    _x, _y = _x / 4, _y / 4

                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            else:

                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                    _x, _y = _x / 4, _y / 4

                    # _x, _y =0,0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            self.l_result = _d_result.regions_dict[frame_idx]

    def resize(self,r):
        if (r.x+r.w)>1:
            r.w=1-r.x
        if (r.y + r.h) > 1:
            r.w = 1 - r.y
        r.x=max(0,r.x)
        r.y=max(0,r.y)
        r.h=max(0,r.h)
        r.w=max(0,r.w)
        r.x = min(1, r.x)
        r.y = min(1, r.y)
        return r

    def isexist(self,region,result):
        for r in result:
            if calc_iou(r,region)>0.1:
                return True
        return False
    def isdelete(self,r):

        if r.w==0 or r.h==0:
            return False
        return True

    def clear_sr(self,frame_idx):
        res=[]
        for region in self.l_result:
            flag=False
            for h_region in self.result[frame_idx]:
                if filter_bbox_group(region, h_region, 0.1) or h_region.w*h_region.h>0.005:
                    flag=True
                    break
            if not flag:
                res.append(region)
        self.l_result=res+self.result[frame_idx]
    def step(self, action):
        # 先通过a映射出edge difference阈值选出select_frames，再通过映射出的res阈值再selected_frames中选出SR list
        # self.idx  30 60 90...(下一个gop的第一帧帧号，用来索引本gop的末尾帧号,self.goplen 恒定30
        s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(int(action/25))], self.state, (self.idx - self.goplen),
                                                    self.goplen)
        #self.select.append(s_frames)r_thres[int(action/15)]
        SR_list=self.get_SRlist(s_frames,r_thres[int(action%25/5)])
        self.result=self.result_dict[self.last_qp]
        # 对应的qp  SR之后infer的结果
        self.h_result=self.h_result_dict[self.last_qp]
        # self.states=self.states_dict[self.last_qp]
        # self.features=self.features_dict[self.last_qp]
        # self.diff_gop=self.diff_gop_dict[self.last_qp]
        #SR_list=[]
        #s_frames=SR_list
        # s_frames = self.d_pro.baQtch_diff_noobj_last((action%0.1)*10, self.state, (self.idx - self.goplen),self.goplen)
        # s_frames = self.d_pro.batch_diff_noobj_last(action[0], self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        # SR_list=self.get_SRlist(s_frames,r_thres[int(action[1])])
        # SR_list = self.get_SRlist(s_frames,(action-action%0.1)*3)
        #print(action,(action%0.1)*10,(action-action%0.1)*3)
        self.srl+=SR_list
        self.dtl+=s_frames
        # s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres[self.d_a]), self.state, (self.idx - self.goplen),                                             self.goplen)
        # SR_list = self.get_SRlist(s_frames, r_thres[self.s_a])
        # s_frames = [0, 16, 29, 39, 50, 58, 67, 78, 87, 101, 115, 116, 135, 145, 162, 174, 203, 232, 261, 290, 294, 297, 302, 307,
        #            310, 318, 319, 330, 345, 348, 351, 354, 356, 365, 377, 382, 386, 394, 406, 412, 428, 435, 441, 446, 450, 454,
        #            457, 462, 464, 480, 493, 497, 500, 505, 511, 518, 522, 527, 532, 540, 549, 551, 567, 580, 588, 600, 604, 608,
        #            609, 629, 638, 661, 667, 671, 674, 679, 684, 689, 694, 696, 713, 725, 746, 754, 766, 780, 783, 796, 809, 812,
        #            821, 831, 841, 854, 869, 870, 882, 895, 899, 901, 903, 905, 907, 909, 912, 915, 918, 921, 925, 928, 932, 936,
        #            941, 948, 956, 957, 977, 986, 1003, 1015, 1044, 1052, 1058, 1065, 1072, 1073, 1101, 1102, 1111, 1122, 1131,
        #            1147, 1160, 1177, 1189, 1214, 1218, 1246, 1247, 1276, 1304, 1305, 1333, 1334, 1352, 1363, 1385, 1392, 1410,
        #            1421, 1440, 1450, 1468, 1479, 1496, 1508, 1526, 1537, 1557, 1566, 1587, 1595, 1614, 1624, 1644, 1653, 1674,
        #            1682, 1697, 1711, 1720, 1731, 1740, 1751, 1762, 1769, 1787, 1798, 1815, 1827, 1843, 1856, 1871, 1885, 1900,
        #            1914, 1929, 1943, 1958, 1972, 1988, 2001, 2017, 2030, 2046, 2059, 2075, 2088, 2103, 2117, 2118, 2120, 2122,
        #            2123, 2124, 2125, 2126, 2127, 2129, 2130, 2132, 2134, 2136, 2138, 2140, 2142, 2144, 2146, 2164, 2175, 2193,
        #            2204, 2214, 2225, 2233, 2252, 2262, 2268, 2274, 2280, 2284, 2288, 2291, 2306, 2320, 2333, 2347, 2349, 2361,
        #            2376, 2378, 2390, 2403, 2407, 2422, 2436, 2451, 2465, 2475, 2490, 2494, 2511, 2523, 2538, 2552, 2568, 2581,
        #            2597, 2610, 2627, 2639, 2655, 2668, 2681, 2696, 2697, 2708, 2726, 2745, 2755, 2770, 2784, 2803, 2813, 2830,
        # #            2842, 2845, 2848, 2850, 2854, 2859, 2862, 2867, 2871, 2893, 2900, 2922, 2929, 2946]
        # s_frames=[0, 29, 41, 55, 58, 63, 68, 73, 77, 82, 87, 96, 100, 105, 113, 116, 145, 174, 203, 232, 261, 290, 319, 348, 358, 376, 377, 406, 435, 444, 450, 458, 464, 474, 483, 493, 505, 517, 522, 531, 543, 551, 560, 570, 580, 590, 600, 609, 619, 629, 638, 648, 658, 667, 678, 689, 695, 696, 701, 710, 718, 725, 732, 741, 749, 750, 754, 762, 769, 776, 783, 807, 812, 823, 836, 841, 852, 866, 870, 877, 887, 896, 899, 902, 911, 919, 927, 928, 937, 947, 957, 968, 979, 986, 995, 1005, 1015, 1029, 1041, 1044, 1055, 1063, 1070, 1073, 1093, 1102, 1119, 1131, 1147, 1160, 1183, 1189, 1210, 1218, 1239, 1247, 1270, 1276, 1298, 1305, 1327, 1334, 1353, 1363, 1392, 1416, 1421, 1449, 1450, 1479, 1505, 1508, 1537, 1566, 1595, 1624, 1653, 1680, 1682, 1705, 1711, 1740, 1765, 1769, 1782, 1794, 1798, 1816, 1827, 1851, 1856, 1881, 1885, 1906, 1914, 1927, 1940, 1943, 1963, 1972, 1990, 2001, 2022, 2030, 2051, 2059, 2069, 2077, 2085, 2088, 2111, 2117, 2123, 2129, 2135, 2141, 2146, 2158, 2167, 2175, 2187, 2197, 2204, 2211, 2218, 2225, 2232, 2233, 2248, 2262, 2272, 2281, 2291, 2298, 2305, 2312, 2319, 2320, 2329, 2338, 2348, 2349, 2357, 2366, 2374, 2378, 2388, 2396, 2407, 2419, 2430, 2436, 2447, 2459, 2465, 2471, 2476, 2481, 2486, 2490, 2494, 2510, 2521, 2523, 2533, 2548, 2552, 2564, 2577, 2581, 2587, 2597, 2606, 2610, 2620, 2627, 2635, 2639, 2649, 2658, 2663, 2667, 2668, 2674, 2681, 2689, 2697, 2713, 2726, 2734, 2741, 2749, 2754, 2755, 2784, 2807, 2813, 2842, 2850, 2859, 2866, 2871, 2877, 2884, 2891, 2897, 2900, 2929, 2934, 2939, 2945, 2950, 2956, 2958, 2986]
        #
        # SR_list =[]
        # s=range(180)
        # s_frames=np.array(s).tolist()
        #s_frames=[0,30,60,90,120,150]
        #SR_list=s_frames
        # print("s_a",int(action/15))

        final_results = Results()
        # 取s_frames中帧做检测的f1score
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        #all_time = 0
        # print(self.idx-self.goplen)
        # [0，30） [30,60)
        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                if frame_idx in SR_list:

                    self.l_result = self.h_result[frame_idx]
                    #self.l_result = self.result[frame_idx]
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                    # sr   1080 infer
                    time += 0.1724+0.0924+0.02
                #     0.2848
                else:
                    #for r in self.self.result[frame_idx]:
                    _d_result = Results()
                    #self.move(frame_idx)
                    # self.clear_sr(frame_idx)
                    # 就用lr infer 代替transfer+infer
                    self.l_result = self.result[frame_idx]

                    time += 0.0924+0.02
                    # 0.1124
                    # for r in self.l_result:
                    #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    #     r = Region(frame_idx, x, y, w, h, conf, label,
                    #                0, origin="mpeg")
                    #     r = self.resize(r)
                    #     if self.isdelete(r):
                    #         _d_result.append(r)
                    # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]

                    for r in self.l_result:
                        label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                        #print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                #all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         # print(y)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         r = self.resize(r)
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         if self.isdelete(r):
            #             final_results.append(r)
            #     continue
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         final_results.append(r)
            #     all_time += self.times[frame_idx]
            #     continue
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    r = self.resize(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    if self.isdelete(r):
                        final_results.append(r)
#0.7868061224489794


            else:
                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            #all_time += self.times[frame_idx]
            # print('result',results)

        #final_results = merge_boxes_in_results(final_results.regions_dict, 0.5, 0.5)
        #self.show_results.combine_results(final_results)
        try:
            tp, fp, fn, _, _, _, f1 = evaluate(
                self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
                0.5, 0.5, 0.4, 0.4)
        except Exception:
            f1=0
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.04, 0.04)
        if SR_list!=s_frames:
            self.t_reward.append(f1)
        # final_results = Results()
        # # 取s_frames中帧做检测的f1score
        # #print('f1:',f1)
        # # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        self.f1_list.append(f1)
        self.last_compute_time=time
        time_trans=self.qp_bitrate[self.last_qp]/self.bandwith[int(self.idx / 30-1)+N_CHUNK-1]
        time+=time_trans
        # reward = self.a1 * (f1)  - self.a2 * (1 if time>1 else 0)
        reward = self.a1 * (f1) - self.a2 * (1 if time > 1 else 0)
        tuple=(time_trans,self.last_compute_time,time,self.bandwith[int(self.idx / 30-1)+N_CHUNK-1],self.qp_bitrate[self.last_qp],self.last_qp,f1,reward)
        self.reward_list.append(tuple)
        self.reward_sum += reward
        # 排队时间
        # self.queue =max(0,(self.queue+time-1))
        ###
        # 计算前一个gop剩余feature差值
        new_qp = qp_thres[int(action % 5)]
        self.diff_gop=self.diff_gop_dict[new_qp]
        # state_要拷贝新的chunk选的qp
        import copy
        state_ = copy.deepcopy(self.states_dict[new_qp][int(self.idx / self.goplen)])
        # print("state_.shape", np.array(state_).shape)
        # res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        # sun  31  60
        res_ = copy.deepcopy(self.res[self.idx+ 1:self.idx+self.goplen])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('余下的',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            # res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
            # [20 31)
            res_.insert(0, sum(self.res[self.last_sr_frame + 1:self.idx + 1]))

        else:
            res_.insert(0,0)
            state_.insert(0, 0)
        # print("state_.shape",np.array(state_).shape)
        ###
        self.state = np.array(state_)
        # idx变成60
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == 2970:
            done = True
        else:
            done = False
        #print(len(res_), self.idx)

        # 下一个chunk

        self.features=self.features_dict[new_qp]
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        a=np.array(self.chunk_sizes)
        a=np.roll(a,1)
        a[0]=self.qp_bitrate[self.last_qp]
        self.chunk_sizes=a.tolist()
        a=np.array(self.chunk_times)
        a=np.roll(a,1)
        a[0]=self.chunk_sizes[0]/self.bandwith[int(self.idx/30-2)+N_CHUNK-1]
        # print(time_trans,a[0],"is equal")  相等
        self.chunk_times=a.tolist()
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)
        state_.append(self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(time_trans)
        state_.append(self.last_compute_time)
        print("action", action, '\n', 'idx', self.idx, '\n', 'goplen', self.goplen)
        print("s sr q_a last_qp qp", int(action / 25), int(action % 25 / 5), action % 5, self.last_qp,
              qp_thres[action % 5])
        print('s_frames', s_frames)
        print('SR',SR_list)
        print("last_qp bitrate bandwidth",self.last_qp,self.qp_bitrate[self.last_qp] ,self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        print('time ',time_trans,self.last_compute_time,time)
        print('f1',f1)
        # print('chunk_sizes',self.chunk_sizes)
        # print('chunk_times',self.chunk_times)
        # if self.idx>=2940:
        #     print("qp list:",self.qp_trace)
        # 这里的qp决定的是下一个chunk的qp，但是这一个chunk已经编码好了（并且取出了编码器信息），所以不太好修改
        # 根据action更新qp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.last_qp=new_qp
        self.qp_trace.append(self.last_qp)
        return np.array(state_), reward, done, None

    def evaluate_(self,model_name):
        # gt 和 待检测帧号不匹配没关系，dds evaluate函数会从0~maxfid搜索帧号是否存在于待检测文件中
        print(self.idx)
        print(self.reward_list)
        print("f1_mean", mean(self.f1_list))
        print(" total reward", self.reward_sum)
        # print('f1_all', f1, " total reward", self.reward_sum)
        band_ = []
        qp_ = []
        # tuple = (time_trans, self.last_compute_time, time, self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1],
        #          self.qp_bitrate[self.last_qp], self.last_qp, f1, reward)
        reward_=[]
        for _, _, _, band, _,qp,_, reward in self.reward_list:
            band_.append(band)
            qp_.append(self.qp_bitrate[qp])
            reward_.append(reward)

        # band_=Normalize(band_)
        # qp_=Normalize(qp_)

        print(band_)
        print(qp_)
        x = np.arange(len(qp_))
        plt.figure(1)
        plt.plot(x, band_, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='Bandwidth')
        plt.plot(x, qp_, color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Adaptive qp")
        plt.ylabel('Bandwidth (Mbps)')
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"band_qp_{model_name}.png")
        plt.show()

        plt.figure(2)
        plt.plot(x, self.f1_list, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='F1-score')
        plt.plot(x, reward_, color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Reward")
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"f1_reward_{model_name}.png")
        plt.show()





class Envs6:

    def __init__(self, height, width, length, states_dict, diff_gop_dict, times, result,h_result,res, features_dict,chunksizes,bandwith):
        # self.args=create_args()
        self.last_qp=30
        with open('./I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
        #print(self.I_frame)
        self.environment_title='video_V0'
        #self.action_space=spaces.Discrete(75)
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )
        high=np.zeros(128+60,dtype=np.float32)
        high=np.array([np.finfo(np.float32).max for _ in high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states_dict=states_dict
        self.states = states_dict[self.last_qp]
        self.diff_gop_dict = diff_gop_dict
        self.diff_gop = diff_gop_dict[self.last_qp]

        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = None
        self._max_episode_steps=100
        self.t_reward=[]
        #self.model = Detector()
        # with open("dds_results.txt", "rb") as get_myprofile:
        #     h_result = pickle.load(get_myprofile)
        # self.ground_truth_dict = read_results_dict(ground_path)
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     1000, self.ground_truth_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.4, 0.4)
        # print(f1)
        #self.ground_truth_dict =h_result.regions_dict
        self.times = times
        self.result_dict = result
        self.show_results = Results()
        self.h_result_dict=h_result
        # gt是质量最好的540p  每帧SR之后 infer的结果
        self.ground_truth_dict=h_result[10]

        self.res=res
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features_dict = features_dict
        self.features=features_dict[self.last_qp]


        with open("D:/workplace/DRL_SR_Infer/models/pca.pkl","rb") as get_myprofile:
            self.pca=pickle.load(get_myprofile)
        # pca = PCA(n_components=128)  # 实例化
        # self.pca = pca.fit(np.array(self.features))  # 拟合模型
        self.srl=[]
        self.dtl=[]
        self.s_a=0
        self.d_a=0
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)
        self.qp_bitrate = chunksizes
        self.chunk_times=np.zeros((N_CHUNK))
        self.chunk_sizes=np.zeros((N_CHUNK))
        # self.chunk_sizes=[ 1.6027964834173045, 1.6027964834173045,
        # 1.6027964834173045,
        # 1.6027964834173045,
        # 1.5982735510189188,
        # 1.5982735510189188,
        #
        # 1.5982735510189188,
        # 1.5982735510189188
        # ]
        #
        # self.chunk_times=[self.qp_bitrate[self.last_qp]/i for i in self.chunk_sizes]

        self.bandwith=bandwith
        self.qp_trace=[]
        self.last_compute_time=0
        self.f1_list=[]
        self.reward_list=[]
        self.reward_sum = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.idx = 30
        self.queue = 0
        # self.states[0] 取一个gop的state  29张（不算第一帧）
        import copy
        state_ = copy.deepcopy(self.states[0])
        res_ = copy.deepcopy(self.res[1:30])
        self.last_sr_frame=0
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        res_.insert(0,0)
        #print(len(res_),self.idx)
        #
        self.state = np.array(state_)
        # pca输入是两维
        random_integer = random.randint(0, 10)
        self.last_qp=qp_thres[random_integer % 5]
        print("reset random qp", self.last_qp)
        self.features=self.features_dict[self.last_qp]
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        # print(len(self.states[0]))
        # print(len(state_))

        # self.chunk_sizes[0]=self.qp_bitrate[self.last_qp]
        # # 这个后面需要改时间和size，现在只是简单模拟！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # self.chunk_times[0]=self.chunk_sizes[0]/self.bandwith[0]
        for i in range(N_CHUNK):
            self.chunk_sizes[i]=self.bandwith[N_CHUNK-1-i]

        # self.chunk_sizes = [1.6027964834173045, 1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.5982735510189188,
        #                     1.5982735510189188,
        #
        #                     1.5982735510189188,
        #                     1.5982735510189188
        #                     ]

        self.chunk_times = [self.qp_bitrate[self.last_qp] / i for i in self.chunk_sizes]

        self.qp_trace.append(self.last_qp)
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)

        state_.append(self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(self.qp_bitrate[self.last_qp]/self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.last_compute_time)
        # state_ list:205
        # print(state_)
        return np.array(state_)
    # def delete(self):
    #     for r in self.l_result:
    #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist
    def move(self,frame_idx):
        if frame_idx + 1 in self.I_frame:
            for r in self.l_result:
                label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                #final_results.append(r)
        else:
            _d_result = Results()
            refer = find_refer(frame_idx + 1) - 1
            if refer + 1 == frame_idx:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    _x, _y = _x / 4, _y / 4
                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                    if _x != 0 and _y != 0:
                        _x = _x - __x
                        _y = _y - __y
                    _x, _y = _x / 4, _y / 4

                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            else:

                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                    _x, _y = _x / 4, _y / 4

                    # _x, _y =0,0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            self.l_result = _d_result.regions_dict[frame_idx]

    def resize(self,r):
        if (r.x+r.w)>1:
            r.w=1-r.x
        if (r.y + r.h) > 1:
            r.w = 1 - r.y
        r.x=max(0,r.x)
        r.y=max(0,r.y)
        r.h=max(0,r.h)
        r.w=max(0,r.w)
        r.x = min(1, r.x)
        r.y = min(1, r.y)
        return r

    def isexist(self,region,result):
        for r in result:
            if calc_iou(r,region)>0.1:
                return True
        return False
    def isdelete(self,r):

        if r.w==0 or r.h==0:
            return False
        return True

    def clear_sr(self,frame_idx):
        res=[]
        for region in self.l_result:
            flag=False
            for h_region in self.result[frame_idx]:
                if filter_bbox_group(region, h_region, 0.1) or h_region.w*h_region.h>0.005:
                    flag=True
                    break
            if not flag:
                res.append(region)
        self.l_result=res+self.result[frame_idx]
    def step(self, action):
        # 先通过a映射出edge difference阈值选出select_frames，再通过映射出的res阈值再selected_frames中选出SR list
        # self.idx  30 60 90...(下一个gop的第一帧帧号，用来索引本gop的末尾帧号,self.goplen 恒定30
        #
        thre_infer=0.7*action[0]+0.05
        thre_sr=3*action[1]
        thre_qp=20*action[2]+20
        s_frames = self.d_pro.batch_diff_noobj_last(thre_infer, self.state,
                                                    (self.idx - self.goplen),
                                                    self.goplen)
        # self.select.append(s_frames)r_thres[int(action/15)]
        SR_list = self.get_SRlist(s_frames, thre_sr)

        #
        # s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(int(action/25))], self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        # #self.select.append(s_frames)r_thres[int(action/15)]
        # SR_list=self.get_SRlist(s_frames,r_thres[int(action%25/5)])
        self.result=self.result_dict[self.last_qp]
        # 对应的qp  SR之后infer的结果
        self.h_result=self.h_result_dict[self.last_qp]
        # self.states=self.states_dict[self.last_qp]
        # self.features=self.features_dict[self.last_qp]
        # self.diff_gop=self.diff_gop_dict[self.last_qp]
        #SR_list=[]
        #s_frames=SR_list
        # s_frames = self.d_pro.baQtch_diff_noobj_last((action%0.1)*10, self.state, (self.idx - self.goplen),self.goplen)
        # s_frames = self.d_pro.batch_diff_noobj_last(action[0], self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        # SR_list=self.get_SRlist(s_frames,r_thres[int(action[1])])
        # SR_list = self.get_SRlist(s_frames,(action-action%0.1)*3)
        #print(action,(action%0.1)*10,(action-action%0.1)*3)
        self.srl+=SR_list
        self.dtl+=s_frames
        # s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres[self.d_a]), self.state, (self.idx - self.goplen),                                             self.goplen)
        # SR_list = self.get_SRlist(s_frames, r_thres[self.s_a])
        # s_frames = [0, 16, 29, 39, 50, 58, 67, 78, 87, 101, 115, 116, 135, 145, 162, 174, 203, 232, 261, 290, 294, 297, 302, 307,
        #            310, 318, 319, 330, 345, 348, 351, 354, 356, 365, 377, 382, 386, 394, 406, 412, 428, 435, 441, 446, 450, 454,
        #            457, 462, 464, 480, 493, 497, 500, 505, 511, 518, 522, 527, 532, 540, 549, 551, 567, 580, 588, 600, 604, 608,
        #            609, 629, 638, 661, 667, 671, 674, 679, 684, 689, 694, 696, 713, 725, 746, 754, 766, 780, 783, 796, 809, 812,
        #            821, 831, 841, 854, 869, 870, 882, 895, 899, 901, 903, 905, 907, 909, 912, 915, 918, 921, 925, 928, 932, 936,
        #            941, 948, 956, 957, 977, 986, 1003, 1015, 1044, 1052, 1058, 1065, 1072, 1073, 1101, 1102, 1111, 1122, 1131,
        #            1147, 1160, 1177, 1189, 1214, 1218, 1246, 1247, 1276, 1304, 1305, 1333, 1334, 1352, 1363, 1385, 1392, 1410,
        #            1421, 1440, 1450, 1468, 1479, 1496, 1508, 1526, 1537, 1557, 1566, 1587, 1595, 1614, 1624, 1644, 1653, 1674,
        #            1682, 1697, 1711, 1720, 1731, 1740, 1751, 1762, 1769, 1787, 1798, 1815, 1827, 1843, 1856, 1871, 1885, 1900,
        #            1914, 1929, 1943, 1958, 1972, 1988, 2001, 2017, 2030, 2046, 2059, 2075, 2088, 2103, 2117, 2118, 2120, 2122,
        #            2123, 2124, 2125, 2126, 2127, 2129, 2130, 2132, 2134, 2136, 2138, 2140, 2142, 2144, 2146, 2164, 2175, 2193,
        #            2204, 2214, 2225, 2233, 2252, 2262, 2268, 2274, 2280, 2284, 2288, 2291, 2306, 2320, 2333, 2347, 2349, 2361,
        #            2376, 2378, 2390, 2403, 2407, 2422, 2436, 2451, 2465, 2475, 2490, 2494, 2511, 2523, 2538, 2552, 2568, 2581,
        #            2597, 2610, 2627, 2639, 2655, 2668, 2681, 2696, 2697, 2708, 2726, 2745, 2755, 2770, 2784, 2803, 2813, 2830,
        # #            2842, 2845, 2848, 2850, 2854, 2859, 2862, 2867, 2871, 2893, 2900, 2922, 2929, 2946]
        # s_frames=[0, 29, 41, 55, 58, 63, 68, 73, 77, 82, 87, 96, 100, 105, 113, 116, 145, 174, 203, 232, 261, 290, 319, 348, 358, 376, 377, 406, 435, 444, 450, 458, 464, 474, 483, 493, 505, 517, 522, 531, 543, 551, 560, 570, 580, 590, 600, 609, 619, 629, 638, 648, 658, 667, 678, 689, 695, 696, 701, 710, 718, 725, 732, 741, 749, 750, 754, 762, 769, 776, 783, 807, 812, 823, 836, 841, 852, 866, 870, 877, 887, 896, 899, 902, 911, 919, 927, 928, 937, 947, 957, 968, 979, 986, 995, 1005, 1015, 1029, 1041, 1044, 1055, 1063, 1070, 1073, 1093, 1102, 1119, 1131, 1147, 1160, 1183, 1189, 1210, 1218, 1239, 1247, 1270, 1276, 1298, 1305, 1327, 1334, 1353, 1363, 1392, 1416, 1421, 1449, 1450, 1479, 1505, 1508, 1537, 1566, 1595, 1624, 1653, 1680, 1682, 1705, 1711, 1740, 1765, 1769, 1782, 1794, 1798, 1816, 1827, 1851, 1856, 1881, 1885, 1906, 1914, 1927, 1940, 1943, 1963, 1972, 1990, 2001, 2022, 2030, 2051, 2059, 2069, 2077, 2085, 2088, 2111, 2117, 2123, 2129, 2135, 2141, 2146, 2158, 2167, 2175, 2187, 2197, 2204, 2211, 2218, 2225, 2232, 2233, 2248, 2262, 2272, 2281, 2291, 2298, 2305, 2312, 2319, 2320, 2329, 2338, 2348, 2349, 2357, 2366, 2374, 2378, 2388, 2396, 2407, 2419, 2430, 2436, 2447, 2459, 2465, 2471, 2476, 2481, 2486, 2490, 2494, 2510, 2521, 2523, 2533, 2548, 2552, 2564, 2577, 2581, 2587, 2597, 2606, 2610, 2620, 2627, 2635, 2639, 2649, 2658, 2663, 2667, 2668, 2674, 2681, 2689, 2697, 2713, 2726, 2734, 2741, 2749, 2754, 2755, 2784, 2807, 2813, 2842, 2850, 2859, 2866, 2871, 2877, 2884, 2891, 2897, 2900, 2929, 2934, 2939, 2945, 2950, 2956, 2958, 2986]
        #
        # SR_list =[]
        # s=range(180)
        # s_frames=np.array(s).tolist()
        #s_frames=[0,30,60,90,120,150]
        #SR_list=s_frames
        # print("s_a",int(action/15))

        final_results = Results()
        # 取s_frames中帧做检测的f1score
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        #all_time = 0
        # print(self.idx-self.goplen)
        # [0，30） [30,60)
        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                if frame_idx in SR_list:

                    self.l_result = self.h_result[frame_idx]
                    #self.l_result = self.result[frame_idx]
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                    # sr   1080 infer
                    time += 0.1724+0.0924+0.02
                #     0.2848
                else:
                    #for r in self.self.result[frame_idx]:
                    _d_result = Results()
                    #self.move(frame_idx)
                    # self.clear_sr(frame_idx)
                    # 就用lr infer 代替transfer+infer
                    self.l_result = self.result[frame_idx]

                    time += 0.0924+0.02
                    # 0.1124
                    # for r in self.l_result:
                    #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    #     r = Region(frame_idx, x, y, w, h, conf, label,
                    #                0, origin="mpeg")
                    #     r = self.resize(r)
                    #     if self.isdelete(r):
                    #         _d_result.append(r)
                    # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]

                    for r in self.l_result:
                        label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                        #print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                #all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         # print(y)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         r = self.resize(r)
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         if self.isdelete(r):
            #             final_results.append(r)
            #     continue
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         final_results.append(r)
            #     all_time += self.times[frame_idx]
            #     continue
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    r = self.resize(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    if self.isdelete(r):
                        final_results.append(r)
#0.7868061224489794


            else:
                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            #all_time += self.times[frame_idx]
            # print('result',results)

        #final_results = merge_boxes_in_results(final_results.regions_dict, 0.5, 0.5)
        #self.show_results.combine_results(final_results)
        try:
            tp, fp, fn, _, _, _, f1 = evaluate(
                self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
                0.5, 0.5, 0.4, 0.4)
        except Exception:
            f1=0
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.04, 0.04)
        if SR_list!=s_frames:
            self.t_reward.append(f1)
        # final_results = Results()
        # # 取s_frames中帧做检测的f1score
        # #print('f1:',f1)
        # # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        self.f1_list.append(f1)
        self.last_compute_time=time
        time_trans=self.qp_bitrate[self.last_qp]/self.bandwith[int(self.idx / 30-1)+N_CHUNK-1]
        time+=time_trans
        # reward = self.a1 * (f1)  - self.a2 * (1 if time>1 else 0)
        reward = self.a1 * (f1) - self.a2 * (1 if time > 1 else 0)
        tuple=(time_trans,self.last_compute_time,time,self.bandwith[int(self.idx / 30-1)+N_CHUNK-1],self.qp_bitrate[self.last_qp],self.last_qp,f1,reward)
        self.reward_list.append(tuple)
        self.reward_sum += reward
        # 排队时间
        # self.queue =max(0,(self.queue+time-1))
        ###
        # 计算前一个gop剩余feature差值
        # new_qp = qp_thres[int(action % 5)]
        min_fabs=100
        for qp in qp_thres:
            tmp=math.fabs(qp-thre_qp)
            if tmp<min_fabs:
                min_fabs=tmp
                new_qp=qp
        self.diff_gop=self.diff_gop_dict[new_qp]
        # state_要拷贝新的chunk选的qp
        import copy
        state_ = copy.deepcopy(self.states_dict[new_qp][int(self.idx / self.goplen)])
        # print("state_.shape", np.array(state_).shape)
        # res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        # sun  31  60
        res_ = copy.deepcopy(self.res[self.idx+ 1:self.idx+self.goplen])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('余下的',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            # res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
            # [20 31)
            res_.insert(0, sum(self.res[self.last_sr_frame + 1:self.idx + 1]))

        else:
            res_.insert(0,0)
            state_.insert(0, 0)
        # print("state_.shape",np.array(state_).shape)
        ###
        self.state = np.array(state_)
        # idx变成60
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == 2970:
            done = True
        else:
            done = False
        #print(len(res_), self.idx)

        # 下一个chunk

        self.features=self.features_dict[new_qp]
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        a=np.array(self.chunk_sizes)
        a=np.roll(a,1)
        a[0]=self.qp_bitrate[self.last_qp]
        self.chunk_sizes=a.tolist()
        a=np.array(self.chunk_times)
        a=np.roll(a,1)
        a[0]=self.chunk_sizes[0]/self.bandwith[int(self.idx/30-2)+N_CHUNK-1]
        # print(time_trans,a[0],"is equal")  相等
        self.chunk_times=a.tolist()
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)
        state_.append(self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(time_trans)
        state_.append(self.last_compute_time)
        print("action", action, '\n', 'idx', self.idx, '\n', 'goplen', self.goplen)
        print("s:",thre_infer,"sr:",thre_sr,"q_a:",thre_qp,"last_qp:",self.last_qp," new_qp:",new_qp)
        print('s_frames', s_frames)
        print('SR',SR_list)
        print("last_qp bitrate bandwidth",self.last_qp,self.qp_bitrate[self.last_qp] ,self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        print('time of trans,comp,total ',time_trans,self.last_compute_time,time)
        print("f1:",f1," reward:",reward)
        # print('chunk_sizes',self.chunk_sizes)
        # print('chunk_times',self.chunk_times)
        # if self.idx>=2940:
        #     print("qp list:",self.qp_trace)
        # 这里的qp决定的是下一个chunk的qp，但是这一个chunk已经编码好了（并且取出了编码器信息），所以不太好修改
        # 根据action更新qp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.last_qp=new_qp
        self.qp_trace.append(self.last_qp)
        return np.array(state_), reward, done, None

    def evaluate_(self,model_name):
        # gt 和 待检测帧号不匹配没关系，dds evaluate函数会从0~maxfid搜索帧号是否存在于待检测文件中
        print(self.idx)
        print(self.reward_list)
        print("f1_mean", mean(self.f1_list))
        print(" total reward", self.reward_sum)
        # print('f1_all', f1, " total reward", self.reward_sum)
        band_ = []
        qp_ = []
        # tuple = (time_trans, self.last_compute_time, time, self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1],
        #          self.qp_bitrate[self.last_qp], self.last_qp, f1, reward)
        reward_=[]
        for _, _, _, band, _,qp,_, reward in self.reward_list:
            band_.append(band)
            qp_.append(self.qp_bitrate[qp])
            reward_.append(reward)

        # band_=Normalize(band_)
        # qp_=Normalize(qp_)

        print(band_)
        print(qp_)
        x = np.arange(len(qp_))
        plt.figure(1)
        plt.plot(x, band_, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='Bandwidth')
        plt.plot(x, qp_, color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Adaptive qp")
        plt.ylabel('Bandwidth (Mbps)')
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"band_qp_{model_name}.png")
        plt.show()

        plt.figure(2)
        plt.plot(x, self.f1_list, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='F1-score')
        plt.plot(x, reward_, color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Reward")
        plt.xlabel('Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"f1_reward_{model_name}.png")
        plt.show()


class Envs7:

    def __init__(self, height, width, length, states_dict, diff_gop_dict, times, result,h_result,res, features_dict,chunksizes,bandwith):
        # self.args=create_args()
        self.last_qp=30
        with open('./I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
        #print(self.I_frame)
        self.environment_title='video_V0'
        #self.action_space=spaces.Discrete(75)
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )
        high=np.zeros(128+60,dtype=np.float32)
        high=np.array([np.finfo(np.float32).max for _ in high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states_dict=states_dict
        self.states = states_dict[self.last_qp]
        self.diff_gop_dict = diff_gop_dict
        self.diff_gop = diff_gop_dict[self.last_qp]

        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = None
        self._max_episode_steps=100
        self.t_reward=[]
        #self.model = Detector()
        # with open("dds_results.txt", "rb") as get_myprofile:
        #     h_result = pickle.load(get_myprofile)
        # self.ground_truth_dict = read_results_dict(ground_path)
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     1000, self.ground_truth_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.4, 0.4)
        # print(f1)
        #self.ground_truth_dict =h_result.regions_dict
        self.times = times
        self.result_dict = result
        self.show_results = Results()
        self.h_result_dict=h_result
        # gt是质量最好的540p  每帧SR之后 infer的结果
        self.ground_truth_dict=h_result[10]

        self.res=res
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features_dict = features_dict
        self.features=features_dict[self.last_qp]


        with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/models/pca.pkl","rb") as get_myprofile:
            self.pca=pickle.load(get_myprofile)
        # pca = PCA(n_components=128)  # 实例化
        # self.pca = pca.fit(np.array(self.features))  # 拟合模型
        self.srl=[]
        self.dtl=[]
        self.s_a=0
        self.d_a=0
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)
        self.qp_bitrate = chunksizes
        self.chunk_times=np.zeros((N_CHUNK))
        self.chunk_sizes=np.zeros((N_CHUNK))
        # self.chunk_sizes=[ 1.6027964834173045, 1.6027964834173045,
        # 1.6027964834173045,
        # 1.6027964834173045,
        # 1.5982735510189188,
        # 1.5982735510189188,
        #
        # 1.5982735510189188,
        # 1.5982735510189188
        # ]
        #
        # self.chunk_times=[self.qp_bitrate[self.last_qp]/i for i in self.chunk_sizes]

        self.bandwith=bandwith
        self.qp_trace=[]
        self.last_compute_time=0
        self.f1_list=[]
        self.reward_list=[]
        self.reward_sum = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.idx = 30
        self.queue = 0
        # self.states[0] 取一个gop的state  29张（不算第一帧）
        import copy
        state_ = copy.deepcopy(self.states[0])
        res_ = copy.deepcopy(self.res[1:30])
        self.last_sr_frame=0
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        res_.insert(0,0)
        #print(len(res_),self.idx)
        #
        self.state = np.array(state_)
        # pca输入是两维
        random_integer = random.randint(0, 100)
        self.last_qp=qp_thres[random_integer % N_qp]
        print("reset random qp", self.last_qp)
        self.features=self.features_dict[self.last_qp]
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        # print(len(self.states[0]))
        # print(len(state_))

        # self.chunk_sizes[0]=self.qp_bitrate[self.last_qp]
        # # 这个后面需要改时间和size，现在只是简单模拟！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # self.chunk_times[0]=self.chunk_sizes[0]/self.bandwith[0]
        for i in range(N_CHUNK):
            self.chunk_sizes[i]=self.bandwith[N_CHUNK-1-i]

        # self.chunk_sizes = [1.6027964834173045, 1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.5982735510189188,
        #                     1.5982735510189188,
        #
        #                     1.5982735510189188,
        #                     1.5982735510189188
        #                     ]

        self.chunk_times = [self.qp_bitrate[self.last_qp] / i for i in self.chunk_sizes]

        self.qp_trace.append(self.last_qp)
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)

        state_.append(self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(self.qp_bitrate[self.last_qp]/self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.last_compute_time)
        # state_ list:205
        # print(state_)
        return np.array(state_)
    # def delete(self):
    #     for r in self.l_result:
    #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist
    def move(self,frame_idx):
        if frame_idx + 1 in self.I_frame:
            for r in self.l_result:
                label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                #final_results.append(r)
        else:
            _d_result = Results()
            refer = find_refer(frame_idx + 1) - 1
            if refer + 1 == frame_idx:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    _x, _y = _x / 4, _y / 4
                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                    if _x != 0 and _y != 0:
                        _x = _x - __x
                        _y = _y - __y
                    _x, _y = _x / 4, _y / 4

                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            else:

                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                    _x, _y = _x / 4, _y / 4

                    # _x, _y =0,0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            self.l_result = _d_result.regions_dict[frame_idx]

    def resize(self,r):
        if (r.x+r.w)>1:
            r.w=1-r.x
        if (r.y + r.h) > 1:
            r.w = 1 - r.y
        r.x=max(0,r.x)
        r.y=max(0,r.y)
        r.h=max(0,r.h)
        r.w=max(0,r.w)
        r.x = min(1, r.x)
        r.y = min(1, r.y)
        return r

    def isexist(self,region,result):
        for r in result:
            if calc_iou(r,region)>0.1:
                return True
        return False
    def isdelete(self,r):

        if r.w==0 or r.h==0:
            return False
        return True

    def clear_sr(self,frame_idx):
        res=[]
        for region in self.l_result:
            flag=False
            for h_region in self.result[frame_idx]:
                if filter_bbox_group(region, h_region, 0.1) or h_region.w*h_region.h>0.005:
                    flag=True
                    break
            if not flag:
                res.append(region)
        self.l_result=res+self.result[frame_idx]
    def step(self, action):
        # 能够区分内容简单的f1比价高 0.8+
        # self.last_qp=45

        #
        # self.last_qp = 20
        # 先通过a映射出edge difference阈值选出select_frames，再通过映射出的res阈值再selected_frames中选出SR list
        # self.idx  30 60 90...(下一个gop的第一帧帧号，用来索引本gop的末尾帧号,self.goplen 恒定30
        s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(int(action/(N_rthres*N_qp)))], self.state, (self.idx - self.goplen),
                                                    self.goplen)
        #self.select.append(s_frames)r_thres[int(action/15)]
        SR_list=self.get_SRlist(s_frames,r_thres[int(action%(N_rthres*N_qp)/N_qp)])
        self.result=self.result_dict[self.last_qp]
        # 对应的qp  SR之后infer的结果
        self.h_result=self.h_result_dict[self.last_qp]
        # self.states=self.states_dict[self.last_qp]
        # self.features=self.features_dict[self.last_qp]
        # self.diff_gop=self.diff_gop_dict[self.last_qp]
        #SR_list=[]
        #s_frames=SR_list
        # s_frames = self.d_pro.baQtch_diff_noobj_last((action%0.1)*10, self.state, (self.idx - self.goplen),self.goplen)
        # s_frames = self.d_pro.batch_diff_noobj_last(action[0], self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        # SR_list=self.get_SRlist(s_frames,r_thres[int(action[1])])
        # SR_list = self.get_SRlist(s_frames,(action-action%0.1)*3)
        #print(action,(action%0.1)*10,(action-action%0.1)*3)
        self.srl+=SR_list
        self.dtl+=s_frames
        # s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres[self.d_a]), self.state, (self.idx - self.goplen),                                             self.goplen)
        # SR_list = self.get_SRlist(s_frames, r_thres[self.s_a])
        # s_frames = [0, 16, 29, 39, 50, 58, 67, 78, 87, 101, 115, 116, 135, 145, 162, 174, 203, 232, 261, 290, 294, 297, 302, 307,
        #            310, 318, 319, 330, 345, 348, 351, 354, 356, 365, 377, 382, 386, 394, 406, 412, 428, 435, 441, 446, 450, 454,
        #            457, 462, 464, 480, 493, 497, 500, 505, 511, 518, 522, 527, 532, 540, 549, 551, 567, 580, 588, 600, 604, 608,
        #            609, 629, 638, 661, 667, 671, 674, 679, 684, 689, 694, 696, 713, 725, 746, 754, 766, 780, 783, 796, 809, 812,
        #            821, 831, 841, 854, 869, 870, 882, 895, 899, 901, 903, 905, 907, 909, 912, 915, 918, 921, 925, 928, 932, 936,
        #            941, 948, 956, 957, 977, 986, 1003, 1015, 1044, 1052, 1058, 1065, 1072, 1073, 1101, 1102, 1111, 1122, 1131,
        #            1147, 1160, 1177, 1189, 1214, 1218, 1246, 1247, 1276, 1304, 1305, 1333, 1334, 1352, 1363, 1385, 1392, 1410,
        #            1421, 1440, 1450, 1468, 1479, 1496, 1508, 1526, 1537, 1557, 1566, 1587, 1595, 1614, 1624, 1644, 1653, 1674,
        #            1682, 1697, 1711, 1720, 1731, 1740, 1751, 1762, 1769, 1787, 1798, 1815, 1827, 1843, 1856, 1871, 1885, 1900,
        #            1914, 1929, 1943, 1958, 1972, 1988, 2001, 2017, 2030, 2046, 2059, 2075, 2088, 2103, 2117, 2118, 2120, 2122,
        #            2123, 2124, 2125, 2126, 2127, 2129, 2130, 2132, 2134, 2136, 2138, 2140, 2142, 2144, 2146, 2164, 2175, 2193,
        #            2204, 2214, 2225, 2233, 2252, 2262, 2268, 2274, 2280, 2284, 2288, 2291, 2306, 2320, 2333, 2347, 2349, 2361,
        #            2376, 2378, 2390, 2403, 2407, 2422, 2436, 2451, 2465, 2475, 2490, 2494, 2511, 2523, 2538, 2552, 2568, 2581,
        #            2597, 2610, 2627, 2639, 2655, 2668, 2681, 2696, 2697, 2708, 2726, 2745, 2755, 2770, 2784, 2803, 2813, 2830,
        # #            2842, 2845, 2848, 2850, 2854, 2859, 2862, 2867, 2871, 2893, 2900, 2922, 2929, 2946]
        # s_frames=[0, 29, 41, 55, 58, 63, 68, 73, 77, 82, 87, 96, 100, 105, 113, 116, 145, 174, 203, 232, 261, 290, 319, 348, 358, 376, 377, 406, 435, 444, 450, 458, 464, 474, 483, 493, 505, 517, 522, 531, 543, 551, 560, 570, 580, 590, 600, 609, 619, 629, 638, 648, 658, 667, 678, 689, 695, 696, 701, 710, 718, 725, 732, 741, 749, 750, 754, 762, 769, 776, 783, 807, 812, 823, 836, 841, 852, 866, 870, 877, 887, 896, 899, 902, 911, 919, 927, 928, 937, 947, 957, 968, 979, 986, 995, 1005, 1015, 1029, 1041, 1044, 1055, 1063, 1070, 1073, 1093, 1102, 1119, 1131, 1147, 1160, 1183, 1189, 1210, 1218, 1239, 1247, 1270, 1276, 1298, 1305, 1327, 1334, 1353, 1363, 1392, 1416, 1421, 1449, 1450, 1479, 1505, 1508, 1537, 1566, 1595, 1624, 1653, 1680, 1682, 1705, 1711, 1740, 1765, 1769, 1782, 1794, 1798, 1816, 1827, 1851, 1856, 1881, 1885, 1906, 1914, 1927, 1940, 1943, 1963, 1972, 1990, 2001, 2022, 2030, 2051, 2059, 2069, 2077, 2085, 2088, 2111, 2117, 2123, 2129, 2135, 2141, 2146, 2158, 2167, 2175, 2187, 2197, 2204, 2211, 2218, 2225, 2232, 2233, 2248, 2262, 2272, 2281, 2291, 2298, 2305, 2312, 2319, 2320, 2329, 2338, 2348, 2349, 2357, 2366, 2374, 2378, 2388, 2396, 2407, 2419, 2430, 2436, 2447, 2459, 2465, 2471, 2476, 2481, 2486, 2490, 2494, 2510, 2521, 2523, 2533, 2548, 2552, 2564, 2577, 2581, 2587, 2597, 2606, 2610, 2620, 2627, 2635, 2639, 2649, 2658, 2663, 2667, 2668, 2674, 2681, 2689, 2697, 2713, 2726, 2734, 2741, 2749, 2754, 2755, 2784, 2807, 2813, 2842, 2850, 2859, 2866, 2871, 2877, 2884, 2891, 2897, 2900, 2929, 2934, 2939, 2945, 2950, 2956, 2958, 2986]
        #
        # SR_list =[]
        # s=range(180)
        # s_frames=np.array(s).tolist()
        #s_frames=[0,30,60,90,120,150]
        #SR_list=s_frames
        # print("s_a",int(action/15))

        final_results = Results()
        # 取s_frames中帧做检测的f1score
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        #all_time = 0
        # print(self.idx-self.goplen)
        # [0，30） [30,60)
        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                if frame_idx in SR_list:

                    self.l_result = self.h_result[frame_idx]
                    #self.l_result = self.result[frame_idx]
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                    # sr   1080 infer
                    time += 0.1724+0.0924+0.02
                #     0.2848
                else:
                    #for r in self.self.result[frame_idx]:
                    _d_result = Results()
                    #self.move(frame_idx)
                    # self.clear_sr(frame_idx)
                    # 就用lr infer 代替transfer+infer
                    self.l_result = self.result[frame_idx]

                    time += 0.0924+0.02
                    # 0.1124
                    # for r in self.l_result:
                    #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    #     r = Region(frame_idx, x, y, w, h, conf, label,
                    #                0, origin="mpeg")
                    #     r = self.resize(r)
                    #     if self.isdelete(r):
                    #         _d_result.append(r)
                    # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]

                    for r in self.l_result:
                        label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                        #print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                #all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         # print(y)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         r = self.resize(r)
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         if self.isdelete(r):
            #             final_results.append(r)
            #     continue
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         final_results.append(r)
            #     all_time += self.times[frame_idx]
            #     continue
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    r = self.resize(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    if self.isdelete(r):
                        final_results.append(r)
#0.7868061224489794


            else:
                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            #all_time += self.times[frame_idx]
            # print('result',results)

        #final_results = merge_boxes_in_results(final_results.regions_dict, 0.5, 0.5)
        #self.show_results.combine_results(final_results)

        try:
            tp, fp, fn, _, _, _, f1 = evaluate(
                self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
                0.5, 0.5, 0.4, 0.4)
        except Exception:
            f1=0
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.04, 0.04)
        if SR_list!=s_frames:
            self.t_reward.append(f1)
        # final_results = Results()
        # # 取s_frames中帧做检测的f1score
        # #print('f1:',f1)
        # # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        self.f1_list.append(f1)
        self.last_compute_time=time
        time_trans=self.qp_bitrate[self.last_qp]/self.bandwith[int(self.idx / 30-1)+N_CHUNK-1]
        time+=time_trans
        # reward = self.a1 * (f1)  - self.a2 * (1 if time>1 else 0)
        reward = self.a1 * (f1) - self.a2 * (1 if time > 1 else 0)
        tuple=(time_trans,self.last_compute_time,time,self.bandwith[int(self.idx / 30-1)+N_CHUNK-1],self.qp_bitrate[self.last_qp],self.last_qp,f1,reward)
        self.reward_list.append(tuple)
        self.reward_sum += reward
        # 排队时间
        # self.queue =max(0,(self.queue+time-1))
        ###
        # 计算前一个gop剩余feature差值
        new_qp = qp_thres[int(action % N_qp)]
        self.diff_gop=self.diff_gop_dict[new_qp]
        # state_要拷贝新的chunk选的qp
        import copy
        state_ = copy.deepcopy(self.states_dict[new_qp][int(self.idx / self.goplen)])
        # print("state_.shape", np.array(state_).shape)
        # res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        # sun  31  60
        res_ = copy.deepcopy(self.res[self.idx+ 1:self.idx+self.goplen])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('余下的',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            # res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
            # [20 31)
            res_.insert(0, sum(self.res[self.last_sr_frame + 1:self.idx + 1]))

        else:
            res_.insert(0,0)
            state_.insert(0, 0)
        # print("state_.shape",np.array(state_).shape)
        ###
        self.state = np.array(state_)
        # idx变成60
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == 2970:
            done = True
        else:
            done = False
        #print(len(res_), self.idx)

        # 下一个chunk

        self.features=self.features_dict[new_qp]
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        a=np.array(self.chunk_sizes)
        a=np.roll(a,1)
        a[0]=self.qp_bitrate[self.last_qp]
        self.chunk_sizes=a.tolist()
        a=np.array(self.chunk_times)
        a=np.roll(a,1)
        a[0]=self.chunk_sizes[0]/self.bandwith[int(self.idx/30-2)+N_CHUNK-1]
        # print(time_trans,a[0],"is equal")  相等
        self.chunk_times=a.tolist()
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)
        state_.append(self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(time_trans)
        state_.append(self.last_compute_time)
        print("action", action, '\n', 'idx', self.idx, '\n', 'goplen', self.goplen)
        print("s sr q_a last_qp qp", int(action / (N_rthres*N_qp)), int(action %  (N_rthres*N_qp) / N_qp), action % N_qp, self.last_qp,
              qp_thres[action % N_qp])
        print('s_frames', s_frames)
        print('SR',SR_list)
        print("last_qp bitrate bandwidth",self.last_qp,self.qp_bitrate[self.last_qp] ,self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        print('time ',time_trans,self.last_compute_time,time)
        print('f1',f1)
        # print('chunk_sizes',self.chunk_sizes)
        # print('chunk_times',self.chunk_times)
        # if self.idx>=2940:
        #     print("qp list:",self.qp_trace)
        # 这里的qp决定的是下一个chunk的qp，但是这一个chunk已经编码好了（并且取出了编码器信息），所以不太好修改
        # 根据action更新qp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.last_qp=new_qp
        self.qp_trace.append(self.last_qp)
        return np.array(state_), reward, done, None

    def evaluate_(self,model_name):
        #              max_tuple={"thres": thres1,
        #                                    "rthres": thres2,
        #                                    "qp": qp,
        #                                    "bandwidth": self.bandwith[int(self.idx / 30 - 1)],
        #                                    "trans_time": time_trans,
        #                                    "comp_time": self.last_compute_time,
        #                                    "total_time": time,
        #                                    "f1": f1,
        #                                    "reward": reward,
        #                                    "sr_list":SR_list,
        #                                    "infer_list": s_frames
        #                                    }

        with open("search.txt", "rb") as myfile:
            search_reward = pickle.load(myfile)
        sreward_=[]
        for dic in search_reward:
            sreward_.append(dic["reward"])

        # gt 和 待检测帧号不匹配没关系，dds evaluate函数会从0~maxfid搜索帧号是否存在于待检测文件中
        print(self.idx)
        print(self.reward_list)
        print("f1_mean", mean(self.f1_list))
        print(" total reward", self.reward_sum)
        # print('f1_all', f1, " total reward", self.reward_sum)
        band_ = []
        qp_ = []
        # tuple = (time_trans, self.last_compute_time, time, self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1],
        #          self.qp_bitrate[self.last_qp], self.last_qp, f1, reward)
        reward_=[]
        for _, _, _, band, _,qp,_, reward in self.reward_list:
            band_.append(band)
            qp_.append(self.qp_bitrate[qp])
            reward_.append(reward)

        with open(f"reward_{model_name}_20.txt","wb") as myfile:
            pickle.dump(self.reward_list,myfile)

        # band_=Normalize(band_)
        # qp_=Normalize(qp_)

        print(band_)
        print(qp_)
        x = np.arange(len(qp_))
        plt.figure(1)
        plt.plot(x, band_, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='Bandwidth')
        plt.plot(x, qp_, color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Adaptive bitrate")
        plt.ylabel('Bandwidth (Mbps)')
        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"band_qp_{model_name}.png")
        plt.show()

        plt.figure(2)
        x = np.arange(len(reward_) - 1)
        plt.plot(x, self.f1_list[1:], color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='F1-score')
        plt.plot(x, reward_[1:], color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Reward")

        for i in range(0,100,10):

            plt.plot([i,i], [0,1], 'k--', linewidth=1)

        for i,acc in enumerate(self.f1_list):
            if acc>0.8:
                print(i,acc)

        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"f1_reward_{model_name}.png")
        plt.show()

        plt.figure(3)
        x=np.arange(len(reward_)-1)
        sum=0
        sum_p=0
        print(len(reward_))
        for i in range(1,len(reward_)):
            dif=math.fabs(sreward_[i]-reward_[i])
            sum+=dif
            sum_p+=dif*dif
        print("reward 误差累计  平均误差 均方误差：",sum,sum/(len(reward_)-1),sum_p/(len(reward_)-1))
        plt.plot(x, sreward_[1:], color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='Opt reward')
        plt.plot(x,reward_[1:], color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="DRL reward")
        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"different_reward_{model_name}.png")
        plt.show()



class Envs8:

    def __init__(self, height, width, length, states_dict, diff_gop_dict, times, result,h_result,res, features_dict,chunksizes,bandwith):
        # self.args=create_args()
        self.last_qp=30
        with open('./I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
        #print(self.I_frame)
        self.environment_title='video_V0'
        #self.action_space=spaces.Discrete(75)
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )
        high=np.zeros(128+60,dtype=np.float32)
        high=np.array([np.finfo(np.float32).max for _ in high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states_dict=states_dict
        self.states = states_dict[self.last_qp]
        self.diff_gop_dict = diff_gop_dict
        self.diff_gop = diff_gop_dict[self.last_qp]

        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = None
        self._max_episode_steps=100
        self.t_reward=[]
        #self.model = Detector()
        # with open("dds_results.txt", "rb") as get_myprofile:
        #     h_result = pickle.load(get_myprofile)
        # self.ground_truth_dict = read_results_dict(ground_path)
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     1000, self.ground_truth_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.4, 0.4)
        # print(f1)
        #self.ground_truth_dict =h_result.regions_dict
        self.times = times
        self.result_dict = result
        self.show_results = Results()
        self.h_result_dict=h_result
        # gt是质量最好的540p  每帧SR之后 infer的结果
        self.ground_truth_dict=h_result[10]

        self.res=res
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features_dict = features_dict
        self.features=features_dict[self.last_qp]


        with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/models/pca.pkl","rb") as get_myprofile:
            self.pca=pickle.load(get_myprofile)
        # pca = PCA(n_components=128)  # 实例化
        # self.pca = pca.fit(np.array(self.features))  # 拟合模型
        self.srl=[]
        self.dtl=[]
        self.s_a=0
        self.d_a=0
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)
        self.qp_bitrate = chunksizes
        self.chunk_times=np.zeros((N_CHUNK))
        self.chunk_sizes=np.zeros((N_CHUNK))
        # self.chunk_sizes=[ 1.6027964834173045, 1.6027964834173045,
        # 1.6027964834173045,
        # 1.6027964834173045,
        # 1.5982735510189188,
        # 1.5982735510189188,
        #
        # 1.5982735510189188,
        # 1.5982735510189188
        # ]
        #
        # self.chunk_times=[self.qp_bitrate[self.last_qp]/i for i in self.chunk_sizes]

        self.bandwith=bandwith
        self.qp_trace=[]
        self.last_compute_time=0
        self.f1_list=[]
        self.reward_list=[]
        self.reward_sum = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.idx = 30
        self.queue = 0
        # self.states[0] 取一个gop的state  29张（不算第一帧）
        import copy
        state_ = copy.deepcopy(self.states[0])
        res_ = copy.deepcopy(self.res[1:30])
        self.last_sr_frame=0
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        res_.insert(0,0)
        #print(len(res_),self.idx)
        #
        self.state = np.array(state_)
        # pca输入是两维
        random_integer = random.randint(0, 100)
        self.last_qp=qp_thres[random_integer % N_qp]
        print("reset random qp", self.last_qp)
        self.features=self.features_dict[self.last_qp]
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        # print(len(self.states[0]))
        # print(len(state_))

        # self.chunk_sizes[0]=self.qp_bitrate[self.last_qp]
        # # 这个后面需要改时间和size，现在只是简单模拟！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # self.chunk_times[0]=self.chunk_sizes[0]/self.bandwith[0]
        for i in range(N_CHUNK):
            self.chunk_sizes[i]=self.bandwith[N_CHUNK-1-i]

        # self.chunk_sizes = [1.6027964834173045, 1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.5982735510189188,
        #                     1.5982735510189188,
        #
        #                     1.5982735510189188,
        #                     1.5982735510189188
        #                     ]

        self.chunk_times = [self.qp_bitrate[self.last_qp] / i for i in self.chunk_sizes]

        self.qp_trace.append(self.last_qp)
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)

        state_.append(self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(self.qp_bitrate[self.last_qp]/self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.last_compute_time)
        # state_ list:205
        # print(state_)
        return np.array(state_)
    # def delete(self):
    #     for r in self.l_result:
    #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist
    def move(self,frame_idx):
        if frame_idx + 1 in self.I_frame:
            for r in self.l_result:
                label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                #final_results.append(r)
        else:
            _d_result = Results()
            refer = find_refer(frame_idx + 1) - 1
            if refer + 1 == frame_idx:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    _x, _y = _x / 4, _y / 4
                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                    if _x != 0 and _y != 0:
                        _x = _x - __x
                        _y = _y - __y
                    _x, _y = _x / 4, _y / 4

                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            else:

                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                    _x, _y = _x / 4, _y / 4

                    # _x, _y =0,0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            self.l_result = _d_result.regions_dict[frame_idx]

    def resize(self,r):
        if (r.x+r.w)>1:
            r.w=1-r.x
        if (r.y + r.h) > 1:
            r.w = 1 - r.y
        r.x=max(0,r.x)
        r.y=max(0,r.y)
        r.h=max(0,r.h)
        r.w=max(0,r.w)
        r.x = min(1, r.x)
        r.y = min(1, r.y)
        return r

    def isexist(self,region,result):
        for r in result:
            if calc_iou(r,region)>0.1:
                return True
        return False
    def isdelete(self,r):

        if r.w==0 or r.h==0:
            return False
        return True

    def clear_sr(self,frame_idx):
        res=[]
        for region in self.l_result:
            flag=False
            for h_region in self.result[frame_idx]:
                if filter_bbox_group(region, h_region, 0.1) or h_region.w*h_region.h>0.005:
                    flag=True
                    break
            if not flag:
                res.append(region)
        self.l_result=res+self.result[frame_idx]

    def step(self, action):
        # 先通过a映射出edge difference阈值选出select_frames，再通过映射出的res阈值再selected_frames中选出SR list
        # self.idx  30 60 90...(下一个gop的第一帧帧号，用来索引本gop的末尾帧号,self.goplen 恒定30
        #
        thre_infer =thres[action[0]%N_thres]
        thre_sr = r_thres[action[1]%N_rthres]
        thre_qp = qp_thres[action[2]%N_qp]
        s_frames = self.d_pro.batch_diff_noobj_last(thre_infer, self.state,
                                                    (self.idx - self.goplen),
                                                    self.goplen)
        # self.select.append(s_frames)r_thres[int(action/15)]
        SR_list = self.get_SRlist(s_frames, thre_sr)

        #
        # s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(int(action/25))], self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        # #self.select.append(s_frames)r_thres[int(action/15)]
        # SR_list=self.get_SRlist(s_frames,r_thres[int(action%25/5)])
        self.result = self.result_dict[self.last_qp]
        # 对应的qp  SR之后infer的结果
        self.h_result = self.h_result_dict[self.last_qp]
        # self.states=self.states_dict[self.last_qp]
        # self.features=self.features_dict[self.last_qp]
        # self.diff_gop=self.diff_gop_dict[self.last_qp]
        # SR_list=[]
        # s_frames=SR_list
        # s_frames = self.d_pro.baQtch_diff_noobj_last((action%0.1)*10, self.state, (self.idx - self.goplen),self.goplen)
        # s_frames = self.d_pro.batch_diff_noobj_last(action[0], self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        # SR_list=self.get_SRlist(s_frames,r_thres[int(action[1])])
        # SR_list = self.get_SRlist(s_frames,(action-action%0.1)*3)
        # print(action,(action%0.1)*10,(action-action%0.1)*3)
        self.srl += SR_list
        self.dtl += s_frames
        # s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres[self.d_a]), self.state, (self.idx - self.goplen),                                             self.goplen)
        # SR_list = self.get_SRlist(s_frames, r_thres[self.s_a])
        # s_frames = [0, 16, 29, 39, 50, 58, 67, 78, 87, 101, 115, 116, 135, 145, 162, 174, 203, 232, 261, 290, 294, 297, 302, 307,
        #            310, 318, 319, 330, 345, 348, 351, 354, 356, 365, 377, 382, 386, 394, 406, 412, 428, 435, 441, 446, 450, 454,
        #            457, 462, 464, 480, 493, 497, 500, 505, 511, 518, 522, 527, 532, 540, 549, 551, 567, 580, 588, 600, 604, 608,
        #            609, 629, 638, 661, 667, 671, 674, 679, 684, 689, 694, 696, 713, 725, 746, 754, 766, 780, 783, 796, 809, 812,
        #            821, 831, 841, 854, 869, 870, 882, 895, 899, 901, 903, 905, 907, 909, 912, 915, 918, 921, 925, 928, 932, 936,
        #            941, 948, 956, 957, 977, 986, 1003, 1015, 1044, 1052, 1058, 1065, 1072, 1073, 1101, 1102, 1111, 1122, 1131,
        #            1147, 1160, 1177, 1189, 1214, 1218, 1246, 1247, 1276, 1304, 1305, 1333, 1334, 1352, 1363, 1385, 1392, 1410,
        #            1421, 1440, 1450, 1468, 1479, 1496, 1508, 1526, 1537, 1557, 1566, 1587, 1595, 1614, 1624, 1644, 1653, 1674,
        #            1682, 1697, 1711, 1720, 1731, 1740, 1751, 1762, 1769, 1787, 1798, 1815, 1827, 1843, 1856, 1871, 1885, 1900,
        #            1914, 1929, 1943, 1958, 1972, 1988, 2001, 2017, 2030, 2046, 2059, 2075, 2088, 2103, 2117, 2118, 2120, 2122,
        #            2123, 2124, 2125, 2126, 2127, 2129, 2130, 2132, 2134, 2136, 2138, 2140, 2142, 2144, 2146, 2164, 2175, 2193,
        #            2204, 2214, 2225, 2233, 2252, 2262, 2268, 2274, 2280, 2284, 2288, 2291, 2306, 2320, 2333, 2347, 2349, 2361,
        #            2376, 2378, 2390, 2403, 2407, 2422, 2436, 2451, 2465, 2475, 2490, 2494, 2511, 2523, 2538, 2552, 2568, 2581,
        #            2597, 2610, 2627, 2639, 2655, 2668, 2681, 2696, 2697, 2708, 2726, 2745, 2755, 2770, 2784, 2803, 2813, 2830,
        # #            2842, 2845, 2848, 2850, 2854, 2859, 2862, 2867, 2871, 2893, 2900, 2922, 2929, 2946]
        # s_frames=[0, 29, 41, 55, 58, 63, 68, 73, 77, 82, 87, 96, 100, 105, 113, 116, 145, 174, 203, 232, 261, 290, 319, 348, 358, 376, 377, 406, 435, 444, 450, 458, 464, 474, 483, 493, 505, 517, 522, 531, 543, 551, 560, 570, 580, 590, 600, 609, 619, 629, 638, 648, 658, 667, 678, 689, 695, 696, 701, 710, 718, 725, 732, 741, 749, 750, 754, 762, 769, 776, 783, 807, 812, 823, 836, 841, 852, 866, 870, 877, 887, 896, 899, 902, 911, 919, 927, 928, 937, 947, 957, 968, 979, 986, 995, 1005, 1015, 1029, 1041, 1044, 1055, 1063, 1070, 1073, 1093, 1102, 1119, 1131, 1147, 1160, 1183, 1189, 1210, 1218, 1239, 1247, 1270, 1276, 1298, 1305, 1327, 1334, 1353, 1363, 1392, 1416, 1421, 1449, 1450, 1479, 1505, 1508, 1537, 1566, 1595, 1624, 1653, 1680, 1682, 1705, 1711, 1740, 1765, 1769, 1782, 1794, 1798, 1816, 1827, 1851, 1856, 1881, 1885, 1906, 1914, 1927, 1940, 1943, 1963, 1972, 1990, 2001, 2022, 2030, 2051, 2059, 2069, 2077, 2085, 2088, 2111, 2117, 2123, 2129, 2135, 2141, 2146, 2158, 2167, 2175, 2187, 2197, 2204, 2211, 2218, 2225, 2232, 2233, 2248, 2262, 2272, 2281, 2291, 2298, 2305, 2312, 2319, 2320, 2329, 2338, 2348, 2349, 2357, 2366, 2374, 2378, 2388, 2396, 2407, 2419, 2430, 2436, 2447, 2459, 2465, 2471, 2476, 2481, 2486, 2490, 2494, 2510, 2521, 2523, 2533, 2548, 2552, 2564, 2577, 2581, 2587, 2597, 2606, 2610, 2620, 2627, 2635, 2639, 2649, 2658, 2663, 2667, 2668, 2674, 2681, 2689, 2697, 2713, 2726, 2734, 2741, 2749, 2754, 2755, 2784, 2807, 2813, 2842, 2850, 2859, 2866, 2871, 2877, 2884, 2891, 2897, 2900, 2929, 2934, 2939, 2945, 2950, 2956, 2958, 2986]
        #
        # SR_list =[]
        # s=range(180)
        # s_frames=np.array(s).tolist()
        # s_frames=[0,30,60,90,120,150]
        # SR_list=s_frames
        # print("s_a",int(action/15))

        final_results = Results()
        # 取s_frames中帧做检测的f1score
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        # all_time = 0
        # print(self.idx-self.goplen)
        # [0，30） [30,60)
        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                if frame_idx in SR_list:

                    self.l_result = self.h_result[frame_idx]
                    # self.l_result = self.result[frame_idx]
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                    # sr   1080 infer
                    time += 0.1724 + 0.0924 + 0.02
                #     0.2848
                else:
                    # for r in self.self.result[frame_idx]:
                    _d_result = Results()
                    # self.move(frame_idx)
                    # self.clear_sr(frame_idx)
                    # 就用lr infer 代替transfer+infer
                    self.l_result = self.result[frame_idx]

                    time += 0.0924 + 0.02
                    # 0.1124
                    # for r in self.l_result:
                    #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    #     r = Region(frame_idx, x, y, w, h, conf, label,
                    #                0, origin="mpeg")
                    #     r = self.resize(r)
                    #     if self.isdelete(r):
                    #         _d_result.append(r)
                    # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                # all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         # print(y)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         r = self.resize(r)
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         if self.isdelete(r):
            #             final_results.append(r)
            #     continue
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         final_results.append(r)
            #     all_time += self.times[frame_idx]
            #     continue
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    r = self.resize(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    if self.isdelete(r):
                        final_results.append(r)
            # 0.7868061224489794

            else:
                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            # all_time += self.times[frame_idx]
            # print('result',results)

        # final_results = merge_boxes_in_results(final_results.regions_dict, 0.5, 0.5)
        # self.show_results.combine_results(final_results)
        try:
            tp, fp, fn, _, _, _, f1 = evaluate(
                self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
                0.5, 0.5, 0.4, 0.4)
        except Exception:
            f1 = 0
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.04, 0.04)
        if SR_list != s_frames:
            self.t_reward.append(f1)
        # final_results = Results()
        # # 取s_frames中帧做检测的f1score
        # #print('f1:',f1)
        # # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        self.f1_list.append(f1)
        self.last_compute_time = time
        time_trans = self.qp_bitrate[self.last_qp] / self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1]
        time += time_trans
        # reward = self.a1 * (f1)  - self.a2 * (1 if time>1 else 0)
        reward = self.a1 * (f1) - self.a2 * (1 if time > 1 else 0)
        tuple = (time_trans, self.last_compute_time, time, self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1],
                 self.qp_bitrate[self.last_qp], self.last_qp, f1, reward)
        self.reward_list.append(tuple)
        self.reward_sum += reward
        # 排队时间
        # self.queue =max(0,(self.queue+time-1))
        ###
        # 计算前一个gop剩余feature差值
        # new_qp = qp_thres[int(action % 5)]
        min_fabs = 100
        new_qp=thre_qp
        self.diff_gop = self.diff_gop_dict[new_qp]
        # state_要拷贝新的chunk选的qp
        import copy
        state_ = copy.deepcopy(self.states_dict[new_qp][int(self.idx / self.goplen)])
        # print("state_.shape", np.array(state_).shape)
        # res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        # sun  31  60
        res_ = copy.deepcopy(self.res[self.idx + 1:self.idx + self.goplen])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('余下的',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            # res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
            # [20 31)
            res_.insert(0, sum(self.res[self.last_sr_frame + 1:self.idx + 1]))

        else:
            res_.insert(0, 0)
            state_.insert(0, 0)
        # print("state_.shape",np.array(state_).shape)
        ###
        self.state = np.array(state_)
        # idx变成60
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == 2970:
            done = True
        else:
            done = False
        # print(len(res_), self.idx)

        # 下一个chunk

        self.features = self.features_dict[new_qp]
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_ += res_
        # state_.append(self.queue)
        a = np.array(self.chunk_sizes)
        a = np.roll(a, 1)
        a[0] = self.qp_bitrate[self.last_qp]
        self.chunk_sizes = a.tolist()
        a = np.array(self.chunk_times)
        a = np.roll(a, 1)
        a[0] = self.chunk_sizes[0] / self.bandwith[int(self.idx / 30 - 2) + N_CHUNK - 1]
        # print(time_trans,a[0],"is equal")  相等
        self.chunk_times = a.tolist()
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)
        state_.append(self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(time_trans)
        state_.append(self.last_compute_time)

        print("s:", thre_infer, "sr:", thre_sr, "q_a:", thre_qp, "last_qp:", self.last_qp, " new_qp:", new_qp)
        print('s_frames', s_frames)
        print('SR', SR_list)
        print("last_qp bitrate bandwidth", self.last_qp, self.qp_bitrate[self.last_qp],
              self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        print('time of trans,comp,total ', time_trans, self.last_compute_time, time)
        print("f1:", f1, " reward:", reward)
        # print('chunk_sizes',self.chunk_sizes)
        # print('chunk_times',self.chunk_times)
        # if self.idx>=2940:
        #     print("qp list:",self.qp_trace)
        # 这里的qp决定的是下一个chunk的qp，但是这一个chunk已经编码好了（并且取出了编码器信息），所以不太好修改
        # 根据action更新qp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.last_qp = new_qp
        self.qp_trace.append(self.last_qp)
        return np.array(state_), reward, done, None



    def evaluate_(self,model_name):
        #              max_tuple={"thres": thres1,
        #                                    "rthres": thres2,
        #                                    "qp": qp,
        #                                    "bandwidth": self.bandwith[int(self.idx / 30 - 1)],
        #                                    "trans_time": time_trans,
        #                                    "comp_time": self.last_compute_time,
        #                                    "total_time": time,
        #                                    "f1": f1,
        #                                    "reward": reward,
        #                                    "sr_list":SR_list,
        #                                    "infer_list": s_frames
        #                                    }

        with open("search.txt", "rb") as myfile:
            search_reward = pickle.load(myfile)
        sreward_=[]
        for dic in search_reward:
            sreward_.append(dic["reward"])

        # gt 和 待检测帧号不匹配没关系，dds evaluate函数会从0~maxfid搜索帧号是否存在于待检测文件中
        print(self.idx)
        print(self.reward_list)
        print("f1_mean", mean(self.f1_list))
        print(" total reward", self.reward_sum)
        # print('f1_all', f1, " total reward", self.reward_sum)
        band_ = []
        qp_ = []
        # tuple = (time_trans, self.last_compute_time, time, self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1],
        #          self.qp_bitrate[self.last_qp], self.last_qp, f1, reward)
        reward_=[]
        for _, _, _, band, _,qp,_, reward in self.reward_list:
            band_.append(band)
            qp_.append(self.qp_bitrate[qp])
            reward_.append(reward)

        with open(f"reward_{model_name}.txt","wb") as myfile:
            pickle.dump(reward,myfile)

        # band_=Normalize(band_)
        # qp_=Normalize(qp_)

        print(band_)
        print(qp_)
        x = np.arange(len(qp_))
        plt.figure(1)
        plt.plot(x, band_, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='Bandwidth')
        plt.plot(x, qp_, color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Adaptive bitrate")
        plt.ylabel('Bandwidth (Mbps)')
        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"band_qp_{model_name}.png")
        plt.show()

        plt.figure(2)
        plt.plot(x, self.f1_list, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='F1-score')
        plt.plot(x, reward_, color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Reward")
        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"f1_reward_{model_name}.png")
        plt.show()

        plt.figure(3)
        x=np.arange(len(reward_)-1)
        sum=0
        sum_p=0
        print(len(reward_))
        for i in range(1,len(reward_)):
            dif=math.fabs(sreward_[i]-reward_[i])
            sum+=dif
            sum_p+=dif*dif
        print("reward 误差累计  平均误差 均方误差：",sum,sum/(len(reward_)-1),sum_p/(len(reward_)-1))
        plt.plot(x, sreward_[1:], color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='Opt reward')
        plt.plot(x,reward_[1:], color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="DRL reward")
        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"different_reward_{model_name}.png")
        plt.show()





class Envs9:

    def __init__(self, height, width, length, states_dict, diff_gop_dict, times, result,h_result,res, features_dict,chunksizes,bandwith):
        # self.args=create_args()
        self.last_qp=30
        with open('./I_frame.txt', "rb") as get_myprofile:
            self.I_frame = pickle.load(get_myprofile)
        #print(self.I_frame)
        self.environment_title='video_V0'
        #self.action_space=spaces.Discrete(75)
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )
        high=np.zeros(128+60,dtype=np.float32)
        high=np.array([np.finfo(np.float32).max for _ in high])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.queue = 0
        self.height = height
        self.width = width
        self.length = length
        self.states_dict=states_dict
        self.states = states_dict[self.last_qp]
        self.diff_gop_dict = diff_gop_dict
        self.diff_gop = diff_gop_dict[self.last_qp]

        self.idx = 0
        self.goplen = 30
        self.a1 = 0.5
        self.a2 = 0.5
        self.state = []
        self.last_frame = None
        self.last_sr_frame=None
        self.diff_last = None
        self.l_result = None
        self._max_episode_steps=100
        self.t_reward=[]
        #self.model = Detector()
        # with open("dds_results.txt", "rb") as get_myprofile:
        #     h_result = pickle.load(get_myprofile)
        # self.ground_truth_dict = read_results_dict(ground_path)
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     1000, self.ground_truth_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.4, 0.4)
        # print(f1)
        #self.ground_truth_dict =h_result.regions_dict
        self.times = times
        self.result_dict = result
        self.show_results = Results()
        self.h_result_dict=h_result
        # gt是质量最好的540p  每帧SR之后 infer的结果
        self.ground_truth_dict=h_result[10]

        self.res=res
        self.d_pro = diff_processor.DiffProcessor.str2class('edge')(0)
        self.features_dict = features_dict
        self.features=features_dict[self.last_qp]


        with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/models/pca.pkl","rb") as get_myprofile:
            self.pca=pickle.load(get_myprofile)
        # pca = PCA(n_components=128)  # 实例化
        # self.pca = pca.fit(np.array(self.features))  # 拟合模型
        self.srl=[]
        self.dtl=[]
        self.s_a=0
        self.d_a=0
        # self.server=Server(self.args)
        # self.client=Client(self.args.hname, self.args, self.server)
        self.qp_bitrate = chunksizes
        self.chunk_times=np.zeros((N_CHUNK))
        self.chunk_sizes=np.zeros((N_CHUNK))
        # self.chunk_sizes=[ 1.6027964834173045, 1.6027964834173045,
        # 1.6027964834173045,
        # 1.6027964834173045,
        # 1.5982735510189188,
        # 1.5982735510189188,
        #
        # 1.5982735510189188,
        # 1.5982735510189188
        # ]
        #
        # self.chunk_times=[self.qp_bitrate[self.last_qp]/i for i in self.chunk_sizes]

        self.bandwith=bandwith
        self.qp_trace=[]
        self.last_compute_time=0
        self.f1_list=[]
        self.reward_list=[]
        self.reward_sum = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.idx = 30
        self.queue = 0
        # self.states[0] 取一个gop的state  29张（不算第一帧）
        import copy
        state_ = copy.deepcopy(self.states[0])
        res_ = copy.deepcopy(self.res[1:30])
        self.last_sr_frame=0
        #
        self.last_frame = 0
        self.diff_last = 0
        state_.insert(0, 0)
        res_.insert(0,0)
        #print(len(res_),self.idx)
        #
        self.state = np.array(state_)
        # pca输入是两维
        random_integer = random.randint(0, 100)
        self.last_qp=qp_thres[random_integer % N_qp]
        print("reset random qp", self.last_qp)
        self.features=self.features_dict[self.last_qp]
        state_ += self.pca.transform([self.features[0]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        # print(len(self.states[0]))
        # print(len(state_))

        # self.chunk_sizes[0]=self.qp_bitrate[self.last_qp]
        # # 这个后面需要改时间和size，现在只是简单模拟！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # self.chunk_times[0]=self.chunk_sizes[0]/self.bandwith[0]
        for i in range(N_CHUNK):
            self.chunk_sizes[i]=self.bandwith[N_CHUNK-1-i]

        # self.chunk_sizes = [1.6027964834173045, 1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.6027964834173045,
        #                     1.5982735510189188,
        #                     1.5982735510189188,
        #
        #                     1.5982735510189188,
        #                     1.5982735510189188
        #                     ]

        self.chunk_times = [self.qp_bitrate[self.last_qp] / i for i in self.chunk_sizes]

        self.qp_trace.append(self.last_qp)
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)

        state_.append(self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(self.qp_bitrate[self.last_qp]/self.bandwith[-1+ N_CHUNK - 1])
        state_.append(self.last_compute_time)
        # state_ list:205
        # print(state_)
        return np.array(state_)
    # def delete(self):
    #     for r in self.l_result:
    #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)

    def get_SRlist(self,s_frames,thre):
        srlist=[]
        for fra in s_frames:
            if sum(self.res[self.last_sr_frame:fra+1])>thre or fra in [0,150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
                self.last_sr_frame=fra
                srlist.append(fra)
        return srlist
    def move(self,frame_idx):
        if frame_idx + 1 in self.I_frame:
            for r in self.l_result:
                label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                r = Region(frame_idx, x, y, w, h, conf, label,
                           0, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                #final_results.append(r)
        else:
            _d_result = Results()
            refer = find_refer(frame_idx + 1) - 1
            if refer + 1 == frame_idx:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    _x, _y = _x / 4, _y / 4
                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx) != -1:
                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                    __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                    if _x != 0 and _y != 0:
                        _x = _x - __x
                        _y = _y - __y
                    _x, _y = _x / 4, _y / 4

                    # _x, _y = 0, 0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                               label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            else:

                for r in self.l_result:
                    label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                    _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                    _x, _y = _x / 4, _y / 4

                    # _x, _y =0,0

                    r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                               0, origin="mpeg")
                    _d_result.append(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    #final_results.append(r)
            self.l_result = _d_result.regions_dict[frame_idx]

    def resize(self,r):
        if (r.x+r.w)>1:
            r.w=1-r.x
        if (r.y + r.h) > 1:
            r.w = 1 - r.y
        r.x=max(0,r.x)
        r.y=max(0,r.y)
        r.h=max(0,r.h)
        r.w=max(0,r.w)
        r.x = min(1, r.x)
        r.y = min(1, r.y)
        return r

    def isexist(self,region,result):
        for r in result:
            if calc_iou(r,region)>0.1:
                return True
        return False
    def isdelete(self,r):

        if r.w==0 or r.h==0:
            return False
        return True

    def clear_sr(self,frame_idx):
        res=[]
        for region in self.l_result:
            flag=False
            for h_region in self.result[frame_idx]:
                if filter_bbox_group(region, h_region, 0.1) or h_region.w*h_region.h>0.005:
                    flag=True
                    break
            if not flag:
                res.append(region)
        self.l_result=res+self.result[frame_idx]
    def step(self, action):
        # 能够区分内容简单的f1比价高 0.8+
        # self.last_qp=45

        #
        # self.last_qp = 20
        # 先通过a映射出edge difference阈值选出select_frames，再通过映射出的res阈值再selected_frames中选出SR list
        # self.idx  30 60 90...(下一个gop的第一帧帧号，用来索引本gop的末尾帧号,self.goplen 恒定30
        s_frames = self.d_pro.batch_diff_noobj_last(thres[np.array(int(action/(N_rthres*N_qp)))], self.state, (self.idx - self.goplen),
                                                    self.goplen)
        #self.select.append(s_frames)r_thres[int(action/15)]
        SR_list=self.get_SRlist(s_frames,r_thres[int(action%(N_rthres*N_qp)/N_qp)])
        self.result=self.result_dict[self.last_qp]
        # 对应的qp  SR之后infer的结果
        self.h_result=self.h_result_dict[self.last_qp]
        # self.states=self.states_dict[self.last_qp]
        # self.features=self.features_dict[self.last_qp]
        # self.diff_gop=self.diff_gop_dict[self.last_qp]
        #SR_list=[]
        #s_frames=SR_list
        # s_frames = self.d_pro.baQtch_diff_noobj_last((action%0.1)*10, self.state, (self.idx - self.goplen),self.goplen)
        # s_frames = self.d_pro.batch_diff_noobj_last(action[0], self.state, (self.idx - self.goplen),
        #                                             self.goplen)
        # SR_list=self.get_SRlist(s_frames,r_thres[int(action[1])])
        # SR_list = self.get_SRlist(s_frames,(action-action%0.1)*3)
        #print(action,(action%0.1)*10,(action-action%0.1)*3)
        self.srl+=SR_list
        self.dtl+=s_frames
        # s_frames = self.d_pro.batch_diff_noobj_last(np.array(thres[self.d_a]), self.state, (self.idx - self.goplen),                                             self.goplen)
        # SR_list = self.get_SRlist(s_frames, r_thres[self.s_a])
        # s_frames = [0, 16, 29, 39, 50, 58, 67, 78, 87, 101, 115, 116, 135, 145, 162, 174, 203, 232, 261, 290, 294, 297, 302, 307,
        #            310, 318, 319, 330, 345, 348, 351, 354, 356, 365, 377, 382, 386, 394, 406, 412, 428, 435, 441, 446, 450, 454,
        #            457, 462, 464, 480, 493, 497, 500, 505, 511, 518, 522, 527, 532, 540, 549, 551, 567, 580, 588, 600, 604, 608,
        #            609, 629, 638, 661, 667, 671, 674, 679, 684, 689, 694, 696, 713, 725, 746, 754, 766, 780, 783, 796, 809, 812,
        #            821, 831, 841, 854, 869, 870, 882, 895, 899, 901, 903, 905, 907, 909, 912, 915, 918, 921, 925, 928, 932, 936,
        #            941, 948, 956, 957, 977, 986, 1003, 1015, 1044, 1052, 1058, 1065, 1072, 1073, 1101, 1102, 1111, 1122, 1131,
        #            1147, 1160, 1177, 1189, 1214, 1218, 1246, 1247, 1276, 1304, 1305, 1333, 1334, 1352, 1363, 1385, 1392, 1410,
        #            1421, 1440, 1450, 1468, 1479, 1496, 1508, 1526, 1537, 1557, 1566, 1587, 1595, 1614, 1624, 1644, 1653, 1674,
        #            1682, 1697, 1711, 1720, 1731, 1740, 1751, 1762, 1769, 1787, 1798, 1815, 1827, 1843, 1856, 1871, 1885, 1900,
        #            1914, 1929, 1943, 1958, 1972, 1988, 2001, 2017, 2030, 2046, 2059, 2075, 2088, 2103, 2117, 2118, 2120, 2122,
        #            2123, 2124, 2125, 2126, 2127, 2129, 2130, 2132, 2134, 2136, 2138, 2140, 2142, 2144, 2146, 2164, 2175, 2193,
        #            2204, 2214, 2225, 2233, 2252, 2262, 2268, 2274, 2280, 2284, 2288, 2291, 2306, 2320, 2333, 2347, 2349, 2361,
        #            2376, 2378, 2390, 2403, 2407, 2422, 2436, 2451, 2465, 2475, 2490, 2494, 2511, 2523, 2538, 2552, 2568, 2581,
        #            2597, 2610, 2627, 2639, 2655, 2668, 2681, 2696, 2697, 2708, 2726, 2745, 2755, 2770, 2784, 2803, 2813, 2830,
        # #            2842, 2845, 2848, 2850, 2854, 2859, 2862, 2867, 2871, 2893, 2900, 2922, 2929, 2946]
        # s_frames=[0, 29, 41, 55, 58, 63, 68, 73, 77, 82, 87, 96, 100, 105, 113, 116, 145, 174, 203, 232, 261, 290, 319, 348, 358, 376, 377, 406, 435, 444, 450, 458, 464, 474, 483, 493, 505, 517, 522, 531, 543, 551, 560, 570, 580, 590, 600, 609, 619, 629, 638, 648, 658, 667, 678, 689, 695, 696, 701, 710, 718, 725, 732, 741, 749, 750, 754, 762, 769, 776, 783, 807, 812, 823, 836, 841, 852, 866, 870, 877, 887, 896, 899, 902, 911, 919, 927, 928, 937, 947, 957, 968, 979, 986, 995, 1005, 1015, 1029, 1041, 1044, 1055, 1063, 1070, 1073, 1093, 1102, 1119, 1131, 1147, 1160, 1183, 1189, 1210, 1218, 1239, 1247, 1270, 1276, 1298, 1305, 1327, 1334, 1353, 1363, 1392, 1416, 1421, 1449, 1450, 1479, 1505, 1508, 1537, 1566, 1595, 1624, 1653, 1680, 1682, 1705, 1711, 1740, 1765, 1769, 1782, 1794, 1798, 1816, 1827, 1851, 1856, 1881, 1885, 1906, 1914, 1927, 1940, 1943, 1963, 1972, 1990, 2001, 2022, 2030, 2051, 2059, 2069, 2077, 2085, 2088, 2111, 2117, 2123, 2129, 2135, 2141, 2146, 2158, 2167, 2175, 2187, 2197, 2204, 2211, 2218, 2225, 2232, 2233, 2248, 2262, 2272, 2281, 2291, 2298, 2305, 2312, 2319, 2320, 2329, 2338, 2348, 2349, 2357, 2366, 2374, 2378, 2388, 2396, 2407, 2419, 2430, 2436, 2447, 2459, 2465, 2471, 2476, 2481, 2486, 2490, 2494, 2510, 2521, 2523, 2533, 2548, 2552, 2564, 2577, 2581, 2587, 2597, 2606, 2610, 2620, 2627, 2635, 2639, 2649, 2658, 2663, 2667, 2668, 2674, 2681, 2689, 2697, 2713, 2726, 2734, 2741, 2749, 2754, 2755, 2784, 2807, 2813, 2842, 2850, 2859, 2866, 2871, 2877, 2884, 2891, 2897, 2900, 2929, 2934, 2939, 2945, 2950, 2956, 2958, 2986]
        #
        # SR_list =[]
        # s=range(180)
        # s_frames=np.array(s).tolist()
        #s_frames=[0,30,60,90,120,150]
        #SR_list=s_frames
        # print("s_a",int(action/15))

        final_results = Results()
        # 取s_frames中帧做检测的f1score
        if s_frames:
            self.last_frame = s_frames[-1]
        time = 0
        #all_time = 0
        # print(self.idx-self.goplen)
        # [0，30） [30,60)
        for frame_idx in range(self.idx - self.goplen, self.idx):
            if frame_idx in s_frames:
                if frame_idx in SR_list:

                    self.l_result = self.h_result[frame_idx]
                    #self.l_result = self.result[frame_idx]
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        # print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                    # sr   1080 infer
                    time += 0.1724+0.0924+0.02
                #     0.2848
                else:
                    #for r in self.self.result[frame_idx]:
                    _d_result = Results()
                    #self.move(frame_idx)
                    # self.clear_sr(frame_idx)
                    # 就用lr infer 代替transfer+infer
                    self.l_result = self.result[frame_idx]

                    time += 0.0924+0.02
                    # 0.1124
                    # for r in self.l_result:
                    #     label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    #     r = Region(frame_idx, x, y, w, h, conf, label,
                    #                0, origin="mpeg")
                    #     r = self.resize(r)
                    #     if self.isdelete(r):
                    #         _d_result.append(r)
                    # self.l_result = merge_boxes_in_results(_d_result.regions_dict, 0.3, 0.3).regions_dict[frame_idx]

                    for r in self.l_result:
                        label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                        #print(y)
                        r = Region(frame_idx, x, y, w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)
                #all_time += self.times[frame_idx]
                continue
            # else:
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         # print(y)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         r = self.resize(r)
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         if self.isdelete(r):
            #             final_results.append(r)
            #     continue
            #     for r in self.l_result:
            #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
            #         r = Region(frame_idx, x, y, w, h, conf, label,
            #                    0, origin="mpeg")
            #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            #         final_results.append(r)
            #     all_time += self.times[frame_idx]
            #     continue
            if frame_idx + 1 in self.I_frame:
                for r in self.l_result:
                    label, conf, (x, y, w, h)=r.label,r.conf,(r.x,r.y,r.w,r.h)
                    r = Region(frame_idx, x, y, w, h, conf, label,
                               0, origin="mpeg")
                    r = self.resize(r)
                    # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                    if self.isdelete(r):
                        final_results.append(r)
#0.7868061224489794


            else:
                _d_result = Results()
                refer = find_refer(frame_idx + 1) - 1
                if refer + 1 == frame_idx:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        _x, _y = _x / 4, _y / 4
                        # _x, _y = 0, 0

                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                elif refer == (find_refer(frame_idx) - 1) and find_refer(frame_idx)!=-1:
                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_bbox(frame_idx + 1, x, y, w, h, refer + 1)
                        __x, __y = move_bbox(frame_idx, x, y, w, h, refer + 1)
                        if _x != 0 and _y != 0:
                            _x = _x - __x
                            _y = _y - __y
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf,
                                   label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                else:

                    for r in self.l_result:
                        label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
                        _x, _y = move_re_bbox(frame_idx, x, y, w, h, frame_idx)
                        _x, _y = _x / 4, _y / 4
                        r = Region(frame_idx, (x - _x / 1280), (y - _y / 720), w, h, conf, label,
                                   0, origin="mpeg")
                        r = self.resize(r)
                        _d_result.append(r)
                        # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                        if self.isdelete(r):
                            final_results.append(r)

                self.l_result = _d_result.regions_dict[frame_idx]
                # print('f_r',final_results.regions_dict)
            #all_time += self.times[frame_idx]
            # print('result',results)

        #final_results = merge_boxes_in_results(final_results.regions_dict, 0.5, 0.5)
        #self.show_results.combine_results(final_results)

        try:
            tp, fp, fn, _, _, _, f1 = evaluate(
                self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
                0.5, 0.5, 0.4, 0.4)
        except Exception:
            f1=0
        # tp, fp, fn, _, _, _, f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.5, 0.5, 0.04, 0.04)
        if SR_list!=s_frames:
            self.t_reward.append(f1)
        # final_results = Results()
        # # 取s_frames中帧做检测的f1score
        # #print('f1:',f1)
        # # print(self.idx-self.goplen)
        # for frame_idx in range(self.idx - self.goplen, self.idx):
        #
        #     results = self.result[frame_idx]
        #
        #     # all_time+=self.times[frame_idx]
        #     for r in results:
        #         label, conf, (x, y, w, h) = r.label, r.conf, (r.x, r.y, r.w, r.h)
        #         r = Region(frame_idx, x, y, w, h, conf, label,
        #                    0, origin="mpeg")
        #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
        #         final_results.append(r)
        #         # print('f_r',final_results.regions_dict)
        # tp, fp, fn, _, _, _, s_f1 = evaluate(
        #     self.idx - 1, final_results.regions_dict, self.ground_truth_dict,
        #     0.3, 0.5, 0.4, 0.4)

        self.f1_list.append(f1)
        self.last_compute_time=time
        time_trans=self.qp_bitrate[self.last_qp]/self.bandwith[int(self.idx / 30-1)+N_CHUNK-1]
        time+=time_trans
        # reward = self.a1 * (f1)  - self.a2 * (1 if time>1 else 0)
        reward = self.a1 * (f1) - self.a2 * (1 if time > 1 else 0)
        tuple=(time_trans,self.last_compute_time,time,self.bandwith[int(self.idx / 30-1)+N_CHUNK-1],self.qp_bitrate[self.last_qp],self.last_qp,f1,reward)
        self.reward_list.append(tuple)
        self.reward_sum += reward
        # 排队时间
        # self.queue =max(0,(self.queue+time-1))
        ###
        # 计算前一个gop剩余feature差值
        new_qp = qp_thres[int(action % N_qp)]
        self.diff_gop=self.diff_gop_dict[new_qp]
        # state_要拷贝新的chunk选的qp
        import copy
        state_ = copy.deepcopy(self.states_dict[new_qp][int(self.idx / self.goplen)])
        # print("state_.shape", np.array(state_).shape)
        # res_ = copy.deepcopy(self.res[self.idx-self.goplen+1:self.idx])
        # sun  31  60
        res_ = copy.deepcopy(self.res[self.idx+ 1:self.idx+self.goplen])
        if self.idx not in [150, 300, 450, 600, 750, 900, 1050, 1350, 1680, 2280, 2700]:
            if s_frames:
                # print('余下的',self.state,self.state[(s_frames[-1]%30)+1:])
                self.diff_last = np.sum(self.state[(s_frames[-1] % 30) + 1:]) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            else:
                self.diff_last += np.sum(self.state) + self.diff_gop[int(self.idx / 30) - 1]
                state_.insert(0, self.diff_last)
            # res_.insert(0,sum(self.res[self.last_sr_frame+1:self.idx-self.goplen+1]))
            # [20 31)
            res_.insert(0, sum(self.res[self.last_sr_frame + 1:self.idx + 1]))

        else:
            res_.insert(0,0)
            state_.insert(0, 0)
        # print("state_.shape",np.array(state_).shape)
        ###
        self.state = np.array(state_)
        # idx变成60
        self.idx += self.goplen
        # return state_, reward, (self.idx==self.length)
        if self.idx == 2970:
            done = True
        else:
            done = False
        #print(len(res_), self.idx)

        # 下一个chunk

        self.features=self.features_dict[new_qp]
        state_ += self.pca.transform([self.features[self.idx - self.goplen]])[0].tolist()
        state_+=res_
        # state_.append(self.queue)
        a=np.array(self.chunk_sizes)
        a=np.roll(a,1)
        a[0]=self.qp_bitrate[self.last_qp]
        self.chunk_sizes=a.tolist()
        a=np.array(self.chunk_times)
        a=np.roll(a,1)
        a[0]=self.chunk_sizes[0]/self.bandwith[int(self.idx/30-2)+N_CHUNK-1]
        # print(time_trans,a[0],"is equal")  相等
        self.chunk_times=a.tolist()
        # state_.extend(self.chunk_sizes)
        # state_.extend(self.chunk_times)
        state_.append(self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        state_.append(self.qp_bitrate[self.last_qp])
        state_.append(time_trans)
        state_.append(self.last_compute_time)
        print("action", action, '\n', 'idx', self.idx, '\n', 'goplen', self.goplen)
        print("s sr q_a last_qp qp", int(action / (N_rthres*N_qp)), int(action %  (N_rthres*N_qp) / N_qp), action % N_qp, self.last_qp,
              qp_thres[action % N_qp])
        print('s_frames', s_frames)
        print('SR',SR_list)
        print("last_qp bitrate bandwidth",self.last_qp,self.qp_bitrate[self.last_qp] ,self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1])
        print('time ',time_trans,self.last_compute_time,time)
        print('f1',f1)
        # print('chunk_sizes',self.chunk_sizes)
        # print('chunk_times',self.chunk_times)
        # if self.idx>=2940:
        #     print("qp list:",self.qp_trace)
        # 这里的qp决定的是下一个chunk的qp，但是这一个chunk已经编码好了（并且取出了编码器信息），所以不太好修改
        # 根据action更新qp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.last_qp=new_qp
        self.qp_trace.append(self.last_qp)
        return np.array(state_), reward, done, None

    def evaluate_(self,model_name):
        #              max_tuple={"thres": thres1,
        #                                    "rthres": thres2,
        #                                    "qp": qp,
        #                                    "bandwidth": self.bandwith[int(self.idx / 30 - 1)],
        #                                    "trans_time": time_trans,
        #                                    "comp_time": self.last_compute_time,
        #                                    "total_time": time,
        #                                    "f1": f1,
        #                                    "reward": reward,
        #                                    "sr_list":SR_list,
        #                                    "infer_list": s_frames
        #                                    }

        with open("search.txt", "rb") as myfile:
            search_reward = pickle.load(myfile)
        sreward_=[]
        for dic in search_reward:
            sreward_.append(dic["reward"])

        # gt 和 待检测帧号不匹配没关系，dds evaluate函数会从0~maxfid搜索帧号是否存在于待检测文件中
        print(self.idx)
        print(self.reward_list)
        print("f1_mean", mean(self.f1_list))
        print(" total reward", self.reward_sum)
        # print('f1_all', f1, " total reward", self.reward_sum)
        band_ = []
        qp_ = []
        # tuple = (time_trans, self.last_compute_time, time, self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1],
        #          self.qp_bitrate[self.last_qp], self.last_qp, f1, reward)
        reward_=[]
        for _, _, _, band, _,qp,_, reward in self.reward_list:
            band_.append(band)
            qp_.append(self.qp_bitrate[qp])
            reward_.append(reward)

        with open(f"reward_{model_name}_band1.txt","wb") as myfile:
            pickle.dump(self.reward_list,myfile)

        # band_=Normalize(band_)
        # qp_=Normalize(qp_)

        print(band_)
        print(qp_)
        x = np.arange(len(qp_))
        plt.figure(1)
        plt.plot(x, band_, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='Bandwidth')
        plt.plot(x, qp_, color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Adaptive bitrate")
        plt.ylabel('Bandwidth (Mbps)')
        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"band_qp_{model_name}.png")
        plt.show()

        plt.figure(2)
        x = np.arange(len(reward_) - 1)
        plt.plot(x, self.f1_list[1:], color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='F1-score')
        plt.plot(x, reward_[1:], color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="Reward")

        for i in range(0,100,10):

            plt.plot([i,i], [0,1], 'k--', linewidth=1)

        for i,acc in enumerate(self.f1_list):
            if acc>0.8:
                print(i,acc)

        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"f1_reward_{model_name}.png")
        plt.show()

        plt.figure(3)
        x=np.arange(len(reward_)-1)
        sum=0
        sum_p=0
        print(len(reward_))
        for i in range(1,len(reward_)):
            dif=math.fabs(sreward_[i]-reward_[i])
            sum+=dif
            sum_p+=dif*dif
        print("reward 误差累计  平均误差 均方误差：",sum,sum/(len(reward_)-1),sum_p/(len(reward_)-1))
        plt.plot(x, sreward_[1:], color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
                 label='Opt reward')
        plt.plot(x,reward_[1:], color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
                 label="DRL reward")
        plt.xlabel('Chunk-id')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"different_reward_{model_name}.png")
        plt.show()


