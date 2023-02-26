# # import pickle
# # from dds_utils import evaluate
# # path="/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/DRL_continuous/time_long.txt"
# # # with open("time_sr.txt","rb") as myfile:
# # #     time=pickle.load(myfile)
# #
# #
# #
# # import argparse
# #
# #
# # # parser.add_argument('--video_path',type=str)
# # # parser.add_argument('--qp',type=int)
# # # args=parser.parse_args()
# # # qp=args.qp
# # #
# #
# # # 720
# #
# # # "./DRL_continuous/dds_results.txt"
# # # with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/dds_results360_10_plusSR.txt", "rb") as get_myprofile:
# # #     h_result = pickle.load(get_myprofile)
# # #
# # # # 360
# # # with open("dds_results360_10_plusSR.txt","rb") as myfile:
# # #     result=pickle.load(myfile)
# # #
# # #
# # # tp, fp, fn, _, _, _, f1 = evaluate(
# # #     2999, result.regions_dict, h_result.regions_dict,
# # #     0.3, 0.5, 0.4, 0.4)
# # #
# # # print("tp, fp, fn, f1",tp, fp, fn, f1)
# # # with open("dds_results360_20.txt","rb") as myfile:
# # #     result=pickle.load(myfile)
# # #
# # # tp, fp, fn, _, _, _, f1 = evaluate(
# # #     2999, result.regions_dict, h_result.regions_dict,
# # #     0.3, 0.5, 0.4, 0.4)
# # #
# # # print("tp, fp, fn, f1", tp, fp, fn, f1)
# # # with open("dds_results360_30.txt","rb") as myfile:
# # #     result=pickle.load(myfile)
# # #
# # #
# # # tp, fp, fn, _, _, _, f1 = evaluate(
# # #     2999, result.regions_dict, h_result.regions_dict,
# # #     0.3, 0.5, 0.4, 0.4)
# # #
# # # print("tp, fp, fn, f1",tp, fp, fn, f1)
# # # with open("dds_results360_40.txt","rb") as myfile:
# # #     result=pickle.load(myfile)
# # # tp, fp, fn, _, _, _, f1 = evaluate(
# # #     2999, result.regions_dict, h_result.regions_dict,
# # #     0.3, 0.5, 0.4, 0.4)
# # # print("tp, fp, fn, f1",tp, fp, fn, f1)
# # #
# # #
# # # with open("dds_results360_50.txt","rb") as myfile:
# # #     result=pickle.load(myfile)
# # # tp, fp, fn, _, _, _, f1 = evaluate(
# # #     2999, result.regions_dict, h_result.regions_dict,
# # #     0.3, 0.5, 0.4, 0.4)
# # # print("tp, fp, fn, f1",tp, fp, fn, f1)
# # #
# # #
# # # print("yes")
# # #
# # # #
# #
# # # from numpy import *
# # # # 看一下时间一不一样
# # # print("360不同qp infer平均时间")
# # # with open("time_10.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # # with open("time_20.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # # with open("time_30.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # # with open("time_40.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # # with open("time_50.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # #
# # # print("-----------------------------------")
# # # print("360p不同qp SR 到 720时间")
# # # with open("time_sr.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # #
# # # print("360p不同qp SR 到 720  再infer的平均时间：")
# # # with open("time_10_plusSR.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # #
# # #
# # # with open("time_20_plusSR.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # #
# # # with open("time_30_plusSR.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # #
# # # with open("time_40_plusSR.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # #
# # # with open("time_50_plusSR.txt","rb") as myfile:
# # #     times=pickle.load(myfile)
# # # print(mean(times))
# # # # # gt是720p视频  待检测是360p lr
# # # # tp, fp, fn, f1 41768 819 17908 0.817
# # # # tp, fp, fn, f1 41393 941 18248 0.812
# # # # tp, fp, fn, f1 40177 1143 19412 0.796
# # # # tp, fp, fn, f1 33171 2352 26367 0.698
# # # # tp, fp, fn, f1 16216 4018 43575 0.405
# # #
# # #
# # # !!!!!!!!!!!!!!!!!正常来说SR必须比SR好选帧才会有效果！！！！！！！！这里有问题
# # # # gt是360SR两倍的
# # # # tp, fp, fn, f1 47081 0 11638 0.89
# # # # tp, fp, fn, f1 39331 3003 19072 0.781
# # # # tp, fp, fn, f1 38363 2957 19890 0.771
# # # # tp, fp, fn, f1 32466 3057 25866 0.692
# # # # tp, fp, fn, f1 16152 4082 42356 0.41
# # # # yes
# # # # 360不同qp infer平均时间
# # # # 0.042694468021392824
# # # # 0.04249575368563334
# # # # 0.042218618551890054
# # # # 0.043145729303359984
# # # # 0.04522974459330241
# # # # -----------------------------------
# # # # 360p不同qp SR 到 720时间
# # # # 0.07981106821695963
# # # # 360p不同qp SR 到 720  再infer的平均时间：
# # # # 0.14513396779696147
# # # # 0.14450454211235048
# # # # 0.146069038550059
# # # 0.1391157956123352
# #
#
#
#
#
# import pickle
# from dds_utils import evaluate
# qp=[10,15,20,25,30,35,40,45,50]
# # 其实SR的效果是非常差的，如果拿720p原图当成gt，sr还不如lr高。但是如果拿540每帧SR当做gt，SR效果好，挺合理的
# with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/dds_results540_10_plusSR.txt", "rb") as get_myprofile:
#     h_result = pickle.load(get_myprofile)
# # with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/DRL_continuous/dds_results.txt", "rb") as get_myprofile:
# #     h_result = pickle.load(get_myprofile)
#
# with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/DRL_continuous/dds_results.txt","rb") as myfile:
#     result=pickle.load(myfile)
#
#
# tp, fp, fn, _, _, _, f1 = evaluate(
#     2999, result.regions_dict, h_result.regions_dict,
#     0.5, 0.5, 0.4, 0.4)
#
# print("tp, fp, fn, f1",tp, fp, fn, f1)
#
# # 360
# print('540 lr result----------------')
#
# for i in qp:
#     with open(f"dds_results540_{i}.txt","rb") as myfile:
#         result=pickle.load(myfile)
#
#
#     tp, fp, fn, _, _, _, f1 = evaluate(
#         2999, result.regions_dict, h_result.regions_dict,
#         0.5, 0.5, 0.4, 0.4)
#
#     print("tp, fp, fn, f1",tp, fp, fn, f1)
#
#
#
#
# # SR
#
# print('540 SR result-------------------------')
# for i in qp:
#     with open(f"dds_results540_{i}_plusSR.txt","rb") as myfile:
#         result=pickle.load(myfile)
#
#
#     tp, fp, fn, _, _, _, f1 = evaluate(
#         2999, result.regions_dict, h_result.regions_dict,
#         0.5, 0.5, 0.4, 0.4)
#
#     print("tp, fp, fn, f1",tp, fp, fn, f1)
#
# print("yes")
#
#
#
# # #
# # from numpy import *
# # # 看一下时间一不一样
# # print("540 lr 不同qp infer平均时间")
# # qp=[10,20,30,40,50]
# # for i in qp:
# #     with open(f"time_{i}_infer.txt","rb") as myfile:
# #         times=pickle.load(myfile)
# #     print(mean(times))
# #
# #
# # print("-----------------------------------")
# # print("540p不同qp SR 到 1080时间(sr时间和内容无关）")
# # # f"time540_{qp}_SR.txt"
# #
# # qp=[10,20,30,40,50]
# # for i in qp:
# #     with open(f"time540_{i}_sr.txt","rb") as myfile:
# #         times=pickle.load(myfile)
# #     print(mean(times))
# #
# # print("540p不同qp SR 到 1080后的 infer平均时间（可能越清晰，物体越多，时间越长）：")
# # qp=[10,20,30,40,50]
# # for i in qp:
# #     with open(f"time540x2_{i}_infer.txt","rb") as myfile:
# #         times=pickle.load(myfile)
# #     print(mean(times))
#
# # # 540 lr result----------------
# # # tp, fp, fn, f1 42114 2427 18684 0.8
# # # tp, fp, fn, f1 41884 2538 18852 0.797
# # # tp, fp, fn, f1 41367 2788 19310 0.789
# # # tp, fp, fn, f1 37871 2747 22996 0.746
# # # tp, fp, fn, f1 21927 3754 38974 0.507
# # # 540 SR result-------------------------
# # # tp, fp, fn, f1 48924 0 12069 0.89
# # # tp, fp, fn, f1 48578 377 12366 0.884
# # # tp, fp, fn, f1 46600 935 14218 0.86
# # # tp, fp, fn, f1 42588 2173 18136 0.807
# # # tp, fp, fn, f1 27292 4741 33245 0.59
# # # yes
# # # 540 lr 不同qp infer平均时间
# # # 0.0450307408173879
# # # 0.04493511279424032
# # # 0.044661639531453454
# # # 0.04458990025520325
# # # 0.045493594566980995
# # #
# # # 540infer
# # # 0.0448
# # # -----------------------------------
# # # 540p不同qp SR 到 1080时间(sr时间和内容无关）
# # # 0.17427484075228372
# # # 0.1726638096968333
# # # 0.17296559675534567
# # # 0.17210231637954712
# # # 0.17219318691889446
# # # sr
# # # 0.1724
# #
# # # 540p不同qp SR 到 1080后的 infer平均时间（可能越清晰，物体越多，时间越长）：
# # # 0.09574651392300923
# # # 0.08876817234357198
# # # 0.1006418510278066
# # # 0.08987907425562541
# # # 0.09040443897247315
# # #
# # # 1080infer
# # # 0.0924
# #
# # # 540 lr result----------------
# # # tp, fp, fn, f1 38879 5662 10028 0.832
# # # tp, fp, fn, f1 38615 5807 10243 0.828
# # # tp, fp, fn, f1 38110 6045 10698 0.82
# # # tp, fp, fn, f1 34976 5642 13977 0.781
# # # tp, fp, fn, f1 20400 5281 28486 0.547
# # # 540 SR result-------------------------
# # # tp, fp, fn, f1 48924 0 0 1.0
# # # tp, fp, fn, f1 45955 3000 3018 0.939
# # # tp, fp, fn, f1 43270 4265 5639 0.897
# # # tp, fp, fn, f1 39068 5693 9734 0.835
# # # tp, fp, fn, f1 25179 6854 23500 0.624
# # #
# # import numpy as np
# # a=np.zeros((8,))
# # print("eww")
#
#
# # (dds2) ubuntu@inst10:~/VideoAnalytics/workplace/DRL_SR_Infer$ python try.py
# # tp, fp, fn, f1 40029 6425 9049 0.8380228614495666
# # 540 lr result----------------
# # tp, fp, fn, f1 38879 5662 10028 0.8320991353480011
# # tp, fp, fn, f1 38741 5572 10138 0.8314232981371792
# # tp, fp, fn, f1 38615 5807 10243 0.8279373927958834
# # tp, fp, fn, f1 38420 5938 10365 0.8249680598649388
# # tp, fp, fn, f1 38110 6045 10698 0.8198960876908017
# # tp, fp, fn, f1 37357 5910 11599 0.8101449746809364
# # tp, fp, fn, f1 34976 5642 13977 0.7809670540688393
# # tp, fp, fn, f1 29199 5444 19655 0.6994023737379786
# # tp, fp, fn, f1 20400 5281 28486 0.5471589308943634
# # 540 SR result-------------------------
# # tp, fp, fn, f1 48924 0 0 1.0
# # tp, fp, fn, f1 46594 2269 2381 0.9524724544655451
# # tp, fp, fn, f1 45955 3000 3018 0.9385466873621436
# # tp, fp, fn, f1 44982 3658 3906 0.9224427856615536
# # tp, fp, fn, f1 43270 4265 5639 0.8973082825266476
# # tp, fp, fn, f1 41618 4780 7356 0.8727509122174223
# # tp, fp, fn, f1 39068 5693 9734 0.8351164456034971
# # tp, fp, fn, f1 32926 5860 15831 0.752224621043373
# # tp, fp, fn, f1 25179 6854 23500 0.6239220933690157
# # yes

"""   kkkkkkk  """

# import pickle
# from dds_utils import evaluate
# path="/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/DRL_continuous/time_long.txt"
# # with open("time_sr.txt","rb") as myfile:
# #     time=pickle.load(myfile)
#
#
#
# import argparse
#
#
# # parser.add_argument('--video_path',type=str)
# # parser.add_argument('--qp',type=int)
# # args=parser.parse_args()
# # qp=args.qp
# #
#
# # 720
#
# # "./DRL_continuous/dds_results.txt"
# # with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/dds_results360_10_plusSR.txt", "rb") as get_myprofile:
# #     h_result = pickle.load(get_myprofile)
# #
# # # 360
# # with open("dds_results360_10_plusSR.txt","rb") as myfile:
# #     result=pickle.load(myfile)
# #
# #
# # tp, fp, fn, _, _, _, f1 = evaluate(
# #     2999, result.regions_dict, h_result.regions_dict,
# #     0.3, 0.5, 0.4, 0.4)
# #
# # print("tp, fp, fn, f1",tp, fp, fn, f1)
# # with open("dds_results360_20.txt","rb") as myfile:
# #     result=pickle.load(myfile)
# #
# # tp, fp, fn, _, _, _, f1 = evaluate(
# #     2999, result.regions_dict, h_result.regions_dict,
# #     0.3, 0.5, 0.4, 0.4)
# #
# # print("tp, fp, fn, f1", tp, fp, fn, f1)
# # with open("dds_results360_30.txt","rb") as myfile:
# #     result=pickle.load(myfile)
# #
# #
# # tp, fp, fn, _, _, _, f1 = evaluate(
# #     2999, result.regions_dict, h_result.regions_dict,
# #     0.3, 0.5, 0.4, 0.4)
# #
# # print("tp, fp, fn, f1",tp, fp, fn, f1)
# # with open("dds_results360_40.txt","rb") as myfile:
# #     result=pickle.load(myfile)
# # tp, fp, fn, _, _, _, f1 = evaluate(
# #     2999, result.regions_dict, h_result.regions_dict,
# #     0.3, 0.5, 0.4, 0.4)
# # print("tp, fp, fn, f1",tp, fp, fn, f1)
# #
# #
# # with open("dds_results360_50.txt","rb") as myfile:
# #     result=pickle.load(myfile)
# # tp, fp, fn, _, _, _, f1 = evaluate(
# #     2999, result.regions_dict, h_result.regions_dict,
# #     0.3, 0.5, 0.4, 0.4)
# # print("tp, fp, fn, f1",tp, fp, fn, f1)
# #
# #
# # print("yes")
# #
# # #
#
# # from numpy import *
# # # 看一下时间一不一样
# # print("360不同qp infer平均时间")
# # with open("time_10.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# # with open("time_20.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# # with open("time_30.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# # with open("time_40.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# # with open("time_50.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# #
# # print("-----------------------------------")
# # print("360p不同qp SR 到 720时间")
# # with open("time_sr.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# #
# # print("360p不同qp SR 到 720  再infer的平均时间：")
# # with open("time_10_plusSR.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# #
# #
# # with open("time_20_plusSR.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# #
# # with open("time_30_plusSR.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# #
# # with open("time_40_plusSR.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# #
# # with open("time_50_plusSR.txt","rb") as myfile:
# #     times=pickle.load(myfile)
# # print(mean(times))
# # # # gt是720p视频  待检测是360p lr
# # # tp, fp, fn, f1 41768 819 17908 0.817
# # # tp, fp, fn, f1 41393 941 18248 0.812
# # # tp, fp, fn, f1 40177 1143 19412 0.796
# # # tp, fp, fn, f1 33171 2352 26367 0.698
# # # tp, fp, fn, f1 16216 4018 43575 0.405
# #
# #
# # !!!!!!!!!!!!!!!!!正常来说SR必须比SR好选帧才会有效果！！！！！！！！这里有问题
# # # gt是360SR两倍的
# # # tp, fp, fn, f1 47081 0 11638 0.89
# # # tp, fp, fn, f1 39331 3003 19072 0.781
# # # tp, fp, fn, f1 38363 2957 19890 0.771
# # # tp, fp, fn, f1 32466 3057 25866 0.692
# # # tp, fp, fn, f1 16152 4082 42356 0.41
# # # yes
# # # 360不同qp infer平均时间
# # # 0.042694468021392824
# # # 0.04249575368563334
# # # 0.042218618551890054
# # # 0.043145729303359984
# # # 0.04522974459330241
# # # -----------------------------------
# # # 360p不同qp SR 到 720时间
# # # 0.07981106821695963
# # # 360p不同qp SR 到 720  再infer的平均时间：
# # # 0.14513396779696147
# # # 0.14450454211235048
# # # 0.146069038550059
# # 0.1391157956123352
#


"""不同bitrate"""

import pickle
from dds_utils import evaluate


import ffmpeg

def get_video_info(video_path):
    probe = ffmpeg.probe(video_path)
    # print(probe)  # 获取视频多媒体文件的信息
    format = probe['format']
    # print(format)
    bit_rate = format['bit_rate']
    # print(int(bit_rate))  # 单位 bps
    kbps = int(bit_rate) / 1024

    duration = format['duration']
    duration = int(float(duration))  # 时长（单位秒）
    # print(duration)

    # print(int(format['size']))  # 获取文件大小（单位字节）
    # print(int(int(probe['streams'][0]['r_frame_rate'].split('/')[0]) / int(
    #     probe['streams'][0]['r_frame_rate'].split('/')[1])))  # 获取帧率

    # 通过比特率X时长/8 计算文件大小
    file_size = float(bit_rate) * duration / 8
    # print(file_size)  # 計算得到视频的大小是 B  小一点
    # print(int(format['size']))
    # print(file_size / 1024/1024)  # 计算得到的数据
    # 计算得到数据 和读取得到的数据不太一样，直接读取的大一点，但是查看视频属性更接近直接读取得到的数据
    # print(int(format['size']) / 1024 / 1024)  # 读取得到的数据
    return int(bit_rate),int(format['size'])




bitrate=[200,300,450,750,1200,1850,2850,4300,6000,8000]
qp=[10,15,20,25,30,35,40,45]
#
# bitrates=[]
# sizes=[]
# f=open("bitrates.txt","w")
# f2=open("sizes.txt","w")
#
# video_path=f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_bit/video_test540_8000/video_test540_8000.mp4"
# get_video_info(video_path)
# for i in qp:
#     video_path=f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset/video_test540_{i}/video_test540_{i}.mp4"
#     tmp1,tmp2=get_video_info(video_path)
#     # bps
#     f.write(f"{tmp1}\n")
#     f2.write(f"{tmp2}\n")
#     bitrates.append(tmp1)
#     # byte
#     sizes.append(tmp2)
# f.close()
# f2.close()
# print(bitrates)
# print(sizes)
# 其实SR的效果是非常差的，如果拿720p原图当成gt，sr还不如lr高。但是如果拿540每帧SR当做gt，SR效果好，挺合理的

# with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/constbit_dds_results540_8000_plusSR.txt", "rb") as get_myprofile:
#     h_result = pickle.load(get_myprofile)
# with open("/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/dds_results540_10_plusSR.txt", "rb") as get_myprofile:
#     result2 = pickle.load(get_myprofile)
# with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/DRL_continuous/dds_results.txt","rb") as myfile:
#     gt=pickle.load(myfile)
#
#
# tp, fp, fn, _, _, _, f1 = evaluate(
#     2999, h_result.regions_dict, gt.regions_dict,
#     0.5, 0.5, 0.4, 0.4)
# print("tp, fp, fn, f1",tp, fp, fn, f1)
# tp, fp, fn, _, _, _, f1 = evaluate(
#     2999, result2.regions_dict, gt.regions_dict,
#     0.5, 0.5, 0.4, 0.4)
# print("tp, fp, fn, f1",tp, fp, fn, f1)
# # 符合常识   qp=10 对应的bitrate更大
# # 8000bps   tp, fp, fn, f1 39819 9086 6495 0.8363666915216501
# # qp=10     tp, fp, fn, f1 39875 9049 6425 0.837498949844577

with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/DRL_continuous/dds_results.txt","rb") as myfile:
    h_result = pickle.load(myfile)
print('540 lr result----------------')
bitrate_list=[10910693, 6050811, 3196561, 1627969, 825260, 450706, 273220, 180834]


for i in bitrate_list:
    video_path = f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_profile/video_test540_{i}/video_test540_{i}.mp4"
    bit,size=get_video_info(video_path)

    with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_profile/constbit_dds_results540_{i}.txt","rb") as myfile:
        result=pickle.load(myfile)


    tp, fp, fn, _, _, _, f1 = evaluate(
        2999, result.regions_dict, h_result.regions_dict,
        0.5, 0.5, 0.4, 0.4)

    print(bit,size,f1)

print("720 lr result--------------------------")

for i in bitrate_list:
    video_path = f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_profile/video_test540_{i}/video_test540_{i}_720p.mp4"
    bit,size = get_video_info(video_path)
    with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_profile/constbit_dds_results540_{i}_720p.txt","rb") as myfile:
        result=pickle.load(myfile)


    tp, fp, fn, _, _, _, f1 = evaluate(
        2999, result.regions_dict, h_result.regions_dict,
        0.5, 0.5, 0.4, 0.4)

    print(bit,size,f1)


print("360 lr result--------------------------")

for i in bitrate_list:
    video_path = f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_profile/video_test540_{i}/video_test540_{i}_360p.mp4"
    bit,size = get_video_info(video_path)
    with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_profile/constbit_dds_results540_{i}_360p.txt","rb") as myfile:
        result=pickle.load(myfile)


    tp, fp, fn, _, _, _, f1 = evaluate(
        2999, result.regions_dict, h_result.regions_dict,
        0.5, 0.5, 0.4, 0.4)

    print(bit,size,f1)

# SR

# print('540 SR result-------------------------')
# for i in bitrate:
#     with open(f"constbit_dds_results540_{i}_plusSR.txt","rb") as myfile:
#         result=pickle.load(myfile)
#
#
#     tp, fp, fn, _, _, _, f1 = evaluate(
#         2999, result.regions_dict, h_result.regions_dict,
#         0.5, 0.5, 0.4, 0.4)
#
#     print("tp, fp, fn, f1",tp, fp, fn, f1)
#
# print("yes")




"""不同qp"""



print('540 lr result of different qp----------------')

for i in qp:
    with open(f"dds_results540_{i}.txt","rb") as myfile:
        result=pickle.load(myfile)


    tp, fp, fn, _, _, _, f1 = evaluate(
        2999, result.regions_dict, h_result.regions_dict,
        0.5, 0.5, 0.4, 0.4)

    print(f1)




# # SR
#
# print('540 SR result-------------------------')
# for i in qp:
#     with open(f"dds_results540_{i}_plusSR.txt","rb") as myfile:
#         result=pickle.load(myfile)
#
#
#     tp, fp, fn, _, _, _, f1 = evaluate(
#         2999, result.regions_dict, h_result.regions_dict,
#         0.5, 0.5, 0.4, 0.4)
#
#     print("tp, fp, fn, f1",tp, fp, fn, f1)
#
# print("yes")

# #
# from numpy import *
# # 看一下时间一不一样
# print("540 lr 不同qp infer平均时间")
# qp=[10,20,30,40,50]
# for i in qp:
#     with open(f"time_{i}_infer.txt","rb") as myfile:
#         times=pickle.load(myfile)
#     print(mean(times))
#
#
# print("-----------------------------------")
# print("540p不同qp SR 到 1080时间(sr时间和内容无关）")
# # f"time540_{qp}_SR.txt"
#
# qp=[10,20,30,40,50]
# for i in qp:
#     with open(f"time540_{i}_sr.txt","rb") as myfile:
#         times=pickle.load(myfile)
#     print(mean(times))
#
# print("540p不同qp SR 到 1080后的 infer平均时间（可能越清晰，物体越多，时间越长）：")
# qp=[10,20,30,40,50]
# for i in qp:
#     with open(f"time540x2_{i}_infer.txt","rb") as myfile:
#         times=pickle.load(myfile)
#     print(mean(times))

# # 540 lr result----------------
# # tp, fp, fn, f1 42114 2427 18684 0.8
# # tp, fp, fn, f1 41884 2538 18852 0.797
# # tp, fp, fn, f1 41367 2788 19310 0.789
# # tp, fp, fn, f1 37871 2747 22996 0.746
# # tp, fp, fn, f1 21927 3754 38974 0.507
# # 540 SR result-------------------------
# # tp, fp, fn, f1 48924 0 12069 0.89
# # tp, fp, fn, f1 48578 377 12366 0.884
# # tp, fp, fn, f1 46600 935 14218 0.86
# # tp, fp, fn, f1 42588 2173 18136 0.807
# # tp, fp, fn, f1 27292 4741 33245 0.59
# # yes
# # 540 lr 不同qp infer平均时间
# # 0.0450307408173879
# # 0.04493511279424032
# # 0.044661639531453454
# # 0.04458990025520325
# # 0.045493594566980995
# #
# # 540infer
# # 0.0448
# # -----------------------------------
# # 540p不同qp SR 到 1080时间(sr时间和内容无关）
# # 0.17427484075228372
# # 0.1726638096968333
# # 0.17296559675534567
# # 0.17210231637954712
# # 0.17219318691889446
# # sr
# # 0.1724
#
# # 540p不同qp SR 到 1080后的 infer平均时间（可能越清晰，物体越多，时间越长）：
# # 0.09574651392300923
# # 0.08876817234357198
# # 0.1006418510278066
# # 0.08987907425562541
# # 0.09040443897247315
# #
# # 1080infer
# # 0.0924
#
# # 540 lr result----------------
# # tp, fp, fn, f1 38879 5662 10028 0.832
# # tp, fp, fn, f1 38615 5807 10243 0.828
# # tp, fp, fn, f1 38110 6045 10698 0.82
# # tp, fp, fn, f1 34976 5642 13977 0.781
# # tp, fp, fn, f1 20400 5281 28486 0.547
# # 540 SR result-------------------------
# # tp, fp, fn, f1 48924 0 0 1.0
# # tp, fp, fn, f1 45955 3000 3018 0.939
# # tp, fp, fn, f1 43270 4265 5639 0.897
# # tp, fp, fn, f1 39068 5693 9734 0.835
# # tp, fp, fn, f1 25179 6854 23500 0.624
# #
# import numpy as np
# a=np.zeros((8,))
# print("eww")


# (dds2) ubuntu@inst10:~/VideoAnalytics/workplace/DRL_SR_Infer$ python try.py
# tp, fp, fn, f1 40029 6425 9049 0.8380228614495666
# 540 lr result----------------
# tp, fp, fn, f1 38879 5662 10028 0.8320991353480011
# tp, fp, fn, f1 38741 5572 10138 0.8314232981371792
# tp, fp, fn, f1 38615 5807 10243 0.8279373927958834
# tp, fp, fn, f1 38420 5938 10365 0.8249680598649388
# tp, fp, fn, f1 38110 6045 10698 0.8198960876908017
# tp, fp, fn, f1 37357 5910 11599 0.8101449746809364
# tp, fp, fn, f1 34976 5642 13977 0.7809670540688393
# tp, fp, fn, f1 29199 5444 19655 0.6994023737379786
# tp, fp, fn, f1 20400 5281 28486 0.5471589308943634
# 540 SR result-------------------------
# tp, fp, fn, f1 48924 0 0 1.0
# tp, fp, fn, f1 46594 2269 2381 0.9524724544655451
# tp, fp, fn, f1 45955 3000 3018 0.9385466873621436
# tp, fp, fn, f1 44982 3658 3906 0.9224427856615536
# tp, fp, fn, f1 43270 4265 5639 0.8973082825266476
# tp, fp, fn, f1 41618 4780 7356 0.8727509122174223
# tp, fp, fn, f1 39068 5693 9734 0.8351164456034971
# tp, fp, fn, f1 32926 5860 15831 0.752224621043373
# tp, fp, fn, f1 25179 6854 23500 0.6239220933690157
# yes


# constbit
# tp, fp, fn, f1 39959 6495 9086 0.8368464591252265
# 540 lr result----------------
# tp, fp, fn, f1 33126 5659 15691 0.7562841031026689
# tp, fp, fn, f1 35355 5484 13373 0.7894648698739491
# tp, fp, fn, f1 36926 6278 11742 0.8038575409265064
# tp, fp, fn, f1 37867 6003 10984 0.8167944694297947
# tp, fp, fn, f1 38339 5883 10464 0.8242730448804085
# tp, fp, fn, f1 38539 5934 10263 0.8263521844009649
# tp, fp, fn, f1 38584 5820 10260 0.8275566231983528
# tp, fp, fn, f1 38680 5865 10181 0.8282123204076826
# tp, fp, fn, f1 38662 5691 10156 0.8299148876796428
# tp, fp, fn, f1 38712 5718 10139 0.8300082546284882
# 540 SR result-------------------------
# tp, fp, fn, f1 37184 5550 11882 0.8101089324618737
# tp, fp, fn, f1 39925 5161 8898 0.8502912393913257
# tp, fp, fn, f1 41624 5189 7125 0.8711412486134656
# tp, fp, fn, f1 43257 4360 5595 0.8968062279074107
# tp, fp, fn, f1 44411 3974 4534 0.912586047467379
# tp, fp, fn, f1 45128 3606 3793 0.9242332701858583
# tp, fp, fn, f1 45739 3254 3217 0.9339350069934353
# tp, fp, fn, f1 46029 2794 2844 0.9422903701277432
# tp, fp, fn, f1 46477 2390 2447 0.950537370514669
# tp, fp, fn, f1 48905 0 0 1.0
# yes
