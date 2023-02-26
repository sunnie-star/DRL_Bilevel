# # import cv2 as cv
# # from backend.object_detector import Detector
# # import time as T
# # from dds_utils import (Results, merge_boxes_in_results,Region)
# #
# # import argparse
# # from video_processor import VideoProcessor
# # parser=argparse.ArgumentParser(description="diff_pq")
# # parser.add_argument('--video_path',type=str)
# # parser.add_argument('--qp',type=int)
# # args=parser.parse_args()
# # qp=args.qp
# #
# # video_path=args.video_path
# #
# # result = []
# # model = Detector()
# # times=[]
# #
# # frame_idx=0
# # with VideoProcessor(video_path,0) as video:
# #     for frame in video:
# #         starttime = T.time()
# #         result.append(model.infer(frame))
# #         # result.append([])
# #         endtime = T.time()
# #         times.append(float(endtime - starttime))
# #         # 从0帧 开始
# #         print("finish", frame_idx)
# #         frame_idx+=1
# #
# # print(len(result))
# # final_results=Results()
# # for frame_idx in range(0,3000):
# #     # print(self.l_results,self.result[frame_idx])
# #     for label, conf, (x, y, w, h) in result[frame_idx]:
# #         r = Region(frame_idx, x, y, w, h, conf, label,
# #                    0, origin="mpeg")
# #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
# #         final_results.append(r)
# # result=merge_boxes_in_results(final_results.regions_dict, 0.3, 0.3)
# #
# #
# # import pickle
# # with open(f"dds_results540_{qp}.txt", "wb") as myprofile:
# #     pickle.dump(result, myprofile)
# # # with open(f"time_{qp}_infer.txt","wb") as myprofile:
# # #     pickle.dump(times,myprofile)
# #
# #
# #
# #
# #
# # ################################SR之后infer的结果
# # import cv2 as cv
# # from backend.object_detector import Detector
# # import time as T
# # from dds_utils import (Results, merge_boxes_in_results,Region)
# #
# # import argparse
# # from video_processor import VideoProcessor
# # from backend.server import Server
# # # parser=argparse.ArgumentParser(description="diff_pq")
# # # parser.add_argument('--video_path',type=str)
# # # parser.add_argument('--qp',type=int)
# # # args=parser.parse_args()
# # # qp=args.qp
# # #
# # # video_path=args.video_path
# #
# # result = []
# # model = Detector()
# # times=[]
# #
# # frame_idx=0
# #
# #
# # server=Server()
# # result,times_sr,times_infer=server.easy_perform_detection_edsr_on_video(video_path)
# #
# # final_results=Results()
# # for frame_idx in range(0,3000):
# #     # print(self.l_results,self.result[frame_idx])
# #     for label, conf, (x, y, w, h) in result[frame_idx]:
# #         r = Region(frame_idx, x, y, w, h, conf, label,
# #                    0, origin="mpeg")
# #         # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
# #         final_results.append(r)
# # result=merge_boxes_in_results(final_results.regions_dict, 0.3, 0.3)
# #
# #
# # import pickle
# # with open(f"dds_results540_{qp}_plusSR.txt", "wb") as myprofile:
# #     pickle.dump(result, myprofile)
# # with open(f"time540_{qp}_sr.txt","wb") as myprofile:
# #     pickle.dump(times_sr,myprofile)
# # #     sr的时间和内容复杂度无关，但是
# # with open(f"time540x2_{qp}_infer.txt", "wb") as myprofile:
# #     pickle.dump(times_infer, myprofile)
#
#
# #
# # ######################################################算不同qp
# # #
# import cv2 as cv
# from backend.object_detector import Detector
# import time as T
# from dds_utils import (Results, merge_boxes_in_results,Region)
#
# import argparse
# from video_processor import VideoProcessor
# import torchvision.models as models
# import time
# import torchvision.transforms as transforms
# import torch
# import pickle
# from torch.autograd import Variable
# # parser=argparse.ArgumentParser(description="diff_pq")
# # parser.add_argument('--video_path',type=str)
# # parser.add_argument('--qp',type=int)
# # args=parser.parse_args()
# # qp=args.qp
#
# # video_path=args.video_path
#
# result = []
# model = Detector()
# times=[]
#
# frame_idx=0
# vgg_model_4096 = models.vgg16(pretrained=True)
# new_classifier = torch.nn.Sequential(*list(vgg_model_4096.children())[-1])
# vgg_model_4096.classifier = new_classifier
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# vgg_model_4096.to(device)
# print(device,"device")
#
#
#
# feature_dict={}
# qp_list=[10,15,20,25,30,35,40,45,50]
# for qp in qp_list:
#     #!!!!!!!!!!!!!!!!!!!!!!!!1每个新的dp都要更新啊！！！！！！！！！！！！
#     feature_list = []
#     video_path=f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset/video_test540_{qp}/video_test540_{qp}.mp4"
#     print(video_path)
#     with VideoProcessor(video_path,0) as video:
#         for frame in video:
#             tensor = transforms.ToTensor()(frame)
#             tensor = torch.unsqueeze(tensor, dim=0)
#             # tensor.to() 这个函数功能是产生一个新的tensor，并不会改变原数据。所以需要tensor=...  而模型可以直接moedl.to(device)
#             tensor = tensor.to(device)
#             feature = vgg_model_4096(Variable(tensor))
#             feature = feature.detach().cpu().numpy().tolist()
#             feature_list.append(feature[0])
#     feature_dict[qp]=feature_list
#
#
#
# #
# import pickle
# with open(f"features_dict_long.txt", "wb") as myprofile:
#     pickle.dump(feature_dict, myprofile)
#
#
# print("yes")







"""   kkkkkkkkkkkkkkkkkkkkk  """



from backend.server import Server
from backend.object_detector import Detector
from dds_utils import (Results, merge_boxes_in_results,Region)
from video_processor import VideoProcessor
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import pickle
from torch.autograd import Variable
import time as T
import time
# bitrate_list=[200,300,450,750,1200,1850,2850,4300,6000,8000]
print(time.strftime("%Y-%m-%d %H-%M-%S",time.localtime()))
T_begin=time.time()
bitrate_list=[10910693, 6050811, 3196561, 1627969, 825260, 450706, 273220, 180834]
for bitrate in bitrate_list:
    video_path=f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_profile/video_test540_{bitrate}/video_test540_{bitrate}_360p.mp4"


    result = []
    model = Detector()
    times=[]

    frame_idx=0
    with VideoProcessor(video_path,0) as video:
        for frame in video:
            starttime = T.time()
            result.append(model.infer(frame))
            # result.append([])
            endtime = T.time()
            times.append(float(endtime - starttime))
            # 从0帧 开始
            print("finish", frame_idx)
            frame_idx+=1

    print(len(result))
    final_results=Results()
    for frame_idx in range(0,3000):
        # print(self.l_results,self.result[frame_idx])
        for label, conf, (x, y, w, h) in result[frame_idx]:
            r = Region(frame_idx, x, y, w, h, conf, label,
                       0, origin="mpeg")
            # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
            final_results.append(r)
    result=merge_boxes_in_results(final_results.regions_dict, 0.3, 0.3)



    with open(f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_profile/constbit_dds_results540_{bitrate}_360p.txt", "wb") as myprofile:
        pickle.dump(result, myprofile)
print(time.time()-T_begin)
    # 1161.2982788085938
print("end 540 lr infer")

# for bitrate in bitrate_list:
#     video_path = f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_bit/video_test540_{bitrate}/video_test540_{bitrate}.mp4"
#     result = []
#     model = Detector()
#     times=[]
#
#
#
#     frame_idx=0
#
#
#     server=Server()
#     result,times_sr,times_infer=server.easy_perform_detection_edsr_on_video(video_path)
#
#     final_results=Results()
#     for frame_idx in range(0,3000):
#         # print(self.l_results,self.result[frame_idx])
#         for label, conf, (x, y, w, h) in result[frame_idx]:
#             r = Region(frame_idx, x, y, w, h, conf, label,
#                        0, origin="mpeg")
#             # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
#             final_results.append(r)
#     result=merge_boxes_in_results(final_results.regions_dict, 0.3, 0.3)
#
#
#
#     with open(f"constbit_dds_results540_{bitrate}_plusSR.txt", "wb") as myprofile:
#         pickle.dump(result, myprofile)
#
#
#
# print("end SR+infer---------------------------")



#
# #
# # ######################################################算不同bitrate feature
#
#
# result = []
# model = Detector()
# times=[]
#
# frame_idx=0
# vgg_model_4096 = models.vgg16(pretrained=True)
# new_classifier = torch.nn.Sequential(*list(vgg_model_4096.children())[-1])
# vgg_model_4096.classifier = new_classifier
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# vgg_model_4096.to(device)
# print(device,"device")
#
#
#
# feature_dict={}
# bitrate_list=[200,300,450,750,1200,1850,2850,4300,6000,8000]
# for bitrate in bitrate_list:
#     #!!!!!!!!!!!!!!!!!!!!!!!!1每个新的dp都要更新啊！！！！！！！！！！！！
#     feature_list = []
#     video_path=f"/home/ubuntu/VideoAnalytics/workplace/DRL_SR_Infer/VA/dataset_bit/video_test540_{bitrate}/video_test540_{bitrate}.mp4"
#     print(video_path)
#     with VideoProcessor(video_path,0) as video:
#         for frame in video:
#             tensor = transforms.ToTensor()(frame)
#             tensor = torch.unsqueeze(tensor, dim=0)
#             # tensor.to() 这个函数功能是产生一个新的tensor，并不会改变原数据。所以需要tensor=...  而模型可以直接moedl.to(device)
#             tensor = tensor.to(device)
#             feature = vgg_model_4096(Variable(tensor))
#             feature = feature.detach().cpu().numpy().tolist()
#             feature_list.append(feature[0])
#     feature_dict[bitrate]=feature_list
#
#
#
#
#
#
# with open(f"constbit_features_dict_long.txt", "wb") as myprofile:
#     pickle.dump(feature_dict, myprofile)
#
#
# print("end feature---------------------------------")