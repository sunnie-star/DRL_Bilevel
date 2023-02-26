import matplotlib.pyplot as plt
import pickle
import numpy as np
# tuple = (time_trans, self.last_compute_time, time, self.bandwith[int(self.idx / 30 - 1) + N_CHUNK - 1],
#          self.qp_bitrate[self.last_qp], self.last_qp, f1, reward)

with open("reward_resnew_02-20_01-19_band4.txt", "rb") as file:
    reward_list = pickle.load(file)
f1_list=[]
for _, _, _, band, _, qp, f1, reward in reward_list:
    f1_list.append(f1)

y1=f1_list[int(450/30):int(600/30)]

y1.extend(f1_list[int(1050/30):int(1350/30)])
plt.plot(np.arange(len(y1)), y1, color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
         label='4_4')
x0=5
plt.plot([x0,x0],[0,1], 'k--', linewidth=1)




with open("reward_resnew_02-20_01-19_band7.txt", "rb") as file:
    reward_list = pickle.load(file)
f1_list=[]
for _, _, _, band, _, qp, f1, reward in reward_list:
    f1_list.append(f1)

y2=f1_list[int(450/30):int(600/30)]

with open("reward_resnew_02-20_01-19_band1.txt", "rb") as file:
    reward_list = pickle.load(file)
f1_list=[]
for _, _, _, band, _, qp, f1, reward in reward_list:
    f1_list.append(f1)
y2.extend(f1_list[int(1050/30):int(1350/30)])

plt.plot(np.arange(len(y2)), y2, color='#36738E', linewidth=1.0, linestyle='-', marker="o", markersize=2,
         label='7_1')

plt.xlabel('Chunk-id')
plt.ylabel('F1-score')
plt.legend()
plt.tight_layout()
plt.savefig(f"不同分配方式.png")
plt.show()

#
# # reward_resnew_02-20_01-19_45.txt
#
# with open("reward_resnew_02-20_01-19_45.txt","rb") as file:
#     reward_list=pickle.load(file)
# acc_45=[]
# for _, _, _, band, _, qp, f1, reward in reward_list:
#     acc_45.append(f1)
#
# acc_20=[]
# with open("reward_resnew_02-20_01-19_20.txt","rb") as file:
#     reward_list=pickle.load(file)
# for _, _, _, band, _, qp, f1, reward in reward_list:
#     acc_20.append(f1)
# diff=[]
# for i in range(len(acc_45)):
#     diff.append(acc_20[i]-acc_45[i])
#
# plt.figure(2)
# x = np.arange(len(acc_45) - 1)
# plt.plot(x, acc_45[1:], color='#EA4025', linewidth=1.0, linestyle='-', marker="o", markersize=2,
#          label='qp=45')
# plt.plot(x, acc_20[1:], color='#36738E', linewidth=2.0, linestyle='-', marker="s", markersize=2,
#          label="qp=20")
# plt.plot(x, diff[1:], color='#68B984', linewidth=2.0, linestyle='-', marker="s", markersize=2,
#          label="diff")
# for i in [1,499,599,749,899,1049,1349,2279,2699,2999]:
#     plt.plot([i/30, i/30], [0, 1], 'k--', linewidth=1)
#
# # 449
# # 599
# # 749
# # 899
# # 1049
# # 1349
# # 2279
# # 2699
# # 2999
#
# for i in range(0, 100, 10):
#     plt.plot([i, i], [0, 1], 'k--', linewidth=1)



# plt.xlabel('Chunk-id')
# plt.ylabel('F1-score')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"f1_diff_qp.png")
# plt.show()
