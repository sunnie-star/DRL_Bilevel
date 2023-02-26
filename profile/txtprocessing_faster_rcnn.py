
import copy
f=open("faster_rcnn_param.txt","r",encoding="utf-8")
y_tensor_shape=[]
y_params=[]
y_name=[]
for line in f:
    line.strip()
    #不等于-1说明找到了
    if line.find("recursive")!=-1  or line.find("==")!=-1  or line.find("FasterRCNN ")!=-1:
        continue

    if line.find("[")!=-1 and line.find("]")!=-1:
        left=line.find("[")
        right=line.find("]")
        str_shape=copy.copy(line[left+1:right])
        list=[ int(i.strip()) for i in str_shape.split(",")]
        tensor_shape=1
        for i in list:
          tensor_shape=tensor_shape*i
        y_tensor_shape.append(tensor_shape)
        name = ""
        str = line[:line.find(":")].strip()
        for i in str:
            # 如果是字母或数字
            if i.isalnum():
                name += i
        y_name.append(name)
    #   没有参数，只有output，例如激活层
    if line.find("[")!=-1 and line.find("]")!=-1 and line.find('--')!=-1:
        y_params.append(0)
    #   有参数
    elif line.find("[")!=-1 and line.find("]")!=-1 and line.find('--')==-1:
        str_params=copy.copy((line[right+1:])).strip("\t\n, ").replace(",",'').replace("(","").replace(")","")
        params=int(str_params)
        y_params.append(params)


    # 验证没有问题
    # if line.find("[") != -1 and line.find("]") != -1:
    #     print(y_name[-1],y_tensor_shape[-1],y_params[-1],"+++++++++++")
    # print(line)


print(len(y_name),len(y_tensor_shape),len(y_params))
##########################################绘图
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
def cm_in(a):
    return a/2.54
songTi = matplotlib.font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc")
size = len(y_name)  # 39
x = np.arange(size)
print(x)

total_width, n = 0.85, 3
width = total_width / n
plt.figure(figsize=(cm_in(13.5),cm_in(10)))


 # #74C6C7  浅蓝
# #7C6781   紫色
#  #E08931  橙色
# #F66A6A  粉红
# #36738E  深蓝
plt.bar(x -width/2, [i*4/1024/1024 for i in y_params], width=width-0.05 , color='#F66A6A',label='每层参数个数')  #粉红

plt.bar(x+ width/2, [i*4/1024/1024 for i in y_tensor_shape] ,width=width-0.05,color='#36738E', label='输出tensor数据量')   # 深蓝

# plt.bar(x -width, dds1, width=width-0.05 , color='#E08931',label='dds1')  #  橙色
# plt.bar(x , dds2, width=width-0.05 , color='#F66A6A',label='dds2')  # 粉红
# plt.bar(x+ width, sr ,width=width-0.05 ,color='#36738E', label='sr')   # 深蓝
#设置图例并且设置图例的字体及大小
# plt.ylim(0.7,1)  #范围
# plt.xlim(-1,4)  #减少柱子距离
# plt.xticks(np.arange(4), ('γ=0.5\n\nqp=26',"γ=0.25\n\nqp=26",'γ=0.5\n\nqp=36',"γ=0.25\n\nqp=36"),fontsize=14)
plt.xticks(np.arange(size), y_name,style="italic",fontproperties="Times New Roman",rotation ='vertical')
plt.yticks()
# plt.tick_params(axis='both', labelsize=17)
plt.ylabel('参数量（MB）',fontproperties=songTi,fontsize=14,weight="bold")
# plt.xlabel('QP',fontproperties="Times New Roman",fontsize=14,weight="bold")
plt.grid(axis="y",ls="--")

# 图例
plt.legend(prop=songTi,fontsize=14)
# for i in range(3):    #  3组
#     plt.text([i]- 2 * width, cubic[i],'%.2f' % cubic[i],ha='center',fontsize=10)
#     plt.text(x[i]- width,    greedy[i], '%.2f' % greedy[i],ha='center', fontsize=10)
#     plt.text(x[i],          minerva[i], '%.2f' % minerva[i], ha='center', fontsize=10)
#     plt.text(x[i] + width , precache[i], '%.2f' % precache[i], ha='center', fontsize=10)
#     plt.text(x[i] + 2*width, vsim[i], '%.2f' % vsim[i], ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('txtprocessing_faster_rcnn.png',format="png", bbox_inches='tight') # 替换 plt.show()
# plt.savefig('Bindwidth.pdf',format="pdf", bbox_inches='tight') # 替换 plt.show()
plt.savefig("txtprocessing_faster_rcnn.svg", format="svg")
plt.show()