import matplotlib.pyplot as plt

x1=[136601879,
75756162,
40020945,
20382184,
10332262,
5642844,
3420722,
2264043
]     # size of different qp

y1=[0.916616214,
0.913624673,
0.90765957,
0.899560328,
0.881081141,
0.85387644,
0.821466782,
0.720675502
]     # 单位MB

x2=[130358179,
71089128,
37179688,
18918844,
9806297,
5587917,
3618230,
2559418,
]    # size of different bitrate
y2=[0.917557824,
0.911588153,
0.905578405,
0.893797678,
0.877285466,
0.852523986,
0.817348752,
0.772645155
]

x3=[127116000,
69551245,
36608717,
18808965,
9761424,
5584583,
3604177,
2564639
]    # size of 720p different bitrate
y3=[0.964091076,
0.94941274,
0.932817651,
0.91202384,
0.888194783,
0.856523093,
0.802742555,
0.730193555
]


x4=[119203054,
72376701,
38368526,
19472789,
10037363,
5668634,
3626363,
2574492
]   # size of 360p different bitrate
y4=[0.870733904,
0.86917256,
0.867948732,
0.862556567,
0.85523844,
0.837116397,
0.808928633,
0.776580906
]

x1=[i/10204/1024 for i in x1]
x2=[i/10204/1024 for i in x2]
x3=[i/10204/1024 for i in x3]
x4=[i/10204/1024 for i in x4]
print(x1,x2,x3,x4)
plt.plot(x1,y1,marker = "o",label="CQP of 540p")
plt.plot(x2,y2,marker = "o",label="ABR of 540p")
plt.plot(x3,y3,marker = "o",label="ABR of 720p")
plt.plot(x4,y4,marker = "o",label="ABR of 360p")

plt.xlabel("Video size(MB)")
plt.ylabel("Acc")
plt.legend()
plt.savefig("different_encode_param.png")
plt.savefig("different_encode_param.pdf")
plt.show()