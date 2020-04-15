import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate


def normal_distribution(x , mean=1 , sigma=1) :
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


# mean1 , sigma1 = 0 , 1
# x1 = np.linspace(mean1 - 6 * sigma1 , mean1 + 6 * sigma1 , 100)
#
# mean2 , sigma2 = 0 , 2
# x2 = np.linspace(mean2 - 6 * sigma2 , mean2 + 6 * sigma2 , 100)
#
# mean3 , sigma3 = 5 , 1
# x3 = np.linspace(mean3 - 6 * sigma3 , mean3 + 6 * sigma3 , 100)
#
# y1 = normal_distribution(x1 , mean1 , sigma1)
# y2 = normal_distribution(x2 , mean2 , sigma2)
# y3 = normal_distribution(x3 , mean3 , sigma3)

inf_ = -float('inf')

# 积分
x = np.linspace(-6 , 6 , 100)
y = normal_distribution(x , 0 , 1)
yy = []
for i in x :
    v , err = integrate.quad(normal_distribution , inf_ , i , args=(0 , 1))  # args(n,m)   其中 n是均值，m是方差
    yy.append(v)
yy = np.array(yy)

yy1 = []
for i in x :
    v , err = integrate.quad(normal_distribution , inf_ , i , args=(0 , 1.8))
    yy1.append(v)
yy1 = np.array(yy1)

yy2 = []
for i in x :
    v , err = integrate.quad(normal_distribution , inf_ , i , args=(0 , 2.5))
    yy2.append(v)
yy2 = np.array(yy2)

# plt.plot((x+6)/12 , y , 'r' , label='classfier1')
plt.figure('Figure Object 1' ,  # 图形对象名称  窗口左上角显示
           figsize=(8 , 6) ,  # 窗口大小
           dpi=120 ,  # 分辨率
           facecolor='white' ,  # 背景色
           )
# 设置title
# plt.title('Function Curve', fontsize=14)
# 绘画
plt.plot((x + 6) / 12 , yy , 'r' , label='joint decision')
plt.plot((x + 6) / 12 , yy1 , 'g' , label='c1')
plt.plot((x + 6) / 12 , yy2 , 'b' , label='c2')
# 网格 grid参数
plt.grid(color='y' , linestyle=':' , linewidth=1)
plt.tick_params(labelsize=10)

ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
plt.legend(loc='upper left' , shadow=False , fontsize=12)
# plt.margins(0,0)
plt.show()

# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# np.random.seed(sum(map(ord,"aesthetics")))
# #首先定义一个函数用来画正弦函数，可帮助了解可以控制的不同风格参数
# def sinplot(flip=1):
#     x=np.linspace(0,14,100)
#     for i in range(1,7):
#         plt.plot(x,np.sin(x+i*0.5)*(7-i)*flip)
# sinplot()
# plt.show()
#
#
# # 添加了Seaborn模块
#
# np.random.seed(sum(map(ord,"aesthetics")))
# #首先定义一个函数用来画正弦函数，可帮助了解可以控制的不同风格参数
# def sinplot(flip=1):
#     x=np.linspace(0,14,100)
#     for i in range(1,7):
#         plt.plot(x,np.sin(x+i*0.5)*(7-i)*flip)
# #转换成Seaborn模块，只需要引入seaborn模块
# import seaborn as sns#添加Seaborn模块
# sinplot()
# plt.show()
