# -*- coding: utf-8 -*-
import datetime
import math
import numpy as np
import scipy.io as sio
#程序开始时间
starttime = datetime.datetime.now()


# 读入数据
################################################################################################
filename = 'mnist_train.mat'    #训练样本
sample = sio.loadmat(filename)
sample = sample["mnist_train"]
#print(sample[0]) #第一个训练图像的像素矩阵28*28，从上到下，从左到右
sample /= 256.0       # 特征向量归一化，输入取值0~255，归一为0~1


filename = 'mnist_train_labels.mat'   #训练样本标签
label = sio.loadmat(filename)
label = label["mnist_train_labels"]

##################################################################################################


# 神经网络配置
##################################################################################################
samp_num = len(sample)      # 样本总数，这里60000
inp_num = len(sample[0])    # 输入层节点数，这里28*28=784
out_num = 10                # 输出节点数
hid_num = 10  # 隐层节点数(经验公式)
w1 = 0.2*np.random.random((inp_num, hid_num))- 0.1   # 初始化输入层权矩阵:hid_num个权值数组，每个数组的大小是784
w2 = 0.2*np.random.random((hid_num, out_num))- 0.1   # 初始化隐层权矩阵
hid_offset = np.zeros(hid_num)     # 隐层偏置向量
out_offset = np.zeros(out_num)     # 输出层偏置向量
inp_lrate = 0.3             # 输入层权值学习率
hid_lrate = 0.3             # 隐层学权值习率
err_th = 0.01                # 学习误差门限

###################################################################################################


# 必要函数定义
###################################################################################################
def get_act(x):
    act_vec = []
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))  #激活函数
    act_vec = np.array(act_vec)
    return act_vec

def get_err(e):
    return 0.5*np.dot(e,e) #矩阵的点积

###################################################################################################


# 训练——可使用err_th与get_err() 配合，提前结束训练过程
###################################################################################################
for count in range(0, samp_num):
    #print(count)
    t_label = np.zeros(out_num)
    t_label[label[count]] = 1
    #前向过程
    hid_value = np.dot(sample[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值

    #后向过程
    e = t_label - out_act                          # 输出值与真值间的误差
    out_delta = e * out_act * (1-out_act)                                       # 输出层delta计算
    hid_delta = hid_act * (1-hid_act) * np.dot(w2, out_delta)                   # 隐层delta计算
    for i in range(0, out_num):
        w2[:,i] += hid_lrate * out_delta[i] * hid_act   # 更新隐层到输出层权向量
    for i in range(0, hid_num):
        w1[:,i] += inp_lrate * hid_delta[i] * sample[count]      # 更新输出层到隐层的权向量

    out_offset += hid_lrate * out_delta                             # 输出层偏置更新
    hid_offset += inp_lrate * hid_delta

###################################################################################################


# 测试网络
###################################################################################################
filename = 'mnist_test.mat'
test = sio.loadmat(filename)
test_s = test["mnist_test"]
test_s /= 256.0

filename = 'mnist_test_labels.mat'
testlabel = sio.loadmat(filename)
test_l = testlabel["mnist_test_labels"]
right = np.zeros(10)
numbers = np.zeros(10)
################################ 读入测试数据结束
# 统计测试数据中各个数字的数目
for i in test_l:
    numbers[i] += 1

for count in range(len(test_s)):
    hid_value = np.dot(test_s[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    if np.argmax(out_act) == test_l[count]:
        right[test_l[count]] += 1

############################################################################################


#结果输出
##############################################################################################
endtime = datetime.datetime.now()
print("运行时间：",end='')
print((endtime - starttime).seconds,end='')
print("s")
print("*****************0 , 1, 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9*****************")
print("测试集合（个数）:", end='')
print(numbers)
print("正确结果（个数）:", end='')
print(right)

result = right/numbers
sum = right.sum()
sum/=len(test_s)
print("单个数字识别正确率:", end='')
for i in result:
    print("%.2f" % i,end='')
    print(" , ", end='')
print()
print("总体数字识别正确率:", end='')
print(sum)
print("**********************************************************************")

##############################end######################################################