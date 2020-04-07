import os

# 人分的六类数据集
data_dir_human0 = '/home/liang/disambigular/testdata/0'
data_dir_human1 = '/home/liang/disambigular/testdata/1'
data_dir_human2 = '/home/liang/disambigular/testdata/2'
data_dir_human3 = '/home/liang/disambigular/testdata/3'
data_dir_human4 = '/home/liang/disambigular/testdata/4'
data_dir_human5 = '/home/liang/disambigular/testdata/5'
# 网络分的六类数据集
data_dir_net0 = '/home/liang/disambigular/net_vote_testdata/0'
data_dir_net1 = '/home/liang/disambigular/net_vote_testdata/1'
data_dir_net2 = '/home/liang/disambigular/net_vote_testdata/2'
data_dir_net3 = '/home/liang/disambigular/net_vote_testdata/3'
data_dir_net4 = '/home/liang/disambigular/net_vote_testdata/4'
data_dir_net5 = '/home/liang/disambigular/net_vote_testdata/5'
# 获取groundtruth下的图片路径名称
data_dir_human0_list = [img for img in os.listdir(data_dir_human0)]
data_dir_human1_list = [img for img in os.listdir(data_dir_human1)]
data_dir_human2_list = [img for img in os.listdir(data_dir_human2)]
data_dir_human3_list = [img for img in os.listdir(data_dir_human3)]
data_dir_human4_list = [img for img in os.listdir(data_dir_human4)]
data_dir_human5_list = [img for img in os.listdir(data_dir_human5)]
# 获得机器分的图片文件夹下的路径名称
data_dir_net0_list = [img for img in os.listdir(data_dir_net0)]
data_dir_net1_list = [img for img in os.listdir(data_dir_net1)]
data_dir_net2_list = [img for img in os.listdir(data_dir_net2)]
data_dir_net3_list = [img for img in os.listdir(data_dir_net3)]
data_dir_net4_list = [img for img in os.listdir(data_dir_net4)]
data_dir_net5_list = [img for img in os.listdir(data_dir_net5)]
'''
data_dir = 'H:\\TrainingData\\test\\011\\clo'
list = [img for img in os.listdir(data_dir)]
print(list)
'''
if __name__ == '__main__' :
    inter_list0 = [i for i in data_dir_human0_list if i in data_dir_net0_list]
    size_human0 = len(data_dir_human0_list);
    size_inter0 = len(inter_list0)
    acc0 = size_inter0 / size_human0
    print(acc0)

    inter_list1 = [i for i in data_dir_human1_list if i in data_dir_net1_list]
    size_human1 = len(data_dir_human1_list);
    size_inter1 = len(inter_list1)
    acc1 = size_inter1 / size_human1
    print(acc1)

    inter_list2 = [i for i in data_dir_human2_list if i in data_dir_net2_list]
    size_human2 = len(data_dir_human2_list);
    size_inter2 = len(inter_list2)
    acc2 = size_inter2 / size_human2
    print(acc2)

    inter_list3 = [i for i in data_dir_human3_list if i in data_dir_net3_list]
    size_human3 = len(data_dir_human3_list);
    size_inter3 = len(inter_list3)
    acc3 = size_inter3 / size_human3
    print(acc3)

    inter_list4 = [i for i in data_dir_human4_list if i in data_dir_net4_list]
    size_human4 = len(data_dir_human4_list);
    size_inter4 = len(inter_list4)
    acc4 = size_inter4 / size_human4
    print(acc4)

    inter_list5 = [i for i in data_dir_human5_list if i in data_dir_net5_list]
    size_human5 = len(data_dir_human5_list);
    size_inter5 = len(inter_list5)
    acc5 = size_inter5 / size_human5
    print(acc5)

    acc = (size_inter1 + size_inter2 + size_inter3 + size_inter4 + size_inter5 + size_inter0) / (
                size_human0 + size_human1 + size_human2 + size_human3 + size_human4 + size_human5)
    print(acc)