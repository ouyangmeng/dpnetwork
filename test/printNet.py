###################################################################################################
# 输出网络
###################################################################################################
import ann

Network = open("MyNetWork", 'w')
Network.write(str(ann.inp_num))
Network.write('\n')
Network.write(str(ann.hid_num))
Network.write('\n')
Network.write(str(ann.out_num))
Network.write('\n')
for i in ann.w1:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
    Network.write('\n')
Network.write('\n')

for i in ann.w2:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
Network.write('\n')
Network.close()