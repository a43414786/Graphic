import numpy as np
import pickle as pkl

with open('per_destPQ.pkl','rb') as f:
    per_destPQ = pkl.load(f)

with open('temp.pkl','rb') as f:
    temp = pkl.load(f)

with open('dest_PX.pkl','rb') as f:
    dest_PX = pkl.load(f)

temp_x = per_destPQ[:,:,:,[0]]
temp_y = per_destPQ[:,:,:,[1]]
temp = np.concatenate((temp_y,-temp_x),axis= 3)

print(np.sum((temp * per_destPQ),axis = 3))


temp_pos = temp
temp_neg = -temp
temp = temp * dest_PX
shape = temp.shape
temp = np.sum(temp,axis = 3)
temp = temp < 0
temp = np.reshape(temp,(shape[0],shape[1],shape[2],1))
temp = np.concatenate((temp,temp),axis=3)
rst = np.where(temp,temp_neg,temp_pos)

print(np.sum((rst * per_destPQ),axis = 3))
