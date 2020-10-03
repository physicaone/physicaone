
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.utils.data
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import pandas as pd
import random
import powerlaw as pl
np.seterr(divide='ignore', invalid='ignore')
import pickle as pkl
import pandas as pd
import time
import progressbar
from tqdm import tqdm
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)
# random.seed(1)
# torch.backends.cudnn.enabled=False
# torch.backends.cudnn.deterministic=True

# global list0
# global listloss
from torchvision import datasets # load data

def get_H_k(x, y):
    list100=[]
    list100kmk=[]
    for i in range(len(x)):
        list100kmk.append(x[i]*y[i])
    for i in range(len(x)):
        for j in range(list100kmk[i]):
            list100.append(x[i])
    N=len(list100)
    H_k=0
    for i in range(len(x)):
        H_k-=(x[i]*y[i]/N)*np.log2(x[i]*y[i]/N)
    return H_k

def get_H_s(x, y):
    list100=[]
    list100kmk=[]
    for i in range(len(x)):
        list100kmk.append(x[i]*y[i])
    for i in range(len(x)):
        for j in range(list100kmk[i]):
            list100.append(x[i])
    N=len(list100)
    H_s=0
    for i in range(len(x)):
        H_s-=(x[i]*y[i]/N)*np.log2(x[i]/N)
    return H_s

    
def get_listmk(epoch1):
    config_count={} # 각 hidden layer state 갯수 파악 (k)
    for i in range(len(epoch1)):
        config_count[epoch1[i]]=0
    for i in range(len(epoch1)):
        config_count[epoch1[i]]+=1
        
    listk=[]
    for i in range(len(list(config_count.values()))):
        listk.append(int(list(config_count.values())[i]))
    listmk=[]
    kcount={}


    # 갯수의 갯수 파악 (m_k)
    for i in range(len(listk)):
        kcount[listk[i]]=0
    for i in range(len(listk)):
        kcount[listk[i]]+=1
    for i in range(len(kcount)):
        listmk.append(kcount[sorted(list(kcount))[i]])

    return sorted(list(kcount)), listmk
    
    
def vis2hid(v0, h0):
    for j in range(h_dim):
        de = np.inner(v0[n], w[:,j]) + b[j] #h0[j]=1와 h0[j]=0인 경우 RBM의 에너지 차이
        if(1./(1.+np.exp(-de)) > np.random.rand()):
            h0[n][j] = 1.
        else:
            h0[n][j] = -1.
        return h0
    
def hid2vis(h0, v1):
    for i in range(v_dim):
        de = np.inner(h0[n], w[i, :]) + a[i] #v1[i]=1와 v1[i]=0인 경우 RBM의 에너지 차이
        if(1./(1.+np.exp(-de)) > np.random.rand()):
            v1[n][i] = 1.
        else:
            v1[n][i] = -1.
        return v1    

def train_and_get_data(dataset, v_dim, h_dim, alpha, num_iter, filename):
#     bar = progressbar.ProgressBar()
    N=len(dataset)
    v0=dataset
    h0 = np.zeros((N, h_dim))
    v1 = np.zeros((N, v_dim))
    h1 = np.zeros((N, h_dim))    
    a = 0.5-np.random.rand(v_dim)
    b = 0.5-np.random.rand(h_dim)
    w = 0.5-np.random.rand(v_dim, h_dim)
    states_in_epoch=[]
    for l in tqdm(range(num_iter)):
        for n in range(N):
#             print(n)
            for j in range(h_dim):
                de = np.inner(v0[n], w[:,j]) + b[j] #h0[j]=1와 h0[j]=0인 경우 RBM의 에너지 차이
                if(1./(1.+np.exp(-de)) > np.random.rand()):
                    h0[n][j] = 1.
                else:
                    h0[n][j] = -1.

            for i in range(v_dim):
                de = np.inner(h0[n], w[i, :]) + a[i] #v1[i]=1와 v1[i]=0인 경우 RBM의 에너지 차이
                if(1./(1.+np.exp(-de)) > np.random.rand()):
                    v1[n][i] = 1.
                else:
                    v1[n][i] = -1.

            for j in range(h_dim):
                de = np.inner(v1[n], w[:,j]) + b[j] #h1[j]=1와 h1[j]=0인 경우 RBM의 에너지 차이
                if(1./(1.+np.exp(-de)) > np.random.rand()):
                    h1[n][j] = 1.
                else:
                    h1[n][j] = -1.
            if l==num_iter-1:
                states_in_epoch.append(str(h1[n].tolist()))
        #D_KL의 기울기 계산 
        da = np.mean(v0 - v1, axis = 0) #-dD_KL/da[i] = 1/N\sum_n v0[n][i] - v1[n][i]
        db = np.mean(h0 - h1, axis = 0) #-dD_KL/db[j] = 1/N \sum_n h0[n][j] - h1[n][j]
        dw = (np.matmul(v0.T, h0) - np.matmul(v1.T, h1))/N #-dD_KL/dw[i][j] = 1/N \sum_n v0[n][i]*h0[n][j] - v1[n][i]*h1[n][j]

        #RBM 파라미터들의 업데이트    
        a += alpha*da
        b += alpha*db
        w += alpha*dw

    data_x, data_y=get_listmk(states_in_epoch)



    with open('data/%s_listmkx_n_%s_n_hid=%s.pkl' %(str(datetime.today())[:10], filename, h_dim), 'wb') as f:
        pkl.dump(data_x, f)
    with open('data/%s_listmky_n_%s_n_hid=%s.pkl' %(str(datetime.today())[:10], filename, h_dim), 'wb') as f:
        pkl.dump(data_y, f)
    with open('data/%s_config_%s_n_hid=%s.pkl' %(str(datetime.today())[:10], filename, h_dim), 'wb') as f:
        pkl.dump(states_in_epoch, f)
#     with open('data/%s_loss_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, quan), 'wb') as f:
#         pkl.dump(listloss, f)
#     with open('data/%s_param_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, quan), 'wb') as f:
#         pkl.dump(listparam, f)

        
def list_to_param(listp, n_vis, n_hid):
    p1=[]
    p2=[]
    for i in range(n_vis*n_hid):
        if i%n_vis!=n_vis-1:
            p2.append(listp[i])
        else:
            p2.append(listp[i])
            p1.append(p2)
            p2=[]
    dummy=torch.zeros(n_hid, n_vis)
    for i in range(n_hid):
        for j in range(n_vis):
            dummy[i][j]=float(p1[i][j])
    return dummy

def get_mu_larger_than_1(x, y):
    list100=[]
    list100kmk=[]
    for i in range(len(x)):
        list100kmk.append(x[i]*y[i])
    for i in range(len(x)):
        for j in range(list100kmk[i]):
            list100.append(x[i])
    N=len(list100)
    mu=1+np.log(2)/(np.log2(N)-get_H_s(x,y))
    return mu

def get_mu_smaller_than_1(x, y):
    mu=1-1/(get_H_s(x,y)*np.log(2))
    return mu

def get_entropies(x,y):
    H_k=get_H_k(x,y)
    H_s=get_H_s(x,y)
    mus=get_mu_larger_than_1(x,y)
    mul=get_mu_smaller_than_1(x,y)
    print("H_k = %f, H_s = %f, mu = %f and %f" %(H_k, H_s, mus, mul))

def get_ccdf_y(x,y):
    listkmk=[]
    listkmk_cum=[]
    for i in range(len(y)):
        listkmk.append(x[i]*y[i])
    sum_listkmk=sum(listkmk)
    for i in range(len(listkmk)):
        listkmk_cum.append(sum(listkmk)/sum_listkmk)
        listkmk.pop(0)
    return listkmk_cum

def get_exponent(x,y):
    expo=-1-get_mu_larger_than_1(x,y)
    return expo

def get_state(x,y):
    list00=[]
    for i in range(len(x)):
        for j in range(y[i]):
            list00.append(x[i])

    return list00

def shuffle_list(list000):
    shuffled = sorted(list000, key=lambda k: random.random())
    return shuffled

def flatten_list(list0):
    flattened = [val for sublist in list0 for val in sublist]
    return flattened

def get_logbin(x,y):
    bins = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    list00=get_state(x,y)
    y1,x1,_ = plt.hist(list00, bins = bins, histtype='step', color='white')
    x1 = 0.5*(x1[1:]+x1[:-1])
    y1_=[]
    dummy=0
    for i in range(len(y1)):
        if y1[i]!=0.0:
            dummy+=1
    for i in range(dummy):
        y1_.append(y1[i]/((bins[i+1]-bins[i])))
    plt.close()

    return x1[0:dummy], y1_

def get_logbin_param(list00):
    bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    bins=bins/100000
#     list00=get_state(x,y)
    y1,x1,_ = plt.hist(list00, bins, histtype='step', color='white')
    x1 = 0.5*(x1[1:]+x1[:-1])
    y1_=[]
    dummy=0
#     for i in range(len(y1)):

#         if y1[i]!=0.0:
#             dummy+=1
#     for i in range(dummy):
#         y1_.append(y1[i]/((bins[i+1]-bins[i])))
    plt.close()

    return x1, y1

def get_MLE_exponent(x,y):
    fit1 = pl.Fit(get_state(x, y), discrete=True, xmin=0)
    print(fit1.power_law.alpha)

def get_MLE_exponent_mk_cut(x,y):
    x_bin, y_bin=get_logbin(x,y)
    cri1=0
    cri2=0
    for i in range(len(y_bin)):
        if y_bin[i]<1.:
            cri1=i
            break
    x_cri=x_bin[cri1-1]
    for j in range(len(x)):
        if x[j]>x_cri:
            cri2=j
            break
    x2=x[0:cri2-1]; y2=y[0:cri2-1]
    fit1 = pl.Fit(get_state(x2, y2), discrete=True, xmin=0)
    print(fit1.power_law.alpha)

    return x2,y2

