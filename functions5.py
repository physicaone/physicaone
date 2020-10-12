# from RBM import *
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
# from torchvision import datasets # load data
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
    
def energy(config):
    L = int(np.sqrt(config.size))
    config=np.reshape(config,(L,L))
    E = 0
    L = len(config)
    for i in range(L):
        for j in range(L):
            s = config[i,j]
            neigh = config[(i+1)%L, j] + config[i,(j+1)%L] + config[(i-1)%L,j] + config[i,(j-1)%L] 
            E += -neigh * s
    return E/2
    
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
#     dummy=np.zeros((n_hid, n_vis))
#     for i in range(n_hid):
#         for j in range(n_vis):
#             dummy[i][j]=p1[i][j]
#     return dummy
    return p1

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

def Ising_preprocessing(data):
    data=np.array(data).transpose()
    label_energy=data[1]
    data=data[3]
    data_=[]
    for i in range(len(data)):
        data_.append(flatten_list(data[i]))
    return np.array(data_)
    
    
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

class RBM(nn.Module):

    """
    Restricted Boltzmann Machine.

    Args:
        n_vis (int, optional): The size of visible layer. Defaults to 784.
        n_hid (int, optional): The size of hidden layer. Defaults to 128.
        k (int, optional): The number of Gibbs sampling. Defaults to 1.
    """

    def __init__(self, n_vis, n_hid, k):
        """Create a RBM."""
        super(RBM, self).__init__()
        self.v = nn.Parameter(np.round(torch.rand(1, n_vis)))
        self.h = nn.Parameter(np.round(torch.rand(1, n_hid)))
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))
        self.k = k

    def visible_to_hidden(self, v):
        """
        Conditional sampling a hidden variable given a visible variable.
        Args:
            v (Tensor): The visible variable.
        Returns:
            Tensor: The hidden variable.
        """
        v=torch.tensor(v)
        p = torch.tanh(F.linear(v, self.W, self.h))
#         tmp=p.bernoulli()
        return np.sign(p.detach().numpy())

    def hidden_to_visible(self, h):
        """
        Conditional sampling a visible variable given a hidden variable.
        Args:
            h (Tendor): The hidden variable.
        Returns:
            Tensor: The visible variable.
        """
        h=torch.tensor(h)
        p = torch.tanh(F.linear(h, self.W.t(), self.v))
#         p.bernoulli()
        return np.sign(p.detach().numpy())
        
    def free_energy(self, v):
        """
        Free energy function.
        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}

        Args:
            v (Tensor): The visible variable.

        Returns:
            FloatTensor: The free energy value.

        """
#         v=v.bernoulli()
        v=torch.tensor(v)
        v=np.sign(v.detach().numpy())
        v=torch.tensor(v)

        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)

        return torch.mean(-h_term - v_term)
        
    def forward(self, v):
        """
        Compute the real and generated examples.
        Args:
            v (Tensor): The visible variable.
        Returns:
            (Tensor, Tensor): The real and generagted variables.
        """
#         v=v.bernoulli()
        v=np.sign(v.detach().numpy())
        h = self.visible_to_hidden(v)

        for _ in range(self.k):
            v_gibb = self.hidden_to_visible(h)
            h = self.visible_to_hidden(v_gibb)
        list0.append(h)

        return v, v_gibb

def train(model, train_loader, n_epochs, lr, momentum):
    """
    Train a RBM model.
    Args:
        model: The model.
        train_loader (DataLoader): The data loader.
        n_epochs (int, optional): The number of epochs. Defaults to 20.
        lr (Float, optional): The learning rate. Defaults to 0.01.
    Returns:
        The trained model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # optimizer
    train_op = optim.SGD(model.parameters(), 0, 0)
    model.to(device)
    # train the RBM model
    model.train()
    
    E_origin=[]
    # sampling for not learnt model
    loss0=[]
    for _, (data) in enumerate(train_loader):
#         v, v_gibbs = model(data.view(-1, 784))
        v, v_gibbs = model(data.view(-1, 400))
        v=v.to(device)
        v_gibbs=v_gibbs.to(device)
        loss = model.free_energy(v) - model.free_energy(v_gibbs)

        loss0.append(loss.item())

        train_op.zero_grad()
        loss.backward()
        train_op.step()
    listloss.append(np.mean(loss0))
    
#     for param in model.parameters():
#         None
#     data1=list(torch.flatten(param.data))
#     listparam.append(data1)

    # optimizer
    train_op = optim.SGD(model.parameters(), lr, momentum)
#     train_op = optim.Adam(model.parameters(), lr)

    # train the RBM model
    model.train()
    sign_changed=0
#     E_origin_mean=np.mean(E_origin)
#     E_origin_std=np.std(E_origin)

    for epoch in range(n_epochs):
        loss_ = []
        v_generated=[]
        E_generated=[]
        for _, (data) in enumerate(train_loader):
            v, v_gibbs = model(data.view(-1, 400))
            v=v.to(device)
            v_gibbs=v_gibbs.to(device)
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()
            v_generated.append(v_gibbs)
        listloss.append(np.mean(loss_))
        for i in range(len(v_generated)):
            for j in range(len(v_generated[0])):
                E_generated.append(energy(v_generated[i][j]))
        print('Epoch %3.d | Loss=%6.4f | E_gen_mean=%6.6f | E_gen_std=%6.6f' % (epoch+1, np.mean(loss_), np.mean(E_generated), np.std(E_generated)))
        
#         # save parameters
#         for param in model.parameters():
#             None
#         data1=list(torch.flatten(param.data))
#         listparam.append(data1)

        if listloss[-1]*listloss[-2] <= 0:
            sign_changed+=1
            print('sign changed:%d'%(sign_changed))
        if sign_changed > 3 and np.abs(listloss[-1])<0.1:
            print("Earlystopping : Loss<0.1")
            break
    end_epoch=epoch

    return model, end_epoch, v_generated[0]



class CustomDataset(Dataset): 
    def __init__(self, dataset):
        data_x = dataset
        self.x_data = data_x
#         self.y_data = data_y

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)
    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
#         y = torch.FloatTensor([self.y_data[idx]])
        return x

def train_and_get_data_RBME(dataset, n_vis, n_hid, k, n_epochs, batch_size, lr, momentum, filename):
    # create a Restricted Boltzmann Machine

    global list0
    global listloss
    global listHs
    global listHk
    global listmkx
    global listmky
    global Emean
    list0=[]
    listloss=[]
    listmkx=[]
    listmky=[]
    listHs=[]
    listHk=[]

    model = RBM(n_vis, n_hid, k)


#     dataset = datasets.MNIST('../dataset',
#                          train=True,
#                          download=False,
#                          transform=transforms.Compose([transforms.ToTensor()]))
#     train_set, val_set = torch.utils.data.random_split(dataset, [quan, 60000-quan])
    label_energy=[]
    dataset1 = CustomDataset(dataset)
    datasetlist=list(dataset1)
    L=int(np.sqrt(n_vis))
    for i in range(len(datasetlist)):
        label_energy.append(energy(np.array(list_to_param(datasetlist[i], L, L))))
    print('E origin | mean : %f | std : %f' %(np.mean(label_energy), np.std(label_energy)))
    Emean = np.mean(label_energy)
              
    train_loader = DataLoader(dataset1, batch_size, shuffle=True)

#     train_loader = torch.utils.data.DataLoader(dataset,batch_size, drop_last=True)


    model, end_epoch, v_generated0 = train(model, train_loader, n_epochs=n_epochs, lr=lr, momentum=momentum)
    states_in_epoch=[]
    for e in range(end_epoch+1):
        for i in range(int(e*len(list0)/(end_epoch+1)), int((e+1)*len(list0)/(end_epoch+1))):
            for j in range(batch_size):
                states_in_epoch.append(str(list0[i][j].tolist()))
    print(end_epoch)
    
    for e in range(end_epoch+1):
        a, b = get_listmk(states_in_epoch[int(e*batch_size*len(list0)/(end_epoch+1)):int((e+1)*batch_size*len(list0)/(end_epoch+1))])
        listmkx.append(a)
        listmky.append(b)
        listHs.append(get_H_s(a,b))
        listHk.append(get_H_k(a,b))

    a, b=get_listmk(states_in_epoch)
    with open('data/%s_listmkx_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, filename), 'wb') as f:
        pkl.dump(listmkx, f)
    with open('data/%s_listmky_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, filename), 'wb') as f:
        pkl.dump(listmky, f)
    with open('data/%s_loss_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, filename), 'wb') as f:
        pkl.dump(listloss, f)
    with open('data/%s_listHs_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, filename), 'wb') as f:
        pkl.dump(listHs, f)
    with open('data/%s_listHk_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, filename), 'wb') as f:
        pkl.dump(listHk, f)
    with open('data/%s_generated_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, filename), 'wb') as f:
        pkl.dump(v_generated0, f)
        




def train_and_get_data_KLD(param, dataset, n_vis, n_hid, n_epochs, batch_size, lr, filename):
#     bar = progressbar.ProgressBar()
    v0=dataset
    N=len(v0)
    v0_batch = list_to_param(v0, batch_size, int(N/batch_size))
    h0 = np.zeros((batch_size, n_hid))
    v1 = np.zeros((batch_size, n_vis))
    h1 = np.zeros((batch_size, n_hid))
    if param==0:
        a = 0.5-np.random.rand(n_vis)
        b = 0.5-np.random.rand(n_hid)
        w = 0.5-np.random.rand(n_vis, n_hid)
    else:
        a, b, w = param[0], param[1], param[2]
    states_in_epoch=[]
    sample_energy=[]
    
    label_energy=[]
    for i in range(N):
        label_energy.append(energy(v0[i]))
    print('E origin | mean : %f | std : %f' %(np.mean(label_energy), np.std(label_energy)))
#     for l in tqdm(range(n_epochs)):
    for l in range(n_epochs):
        for batch in range(int(N/batch_size)):
            for n in range(batch_size):
                for j in range(n_hid):
                    de = np.inner(v0_batch[batch][n], w[:,j]) + b[j] #h0[j]=1와 h0[j]=0인 경우 RBM의 에너지 차이
                    if(1./(1.+np.exp(-de)) > np.random.rand()):
                        h0[n][j] = 1.
                    else:
                        h0[n][j] = -1.

                for i in range(n_vis):
                    de = np.inner(h0[n], w[i, :]) + a[i] #v1[i]=1와 v1[i]=0인 경우 RBM의 에너지 차이
                    if(1./(1.+np.exp(-de)) > np.random.rand()):
                        v1[n][i] = 1.
                    else:
                        v1[n][i] = -1.
                for j in range(n_hid):
                    de = np.inner(v1[n], w[:,j]) + b[j] #h1[j]=1와 h1[j]=0인 경우 RBM의 에너지 차이
                    if(1./(1.+np.exp(-de)) > np.random.rand()):
                        h1[n][j] = 1.
                    else:
                        h1[n][j] = -1.
                if l==n_epochs-1:
                    states_in_epoch.append(str(h1[n].tolist()))
            #D_KL의 기울기 계산 
            da = np.mean(v0_batch[batch] - v1, axis = 0) #-dD_KL/da[i] = 1/N\sum_n v0[n][i] - v1[n][i]
            db = np.mean(h0 - h1, axis = 0) #-dD_KL/db[j] = 1/N \sum_n h0[n][j] - h1[n][j]
            dw = (np.matmul(np.array(v0_batch[batch]).T, h0) - np.matmul(v1.T, h1))/N #-dD_KL/dw[i][j] = 1/N \sum_n v0[n][i]*h0[n][j] - v1[n][i]*h1[n][j]
            #RBM 파라미터들의 업데이트    
            a += lr*da
            b += lr*db
            w += lr*dw
            for i in range(len(v1)):
                sample_energy.append(energy(v1[i]))
        print('epoch:%d | E generated | mean : %f | std : %f' %(l+1, np.mean(sample_energy), np.std(sample_energy)))
        with open('data/%s_param_n_%s_n_hid=%s.pkl' %(str(datetime.today())[:10], filename, n_hid), 'wb') as f:
            pkl.dump([a, b, w], f)
#         if abs(np.mean(label_energy)-np.mean(sample_energy)) < abs(np.mean(label_energy)*0.1):
#             print('Earlystop')
#             break
        
        
    data_x, data_y=get_listmk(states_in_epoch)
    with open('data/%s_listmkx_n_%s_n_hid=%s.pkl' %(str(datetime.today())[:10], filename, n_hid), 'wb') as f:
        pkl.dump(data_x, f)
    with open('data/%s_listmky_n_%s_n_hid=%s.pkl' %(str(datetime.today())[:10], filename, n_hid), 'wb') as f:
        pkl.dump(data_y, f)
    with open('data/%s_config_%s_n_hid=%s.pkl' %(str(datetime.today())[:10], filename, n_hid), 'wb') as f:
        pkl.dump(states_in_epoch, f)
    with open('data/%s_generated_%s_n_hid=%s.pkl' %(str(datetime.today())[:10], filename, n_hid), 'wb') as f:
        pkl.dump(v1, f)
#     with open('data/%s_loss_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, quan), 'wb') as f:
#         pkl.dump(listloss, f)
#     with open('data/%s_param_n_hid=%s_%s.pkl' %(str(datetime.today())[:10], n_hid, quan), 'wb') as f:
#         pkl.dump(listparam, f)

