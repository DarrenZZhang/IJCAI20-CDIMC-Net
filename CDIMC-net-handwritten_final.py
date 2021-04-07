# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:55:31 2019

@author: Administrator
"""
# This is the code for our IJCAI paper:
# J Wen, Z Zhang, Y Xu, B Zhang, L Fei, GS Xie,
# CDIMC-net: Cognitive Deep Incomplete Multi-view Clustering Network, IJCAI, 2020.
# Note 1: because of using the kmeans to reorder samples before training, the clustering performance is sensitive to the reoder to some extent.
# Note 2: Selecting suitable parameters 'learning rate' and 'lambda(gamma)' for pre-training and fine-tuning is important.
# If you find the code is useful, please cite our IJCAI paper.
# If you find any problems, please contact Jie Wen via jiewen_pr@126.com


from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from torch.nn import Linear
from sklearn.preprocessing import StandardScaler, MinMaxScaler,normalize
import scipy.io
#from utils import MnistDataset, cluster_acc,load_mnist
from idecutils import cluster_acc
import idecutils
import h5py
import csv
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
import os

import random
#os.environ["CUDA_VISIBLE_DEVICES"]="7"    
def wmse_loss(input, target, weight, reduction='mean'):
    ret = (torch.diag(weight).mm(target - input)) ** 2
    ret = torch.mean(ret)
    return ret
def get_kNNgraph2(data,K_num):
    # each row of data is a sample
    
    x_norm = np.reshape(np.sum(np.square(data), 1), [-1, 1])  # column vector
    x_norm2 = np.reshape(np.sum(np.square(data), 1), [1, -1])  # column vector
    dists = x_norm - 2 * np.matmul(data, np.transpose(data))+x_norm2
    num_sample = data.shape[0]
    graph = np.zeros((num_sample,num_sample),dtype = np.int)
    for i in range(num_sample):
        distance = dists[i,:]
        small_index = np.argsort(distance)
        graph[i,small_index[0:K_num]] = 1
    graph = graph-np.diag(np.diag(graph))
    resultgraph = np.maximum(graph,np.transpose(graph))
    return resultgraph
class AE(nn.Module):

    def __init__(self, n_stacks,n_input, n_z):
        super(AE, self).__init__()
        dims0 = []
        for idim in range(n_stacks-2):
            linshidim=round(n_input[0]*0.8)
            linshidim = int(linshidim)
            dims0.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims0.append(linshidim)
        
        dims1 = []
        for idim in range(n_stacks-2):
            linshidim=round(n_input[1]*0.8)
            linshidim = int(linshidim)
            dims1.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims1.append(linshidim)  
    
        dims2 = []
        for idim in range(n_stacks-2):
            linshidim=round(n_input[2]*0.8)
            linshidim = int(linshidim)
            dims2.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims2.append(linshidim)  
        
        dims3 = []
        for idim in range(n_stacks-2):
            linshidim=round(n_input[3]*0.8)
            linshidim = int(linshidim)
            dims3.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims3.append(linshidim)
    
        dims4 = []
        for idim in range(n_stacks-2):
            linshidim=round(n_input[4]*0.8)
            linshidim = int(linshidim)
            dims4.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims4.append(linshidim)
                
        # encoder0
        self.enc0_1 = Linear(n_input[0], dims0[0])
        self.enc0_2 = Linear(dims0[0], dims0[1])
        self.enc0_3 = Linear(dims0[1], dims0[2])
        self.z0_layer = Linear(dims0[2], n_z)
        # encoder1
        self.enc1_1 = Linear(n_input[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        self.enc1_3 = Linear(dims1[1], dims1[2])
        self.z1_layer = Linear(dims1[2], n_z)
        # encoder2
        self.enc2_1 = Linear(n_input[2], dims2[0])
        self.enc2_2 = Linear(dims2[0], dims2[1])
        self.enc2_3 = Linear(dims2[1], dims2[2])
        self.z2_layer = Linear(dims2[2], n_z)
        # encoder3
        self.enc3_1 = Linear(n_input[3], dims3[0])
        self.enc3_2 = Linear(dims3[0], dims3[1])
        self.enc3_3 = Linear(dims3[1], dims3[2])
        self.z3_layer = Linear(dims3[2], n_z)        
        # encoder4
        self.enc4_1 = Linear(n_input[4], dims4[0])
        self.enc4_2 = Linear(dims4[0], dims4[1])
        self.enc4_3 = Linear(dims4[1], dims4[2])
        self.z4_layer = Linear(dims4[2], n_z)  
   
        
        # decoder0
        self.dec0_0 = Linear(n_z, n_z)
        self.dec0_1 = Linear(n_z, dims0[2])
        self.dec0_2 = Linear(dims0[2], dims0[1])
        self.dec0_3 = Linear(dims0[1], dims0[0])
        self.x0_bar_layer = Linear(dims0[0], n_input[0])
        # decoder1
        self.dec1_0 = Linear(n_z, n_z)        
        self.dec1_1 = Linear(n_z, dims1[2])
        self.dec1_2 = Linear(dims1[2], dims1[1])
        self.dec1_3 = Linear(dims1[1], dims1[0])
        self.x1_bar_layer = Linear(dims1[0], n_input[1])
        # decoder2
        self.dec2_0 = Linear(n_z, n_z)           
        self.dec2_1 = Linear(n_z, dims2[2])
        self.dec2_2 = Linear(dims2[2], dims2[1])
        self.dec2_3 = Linear(dims2[1], dims2[0])
        self.x2_bar_layer = Linear(dims2[0], n_input[2])   
        # decoder3
        self.dec3_0 = Linear(n_z, n_z)   
        self.dec3_1 = Linear(n_z, dims3[2])
        self.dec3_2 = Linear(dims3[2], dims3[1])
        self.dec3_3 = Linear(dims3[1], dims3[0])
        self.x3_bar_layer = Linear(dims3[0], n_input[3])
        # decoder4
        self.dec4_0 = Linear(n_z, n_z)   
        self.dec4_1 = Linear(n_z, dims4[2])
        self.dec4_2 = Linear(dims4[2], dims4[1])
        self.dec4_3 = Linear(dims4[1], dims4[0])
        self.x4_bar_layer = Linear(dims4[0], n_input[4])            
        
    def forward(self, x0,x1,x2,x3,x4,we):
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)        
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3) 
        # encoder2
        enc2_h1 = F.relu(self.enc2_1(x2))
        enc2_h2 = F.relu(self.enc2_2(enc2_h1))
        enc2_h3 = F.relu(self.enc2_3(enc2_h2))
        z2 = self.z2_layer(enc2_h3) 
        # encoder3
        enc3_h1 = F.relu(self.enc3_1(x3))
        enc3_h2 = F.relu(self.enc3_2(enc3_h1))
        enc3_h3 = F.relu(self.enc3_3(enc3_h2))       
        z3 = self.z3_layer(enc3_h3) 
        # encoder4
        enc4_h1 = F.relu(self.enc4_1(x4))
        enc4_h2 = F.relu(self.enc4_2(enc4_h1))
        enc4_h3 = F.relu(self.enc4_3(enc4_h2))  
        z4 = self.z4_layer(enc4_h3)  
        
        summ = torch.diag(we[:,0]).mm(z0)+torch.diag(we[:,1]).mm(z1)+torch.diag(we[:,2]).mm(z2)+torch.diag(we[:,3]).mm(z3)+torch.diag(we[:,4]).mm(z4) 
        wei = 1/torch.sum(we,1)
        z = torch.diag(wei).mm(summ)
        
        # decoder0
        r0 = F.relu(self.dec0_0(z))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x0_bar = self.x0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z))       
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x1_bar = self.x1_bar_layer(dec1_h3)
        # decoder2
        r2 = F.relu(self.dec2_0(z))
        dec2_h1 = F.relu(self.dec2_1(r2))
        dec2_h2 = F.relu(self.dec2_2(dec2_h1))
        dec2_h3 = F.relu(self.dec2_3(dec2_h2))
        x2_bar = self.x2_bar_layer(dec2_h3)        
        # decoder3
        r3 = F.relu(self.dec3_0(z))
        dec3_h1 = F.relu(self.dec3_1(r3))
        dec3_h2 = F.relu(self.dec3_2(dec3_h1))
        dec3_h3 = F.relu(self.dec3_3(dec3_h2))
        x3_bar = self.x3_bar_layer(dec3_h3)         
        # decoder4
        r4 = F.relu(self.dec4_0(z))
        dec4_h1 = F.relu(self.dec4_1(r4))
        dec4_h2 = F.relu(self.dec4_2(dec4_h1))
        dec4_h3 = F.relu(self.dec4_3(dec4_h2))
        x4_bar = self.x4_bar_layer(dec4_h3)         
          
        return x0_bar,x1_bar,x2_bar,x3_bar,x4_bar,z,z0,z1,z2,z3,z4


class IDEC(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 n_clusters,
                 pretrain_path='data/ae_handwritten-5view.pkl'):
        super(IDEC, self).__init__()
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z)

    def pretrain(self, path=''):
        if args.pretrain_flag == 0:
            pretrain_ae(self.ae)
            print('pretrained ae finished')
            args.pretrain_flag = 1
        else:
            self.ae.load_state_dict(torch.load(self.pretrain_path))
            print('load pretrained ae model from',self.pretrain_path)            
            
    def update_label(self,x0,x1,x2,x3,x4,we,cluster_layer):
        _,_,_,_,_, z,_,_,_,_,_ = self.ae(x0,x1,x2,x3,x4,we)
        # kmeans cluster                
        x_norm = torch.reshape(torch.sum(torch.pow(z,2), 1), [-1, 1])  # column vector
        center_norm = torch.reshape(torch.sum(torch.pow(cluster_layer,2), 1), [1, -1])  # row vector
        dists = x_norm - 2 * torch.mm(z, torch.t(cluster_layer.type_as(z))) + center_norm.type_as(z)  # |x-y|^2 = |x|^2 -2*x*y^T + |y|^2
        labels = torch.argmin(dists, 1)
        losses = torch.min(dists, 1)
        return labels, losses.values
    def forward(self,x0,x1,x2,x3,x4,we,ypred,cluster_layer,sp_weight_sub):
        _,_,_,_,_, z,vz0,vz1,vz2,vz3,vz4 = self.ae(x0,x1,x2,x3,x4,we)              
        klloss = torch.mean(torch.diag(sp_weight_sub).mm(torch.pow(z-cluster_layer[ypred.cpu().numpy().tolist()].type_as(z),2)))
        klloss = klloss*len(sp_weight_sub)/sum(sp_weight_sub)
        return z,klloss,vz0,vz1,vz2,vz3,vz4


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    print(model)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)     

    optimizer = SGD(model.parameters(), lr=args.lrae, momentum=args.momentumae)
#    model.train()
    index_array = np.arange(X0.shape[0])
    if args.AE_shuffle==True:
        np.random.shuffle(index_array)
    for epoch in range(args.aeproches):  
        total_loss = 0.
        for batch_idx in range(np.int(np.ceil(X0.shape[0]/args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx+1) * args.batch_size, X0.shape[0])]    
            x0 = X0[idx].to(device)
            x1 = X1[idx].to(device)
            x2 = X2[idx].to(device)
            x3 = X3[idx].to(device)
            x4 = X4[idx].to(device)
            we = WE[idx].to(device)
            affi_graph0 = torch.Tensor(np.copy(pre_affi_graph0[idx,:][:,idx])).to(device)            
            affi_graph1 = torch.Tensor(np.copy(pre_affi_graph1[idx,:][:,idx])).to(device)            
            affi_graph2 = torch.Tensor(np.copy(pre_affi_graph2[idx,:][:,idx])).to(device)            
            affi_graph3 = torch.Tensor(np.copy(pre_affi_graph3[idx,:][:,idx])).to(device)
            affi_graph4 = torch.Tensor(np.copy(pre_affi_graph4[idx,:][:,idx])).to(device)
            
            affi_graph0 = 0.5*(affi_graph0+affi_graph0.t())
            Lap_graph0  = torch.diag(affi_graph0.sum(1))-affi_graph0
            affi_graph1 = 0.5*(affi_graph1+affi_graph1.t())
            Lap_graph1  = torch.diag(affi_graph1.sum(1))-affi_graph1  
            affi_graph2 = 0.5*(affi_graph2+affi_graph2.t())
            Lap_graph2  = torch.diag(affi_graph2.sum(1))-affi_graph2
            affi_graph3 = 0.5*(affi_graph3+affi_graph3.t())
            Lap_graph3  = torch.diag(affi_graph3.sum(1))-affi_graph3
            affi_graph4 = 0.5*(affi_graph4+affi_graph4.t())
            Lap_graph4  = torch.diag(affi_graph4.sum(1))-affi_graph4
              
            optimizer.zero_grad()
            x0_bar,x1_bar,x2_bar,x3_bar,x4_bar,hidden,vz0,vz1,vz2,vz3,vz4 = model(x0,x1,x2,x3,x4,we)
            graph_loss = (1/5)*(torch.trace(vz0.t().mm(Lap_graph0).mm(vz0))+torch.trace(vz1.t().mm(Lap_graph1).mm(vz1))+torch.trace(vz2.t().mm(Lap_graph2).mm(vz2))+torch.trace(vz3.t().mm(Lap_graph3).mm(vz3))+torch.trace(vz4.t().mm(Lap_graph4).mm(vz4)))/len(idx)
            loss = wmse_loss(x0_bar,x0,we[:,0])+wmse_loss(x1_bar,x1,we[:,1])+wmse_loss(x2_bar,x2,we[:,2])+wmse_loss(x3_bar,x3,we[:,3])+wmse_loss(x4_bar,x4,we[:,4])
            fusion_loss = loss+args.gammaae*graph_loss
            total_loss += fusion_loss.item()
            fusion_loss.backward()
            optimizer.step()
        print("ae_epoch {} loss={:.8f} ".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train_idec():
      
    model = IDEC(
        n_stacks = 4,    
        n_input=args.n_input,
        n_z = args.n_clusters,
        n_clusters=args.n_clusters,
        pretrain_path=args.pretrain_path).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0) 
            
    model.pretrain()
    optimizer = Adam(model.parameters(), lr=args.lrkl)
    # cluster parameter initiate
    _,_,_,_,_,hidden,_,_,_,_,_ = model.ae(X0,X1,X2,X3,X4,WE)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
#    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    hidden_np = hidden.data.cpu().numpy()
    hidden_np = np.nan_to_num(hidden_np)
    y_pred = kmeans.fit_predict(hidden_np)
    del hidden_np
    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    y_pred_last = np.copy(y_pred)
    cluster_layer = torch.tensor(kmeans.cluster_centers_).to(device)

#    model.train()
    best_acc2 = 0
    best_epoch = 0
    total_loss_KL = 0
    
    sample_weight = torch.ones(X0.shape[0])
    sample_weight[y_pred == -1] = 0  # do not use the noisy examples     
    clustering_loss = 0
    for epoch in range(int(args.maxiter)):
        if epoch > 0:
            y_pred = y_pred.cpu().numpy()
        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)
        ari = ari_score(y, y_pred)
        if acc>best_acc2:
            best_acc2 = np.copy(acc)
            best_epoch = epoch
        print('best_Iter {}'.format(best_epoch), ':best_Acc2 {:.4f}'.format(best_acc2),'Iter {}'.format(epoch),':Acc {:.4f}'.format(acc),
              ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),'total_loss_KL {:.8f}'.format(clustering_loss))

        # check stop criterion
        delta_y = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if epoch > 80 and delta_y < args.tol:
            print('Training stopped: epoch=%d, delta_label=%.4f, tol=%.4f' % (epoch, delta_y, args.tol))
            break
        y_pred = torch.tensor(y_pred)
        """ Step 1: train the network """
        index_array = np.arange(X0.shape[0])
        if args.AE_shuffle==True:
            np.random.shuffle(index_array) 
        for KL_epoach in range(args.maxKL_epoach):
            total_loss_KL = 0          
            for batch_idx in range(np.int(np.ceil(X0.shape[0]/args.batch_size))):
                idx = index_array[batch_idx * args.batch_size: min((batch_idx+1) * args.batch_size,X0.shape[0])]    
                x0 = X0[idx].to(device)
                x1 = X1[idx].to(device)
                x2 = X2[idx].to(device)
                x3 = X3[idx].to(device)
                x4 = X4[idx].to(device)
                we = WE[idx].to(device)
                y_pred_sub = y_pred[idx].to(device)
                sample_weight_sub = sample_weight[idx].to(device)
                affi_graph0 = torch.Tensor(np.copy(pre_affi_graph0[idx,:][:,idx])).to(device)            
                affi_graph1 = torch.Tensor(np.copy(pre_affi_graph1[idx,:][:,idx])).to(device)            
                affi_graph2 = torch.Tensor(np.copy(pre_affi_graph2[idx,:][:,idx])).to(device)            
                affi_graph3 = torch.Tensor(np.copy(pre_affi_graph3[idx,:][:,idx])).to(device)
                affi_graph4 = torch.Tensor(np.copy(pre_affi_graph4[idx,:][:,idx])).to(device)
                affi_graph0 = 0.5*(affi_graph0+affi_graph0.t())
                Lap_graph0  = torch.diag(affi_graph0.sum(1))-affi_graph0
                affi_graph1 = 0.5*(affi_graph1+affi_graph1.t())
                Lap_graph1  = torch.diag(affi_graph1.sum(1))-affi_graph1  
                affi_graph2 = 0.5*(affi_graph2+affi_graph2.t())
                Lap_graph2  = torch.diag(affi_graph2.sum(1))-affi_graph2
                affi_graph3 = 0.5*(affi_graph3+affi_graph3.t())
                Lap_graph3  = torch.diag(affi_graph3.sum(1))-affi_graph3
                affi_graph4 = 0.5*(affi_graph4+affi_graph4.t())
                Lap_graph4  = torch.diag(affi_graph4.sum(1))-affi_graph4   
               
                optimizer.zero_grad()
                hidden,kl_loss,vz0,vz1,vz2,vz3,vz4 = model(x0,x1,x2,x3,x4,we,y_pred_sub,cluster_layer,sample_weight_sub)   
                if np.isnan(hidden.data.cpu().numpy()).any():
                    break 
                graph_loss = (1/5)*(torch.trace(vz0.t().mm(Lap_graph0).mm(vz0))+torch.trace(vz1.t().mm(Lap_graph1).mm(vz1))+torch.trace(vz2.t().mm(Lap_graph2).mm(vz2))+torch.trace(vz3.t().mm(Lap_graph3).mm(vz3))+torch.trace(vz4.t().mm(Lap_graph4).mm(vz4)))/len(idx)
                fusion_loss = kl_loss+args.gammakl*graph_loss                  
                total_loss_KL +=fusion_loss          
                fusion_loss.backward()
                optimizer.step()
            if np.isnan(hidden.data.cpu().numpy()).any():
                total_loss_KL=0
                break
            else:
                total_loss_KL =  total_loss_KL.item()  / (batch_idx + 1)
            
        """ Step 2: update labels """
        y_pred, prelosses = model.update_label(X0,X1,X2,X3,X4,WE,cluster_layer)     
        
        clustering_loss = torch.sum(prelosses)/len(prelosses)
        """ Step 3: Compute sample weights """
        lam = clustering_loss + epoch*torch.std(prelosses) / args.maxiter
        sample_weight = torch.where(prelosses < lam, torch.full_like(prelosses,1), torch.full_like(prelosses,0))
       
       
    y_pred,_ = model.update_label(X0,X1,X2,X3,X4,WE,cluster_layer)
    y_pred = y_pred.cpu().numpy()
    return y_pred 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='handwritten-5view')
    parser.add_argument('--basis_pretrain_path', type=str, default='data/handwritten-5view')
    parser.add_argument('--percentDel', type=int, default=3)
    parser.add_argument('--AE_shuffle', type=bool, default=False)    
    parser.add_argument('--tol', default=1e-7, type=float)
    parser.add_argument('--basis_save_dir', type=str, default='data/handwritten-5view')
    acc_ite = np.zeros(5)
    nmi_ite = np.zeros(5)
    ari_ite = np.zeros(5)
    pur_ite = np.zeros(5)
    pre_ite = np.zeros(5)
    rec_ite = np.zeros(5)
    Fscore_ite = np.zeros(5)
    args = parser.parse_args()


    ff = 2
    args.save_dir = args.basis_save_dir+'_0_'+str(args.percentDel)+'_ff_'+str(ff)
    best_acc = 0
    best_nmi = 0
                                 
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")  
    data = scipy.io.loadmat('data/'+args.dataset+'.mat')
    foldss = scipy.io.loadmat('data/'+args.dataset+'_percentDel_0.'+str(args.percentDel)+'.mat')    
    label = data['Y']
    label = label.reshape(-1)
    label = np.array(label,'float64')
    X = data['X']
    args.n_clusters = len(np.unique(label))
    y = label
    del label,data
    folds = foldss['folds']
    WE = folds[0,ff]
    del folds
    
    X0 = np.array(X[0,0],'float64')     #240 76 216 47 64 features~
    X1 = np.array(X[0,1],'float64')
    X2 = np.array(X[0,2],'float64')
    X3 = np.array(X[0,3],'float64')
    X4 = np.array(X[0,4],'float64') 
    del X
        
    iv = 0
    WEiv = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)
    linshi_XN0 = np.copy(X0)    
    linshi_XN0[ind_0,:] = np.mean(linshi_XN0[ind_1,:],axis=0)
    linshi_XN0 = normalize(linshi_XN0)   
    X0[ind_1,:] = StandardScaler().fit_transform(X0[ind_1,:])
    X0[ind_0,:] = 0
    
    iv = 1
    WEiv = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)
    linshi_XN1 = np.copy(X1)    
    linshi_XN1[ind_0,:] = np.mean(linshi_XN1[ind_1,:],axis=0)
    linshi_XN1 = normalize(linshi_XN1)     
    X1[ind_1,:] = StandardScaler().fit_transform(X1[ind_1,:])
    X1[ind_0,:] = 0
    
    iv=2
    WEiv = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)
    linshi_XN2 = np.copy(X2)    
    linshi_XN2[ind_0,:] = np.mean(linshi_XN2[ind_1,:],axis=0)
    linshi_XN2 = normalize(linshi_XN2)       
    X2[ind_1,:] = StandardScaler().fit_transform(X2[ind_1,:])
    X2[ind_0,:] = 0
    
    iv=3
    WEiv = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1) 
    linshi_XN3 = np.copy(X3)    
    linshi_XN3[ind_0,:] = np.mean(linshi_XN3[ind_1,:],axis=0)
    linshi_XN3 = normalize(linshi_XN3)      
    X3[ind_1,:] = StandardScaler().fit_transform(X3[ind_1,:])
    X3[ind_0,:] = 0
    
    iv=4
    WEiv = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)  
    linshi_XN4 = np.copy(X4)    
    linshi_XN4[ind_0,:] = np.mean(linshi_XN4[ind_1,:],axis=0)
    linshi_XN4 = normalize(linshi_XN4)      
    X4[ind_1,:] = StandardScaler().fit_transform(X4[ind_1,:])
    X4[ind_0,:] = 0    
    del iv,ind_1,ind_0,WEiv
                                               
    X0 = np.nan_to_num(X0)
    X1 = np.nan_to_num(X1)
    X2 = np.nan_to_num(X2)
    X3 = np.nan_to_num(X3)
    X4 = np.nan_to_num(X4)
           
    X_total = np.concatenate((linshi_XN0,linshi_XN1,linshi_XN2,linshi_XN3,linshi_XN4),axis=1)
    np.random.seed(20)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20,random_state=20)
    

#    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(X_total)    
    del X_total,kmeans
    
    
    X0_train = np.zeros(X0.shape)
    X1_train = np.zeros(X1.shape)
    X2_train = np.zeros(X2.shape)
    X3_train = np.zeros(X3.shape)
    X4_train = np.zeros(X4.shape)

    
    linshi_XN0_train = np.zeros(X0.shape)
    linshi_XN1_train = np.zeros(X1.shape)
    linshi_XN2_train = np.zeros(X2.shape)
    linshi_XN3_train = np.zeros(X3.shape)
    linshi_XN4_train = np.zeros(X4.shape)

        
    label_train = np.zeros(y.shape)
    WE_train = np.zeros(WE.shape)
    basis_index = 0
    for li in range(args.n_clusters):
        index_li = np.where(y_pred==li)
        index_li = (np.array(index_li)).reshape(-1)
        X0_train[np.arange(len(index_li))+basis_index,:] = np.copy(X0[index_li])
        X1_train[np.arange(len(index_li))+basis_index,:] = np.copy(X1[index_li])
        X2_train[np.arange(len(index_li))+basis_index,:] = np.copy(X2[index_li])
        X3_train[np.arange(len(index_li))+basis_index,:] = np.copy(X3[index_li])
        X4_train[np.arange(len(index_li))+basis_index,:] = np.copy(X4[index_li])
        
        label_train[np.arange(len(index_li))+basis_index] = np.copy(y[index_li])
        WE_train[np.arange(len(index_li))+basis_index,:] = np.copy(WE[index_li,:])
        linshi_XN0_train[np.arange(len(index_li))+basis_index,:] = np.copy(linshi_XN0[index_li])
        linshi_XN1_train[np.arange(len(index_li))+basis_index,:] = np.copy(linshi_XN1[index_li])
        linshi_XN2_train[np.arange(len(index_li))+basis_index,:] = np.copy(linshi_XN2[index_li])
        linshi_XN3_train[np.arange(len(index_li))+basis_index,:] = np.copy(linshi_XN3[index_li])
        linshi_XN4_train[np.arange(len(index_li))+basis_index,:] = np.copy(linshi_XN4[index_li])
        basis_index = basis_index + len(index_li)
    
    del X0,X1,X2,X3,X4,WE,y,linshi_XN0,linshi_XN1,linshi_XN2,linshi_XN3,linshi_XN4
    X0 = np.copy(X0_train)
    X1 = np.copy(X1_train)
    X2 = np.copy(X2_train)
    X3 = np.copy(X3_train)
    X4 = np.copy(X4_train)
    
    WE = np.copy(WE_train)
    y = label_train
    del X0_train,X1_train,X2_train,X3_train,X4_train,WE_train,label_train,basis_index,index_li    
    
    
    iv = 0
    WEiv  = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    X_kc = np.copy(linshi_XN0_train[ind_1,:])                       
    pre_affi_graph0 = get_kNNgraph2(X_kc,K_num=11)    
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)
    index_matrix = np.diag(WEiv)
    index_matrix = np.delete(index_matrix,ind_0,axis=1)  #n*nv
    pre_affi_graph0 = np.matmul(np.matmul(index_matrix,pre_affi_graph0),np.transpose(index_matrix))

    iv = 1
    WEiv  = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    X_kc = np.copy(linshi_XN1_train[ind_1,:])                       
    pre_affi_graph1 = get_kNNgraph2(X_kc,K_num=11)    
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)
    index_matrix = np.diag(WEiv)
    index_matrix = np.delete(index_matrix,ind_0,axis=1)  #n*nv
    pre_affi_graph1 = np.matmul(np.matmul(index_matrix,pre_affi_graph1),np.transpose(index_matrix))
    
    iv=2
    WEiv  = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    X_kc = np.copy(linshi_XN2_train[ind_1,:])                       
    pre_affi_graph2 = get_kNNgraph2(X_kc,K_num=11)    
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)
    index_matrix = np.diag(WEiv)
    index_matrix = np.delete(index_matrix,ind_0,axis=1)  #n*nv
    pre_affi_graph2 = np.matmul(np.matmul(index_matrix,pre_affi_graph2),np.transpose(index_matrix))   

    iv = 3
    WEiv  = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    X_kc = np.copy(linshi_XN3_train[ind_1,:])                       
    pre_affi_graph3 = get_kNNgraph2(X_kc,K_num=11)    
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)
    index_matrix = np.diag(WEiv)
    index_matrix = np.delete(index_matrix,ind_0,axis=1)  #n*nv
    pre_affi_graph3 = np.matmul(np.matmul(index_matrix,pre_affi_graph3),np.transpose(index_matrix))    

    iv = 4
    WEiv  = np.copy(WE[:,iv])
    ind_1 = np.where(WEiv==1)
    ind_1 = (np.array(ind_1)).reshape(-1)
    X_kc = np.copy(linshi_XN4_train[ind_1,:])                       
    pre_affi_graph4 = get_kNNgraph2(X_kc,K_num=11)    
    ind_0 = np.where(WEiv==0)
    ind_0 = (np.array(ind_0)).reshape(-1)
    index_matrix = np.diag(WEiv)
    index_matrix = np.delete(index_matrix,ind_0,axis=1)  #n*nv
    pre_affi_graph4 = np.matmul(np.matmul(index_matrix,pre_affi_graph4),np.transpose(index_matrix))    
            
    del ind_1,ind_0,X_kc,index_matrix,linshi_XN0_train,linshi_XN1_train,linshi_XN2_train,linshi_XN3_train,linshi_XN4_train
        
    X0 = torch.Tensor(X0).to(device)
    X1 = torch.Tensor(X1).to(device)
    X2 = torch.Tensor(X2).to(device)
    X3 = torch.Tensor(X3).to(device)
    X4 = torch.Tensor(X4).to(device)
    WE = torch.Tensor(WE).to(device)
        
    args.n_input = [X0.shape[1],X1.shape[1],X2.shape[1],X3.shape[1],X4.shape[1]]
    
    args.lrae = 0.01
    args.momentumae = 0.95
    args.gammaae = 0.001
    args.lrkl = 0.0001
    args.gammakl = 0.001
    args.maxKL_epoach = 7    
    args.maxiter = 100
    args.aeproches = 500   
    args.pretrain_flag = 1
    args.pretrain_path = args.basis_pretrain_path+'_'+str(ff)+'_0.'+str(args.percentDel)+'_aelr_'+str(args.lrae)+'_aeproches_'+str(args.aeproches)+'_pretrained_model'+'.pkl'

#    args.pretrain_flag = 0
#    args.pretrain_path = args.basis_pretrain_path+'_'+str(ff)+'_aelr_'+str(args.lrae)+'_aeproches_'+str(args.aeproches)+'.pkl'
    # SGD parameter

    print(args)
    y_pred  = train_idec()
    
    acc_ite_lin = cluster_acc(y, y_pred)*100
    nmi_ite_lin = nmi_score(y, y_pred)*100
    pur_ite_lin = idecutils.purity_score(y,y_pred)*100

                                  
        

