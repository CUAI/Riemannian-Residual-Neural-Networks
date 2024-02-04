import sys,os,random
sys.path.insert(0,'..')
import numpy as np

import torch
import torch.utils.data

import spd.nn as nn_spd
import spd.functional as functional_spd
import cplx.nn as nn_cplx

data_path_radar='./data/afew/'
classes=7

class DatasetAFEW(torch.utils.data.Dataset):
        def __init__(self, path, names):
            self._path = path
            self._names = names
        def __len__(self):
            return len(self._names)
        def __getitem__(self, item):
            x = np.load(self._path + self._names[item])[None, :, :].real
            x = torch.from_numpy(x).double()
            y = int(self._names[item].split('.')[0].split('_')[-1])
            y = torch.from_numpy(np.array(y)).long()
            return x,y
class DataLoaderAFEW:
    def __init__(self,data_path):
        path_train,path_test=data_path+'train/',data_path+'val/'
        for filenames in os.walk(path_train):
            names_train = sorted(filenames[2])
        for filenames in os.walk(path_test):
            names_test = sorted(filenames[2])
        N_train=len(names_train)
        N_test=len(names_test)
        train_set=DatasetAFEW(path_train,names_train)
        test_set=DatasetAFEW(path_test,names_test)
        self._train_generator=torch.utils.data.DataLoader(train_set,batch_size=N_train,shuffle='False')
        self._test_generator=torch.utils.data.DataLoader(test_set,batch_size=N_test,shuffle='False')

data_loader_radar=DataLoaderAFEW(data_path_radar)

print('Loading training and test data...')
train_data,train_labels=iter(data_loader_radar._train_generator).next()
test_data,test_labels=iter(data_loader_radar._test_generator).next()

print('Computing training Riemannian barycenters per class using Karcher flow...')
train_class_barycenters=[functional_spd.BaryGeom(train_data[train_labels==i]) for i in range(classes)]

print('Computing Riemannian distances of test data to training class barycenters...')
distances_to_train_class_barycenters=np.asarray(
    [functional_spd.dist_riemann(test_data,bary)[:,0].numpy() for bary in train_class_barycenters])

print('Classifying according to closest barycenter...')
decision=distances_to_train_class_barycenters.argmin(axis=0)
test_accuracy=(test_labels.numpy()==decision).sum()/test_labels.shape[0]

print('Test accuracy is: '+str(100*test_accuracy)+' %')

