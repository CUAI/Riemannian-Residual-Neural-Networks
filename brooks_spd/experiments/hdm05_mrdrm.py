import sys,os,random
sys.path.insert(0,'..')
import numpy as np

import torch
import torch.utils.data

import spd.nn as nn_spd
import spd.functional as functional_spd
import cplx.nn as nn_cplx

data_path_radar='./data/hdm05/'
classes=117

class DatasetHDM05(torch.utils.data.Dataset):
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
class DataLoaderHDM05:
    def __init__(self,data_path,ptest):
        for filenames in os.walk(data_path):
            names=sorted(filenames[2])
        random.Random(0).shuffle(names)
        N_test=int(ptest*len(names))
        N_train=len(names)-N_test
        train_set=DatasetHDM05(data_path,names[N_test:])
        test_set=DatasetHDM05(data_path,names[:N_test])
        self._train_generator=torch.utils.data.DataLoader(train_set,batch_size=N_train,shuffle='False')
        self._test_generator=torch.utils.data.DataLoader(test_set,batch_size=N_test,shuffle='False')

ptest=.5
data_loader_radar=DataLoaderHDM05(data_path_radar,ptest)

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

