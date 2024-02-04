import numpy as np
import torch as th
import torch.nn as nn
import torch.optim
from . import functional

class StiefelOptim():
    """ Optimizer with orthogonality constraints """
    def __init__(self, parameters, lr) :
        self.parameters = parameters
        self.lr = lr
    def step(self):
        for W in self.parameters:
            dir_tan=proj_tanX_stiefel(W.grad.data,W.data)
            W_new=ExpX_stiefel(-self.lr*dir_tan.data,W.data)
            W.data=W_new
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.zero_()

class StiefelBlockOptim():
    """ Optimizer with block orthogonality constraints (for BiMapConv) """
    def __init__(self, parameters, lr) :
        self.parameters = parameters
        self.lr = lr
    def step(self):
        for W in self.parameters:
            dir_tan=th.cat([proj_tanX_stiefel(W.grad[:,:,i,:,:].data,W[:,:,i,:,:].data)[:,:,None,:,:]
                            for i in range(W.shape[2])],2)
            W_new=th.cat([ExpX_stiefel(-self.lr*dir_tan[:,:,i,:,:].data,W[:,:,i,:,:].data)[:,:,None,:,:]
                          for i in range(W.shape[2])],2)
            W.data=W_new
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.zero_()

class SPDOptim() :
    """ Optimizer with SPD constraints """
    def __init__(self, parameters, lr) :
        self.parameters = parameters
        self.lr = lr
    def step(self):
        for W in self.parameters:
            dir_tan=proj_tanX_spd(W.grad.data,W.data)
            W.data=functional.ExpG(-self.lr*dir_tan.data,W.data)[0,0]
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.zero_()

class WeightVectorOptim():
    """ Optimizer with weight vector constraints """
    def __init__(self, parameters, lr) :
        self.parameters = parameters
        self.lr = lr
    def step(self):
        for W in self.parameters:
            dw_pos=(W-self.lr*W.grad).abs()
            W.data=dw_pos/dw_pos.sum()
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.zero_()

class MixOptimizer():
    """ Optimizer with mixed constraints """

    def __init__(self, parameters, optimizer=torch.optim.SGD, lr=1e-2, *args, **kwargs):
        parameters = list(parameters)
        parameters=[param for param in parameters if param.requires_grad]
        self.lr = lr
        self.stiefel_parameters = [param for param in parameters if param.__class__.__name__=='StiefelParameter']
        self.stiefel_block_parameters = [param for param in parameters if param.__class__.__name__=='StiefelBlockParameter']
        self.spd_parameters = [param for param in parameters if param.__class__.__name__=='SPDParameter']
        self.weight_vector_parameters = [param for param in parameters if param.__class__.__name__=='WeightVectorParameter']
        self.other_parameters = [param for param in parameters if param.__class__.__name__=='Parameter']

        self.stiefel_optim = StiefelOptim(self.stiefel_parameters, self.lr)
        self.stiefel_block_optim = StiefelBlockOptim(self.stiefel_block_parameters, self.lr)
        self.spd_optim = SPDOptim(self.spd_parameters, self.lr)
        self.weight_vector_optim = WeightVectorOptim(self.weight_vector_parameters, self.lr)
        self.optim = optimizer(self.other_parameters, lr, *args, **kwargs)

    def step(self):
        self.optim.step()
        self.stiefel_optim.step()
        self.stiefel_block_optim.step()
        self.spd_optim.step()
        self.weight_vector_optim.step()

    def zero_grad(self):
        self.optim.zero_grad()
        self.stiefel_optim.zero_grad()
        self.spd_optim.zero_grad()

def proj_tanX_stiefel(x,X):
    """ Projection of x in the Stiefel manifold's tangent space at X """
    return x-X.matmul(x.transpose(2,3)).matmul(X)

def ExpX_stiefel(x,X):
    """ Exponential mapping of x on the Stiefel manifold at X (retraction operation) """
    a=X+x
    Q=th.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # q,_=th.qr(a[i,j])
            q,_=gram_schmidt(a[i,j])
            Q[i,j]=q
    return Q

def proj_tanX_spd(x,X):
    """ Projection of x in the SPD manifold's tangent space at X """
    return X.matmul(sym(x)).matmul(X)

#V is a in M(n,N); output W an semi-orthonormal free family of Rn; we consider n>=N
#also returns R such that WR is the QR decomposition
def gram_schmidt(V):
    n,N=V.shape #dimension, cardinal
    W=th.zeros_like(V)
    R=th.zeros((N,N)).double().to(V.device)
    W[:,0]=V[:,0]/th.norm(V[:,0])
    R[0,0]=W[:,0].dot(V[:,0])
    for i in range(1,N):
        proj=th.zeros(n).double().to(V.device)
        for j in range(i):
            proj=proj+V[:,i].dot(W[:,j])*W[:,j]
            R[j,i]=W[:,j].dot(V[:,i])
        if(isclose(th.norm(V[:,i]-proj),th.DoubleTensor([0]).to(V.device))):
            W[:,i]=V[:,i]/th.norm(V[:,i])
        else:
            W[:,i]=(V[:,i]-proj)/th.norm(V[:,i]-proj)
        R[i,i]=W[:,i].dot(V[:,i])
    return W,R

def isclose(a,b,rtol=1e-05,atol=1e-08):
    return ((a - b).abs() <= (atol + rtol * b.abs())).all()

def sym(X):
    if(len(X.shape)==2):
        if isinstance(X,np.ndarray):
            return 0.5*(X+X.T.conj())
        else:
            return 0.5*(X+X.t())
    elif(len(X.shape)==3):
        if isinstance(X,np.ndarray):
            return 0.5*(X+X.transpose([0,2,1]))
        else:
            return 0.5*(X+X.transpose(1,2))
    elif(len(X.shape)==4):
        if isinstance(X,np.ndarray):
            return 0.5*(X+X.transpose([0,1,3,2]))
        else:
            return 0.5*(X+X.transpose(2,3))
