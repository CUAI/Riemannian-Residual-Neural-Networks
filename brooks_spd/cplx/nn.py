import torch as th
import torch.nn as nn
import numpy.random
import torch.nn.functional as F
import numpy as np
from . import functional

class Conv_cplx(nn.Module):
    '''
    Interface complex conv layer
    '''
    def forward(self,X):
        return functional.conv_cplx(X,self._conv_Re,self._conv_Im)

class Conv1d_cplx(Conv_cplx):
    '''
    1D complex conv layer
    Inputs a 3D Tensor (batch_size,2*C,T)
    C is the number of channels, 2*C=in_channels the effective number of channels for handling complex data
    Contains two real-valued conv layers
    Output is of shape (batch_size,out_channels,T) (complex channels is out_channels//2)
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride,bias=False):
        super(__class__,self).__init__()
        self._conv_Re=nn.Conv1d(in_channels//2,out_channels,kernel_size,stride,bias=bias)
        self._conv_Im=nn.Conv1d(in_channels//2,out_channels,kernel_size,stride,bias=bias)

class FFT(Conv1d_cplx):
    '''
    1D complex conv layer, where the weights are the Fourier atoms
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride,bias=False):
        super(__class__, self).__init__(in_channels,out_channels,kernel_size,stride,bias)
        atoms=signal_utils.fourier_atoms(out_channels,kernel_size).conj()
        atoms_Re=th.from_numpy(utils.cplx2bichan(atoms)[:,0,:][:,None,:]).float()
        atoms_Im=th.from_numpy(utils.cplx2bichan(atoms)[:,1,:][:,None,:]).float()
        self._conv_Re.weight.data=nn.Parameter(atoms_Re)
        self._conv_Im.weight.data=nn.Parameter(atoms_Im)
        for param in list(self.parameters()):
            param.requires_grad=False

class SplitSignal_cplx(Conv1d_cplx):
    '''
    1D to 2D complex conv layer, where the weights are adequately placed zeroes and ones to split a signal
    Input is still (batch_size,2,T)
    Output is 4-D complex signal (batch_size,2,window_size,T') instead of (batch_size,2*window_size,T')
    '''
    def __init__(self,in_channels,window_size,hop_length):
        super(__class__, self).__init__(in_channels,window_size,window_size,hop_length,False)
        self._conv_Re.weight.data=th.eye(window_size)[:,None,:]
        self._conv_Im.weight.data=th.eye(window_size)[:,None,:]
        for param in list(self._conv_Re.parameters()):
            param.requires_grad=False
        for param in list(self._conv_Im.parameters()):
            param.requires_grad=False
    def forward(self,X):
        return functional.split_signal_cplx(X,self._conv_Re,self._conv_Im)

class SplitSignalBlock_cplx(Conv1d_cplx):
    '''
    1D to 2D complex conv layer, where the weights are adequately placed zeroes and ones to split a signal
    Input is now L blocks of 3-D complex signals (batch_size,L,2*C_in,T), which are acted on independently by the conv layer
    Output is L blocks of 3-D complex signals (batch_size,L,2*C_out,window_size,T')
    '''
    def __init__(self,in_channels,window_size,hop_length):
        super(__class__, self).__init__(in_channels,window_size,window_size,hop_length,False)
        # self._conv_Re.weight.data=th.ones_like(self._conv_Re.weight.data)
        self._conv_Re.weight.data=th.eye(window_size)[:,None,:]
        # self._conv_Im.weight.data=th.ones_like(self._conv_Im.weight.data)
        self._conv_Im.weight.data=th.eye(window_size)[:,None,:]
        for param in list(self._conv_Re.parameters()):
            param.requires_grad=False
        for param in list(self._conv_Im.parameters()):
            param.requires_grad=False
    def forward(self,X):
        XX=[functional.split_signal_cplx(X[:,i,:,:],self._conv_Re,self._conv_Im)[:,None,:,:,:]
            for i in range(X.shape[1])]
        return th.cat(XX,1)

class Conv2d_cplx(Conv_cplx):
    '''
    2D complex conv layer
    Inputs a 4D Tensor (N,2*C,H,W)
    C is the number of channels, 2*C the effective number of channels for handling complex data
    Contains two real-valued conv layers
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride=(1,1),bias=True):
        super(Conv2d_cplx,self).__init__()
        self._conv_Re=nn.Conv2d(in_channels//2,out_channels//2,kernel_size,stride,bias=bias)
        self._conv_Im=nn.Conv2d(in_channels//2,out_channels//2,kernel_size,stride,bias=bias)
        sigma=2./(in_channels//2+out_channels//2)
        ampli=numpy.random.rayleigh(sigma,(out_channels//2,in_channels//2,kernel_size[0],kernel_size[1]))
        phase=numpy.random.uniform(-np.pi,np.pi,(out_channels//2,in_channels//2,kernel_size[0],kernel_size[1]))
        self._conv_Re.weight.data=nn.Parameter(th.Tensor(ampli*np.cos(phase)))
        self._conv_Im.weight.data=nn.Parameter(th.Tensor(ampli*np.sin(phase)))

class ReLU_cplx(nn.ReLU):
    pass
    # def forward(self,X):
    #     return F.relu(X)

class Roll(nn.Module):
    def forward(self,X):
        return functional.roll(X)

class Decibel(nn.Module):
    '''
    Inputs a (N,2*C,H,W) complex tensor
    Outputs a (N,C,H,W) real tensor containing the input's decibel amplitude
    '''
    def forward(self,X):
        return functional.decibel(X)

class AbsolutSquared(nn.Module):
    '''
    Inputs a (N,2*C,H,W) complex tensor
    Outputs a (N,C,H,W) real tensor containing the input's squared module
    '''
    def forward(self,X):
        return functional.absolut_squared(X)

class oneD2twoD_cplx(nn.Module):
    '''
    Inputs a 3D complex tensor (N,2*n,T)
    Outputs a 4D complex tensor (N,2,n,T)
    '''
    def forward(self,x):
        return functional.oneD2twoD_cplx(x)

class MaxPool2d_cplx(nn.MaxPool2d):
    pass
    # def __init__(self,kernel_size,stride=None,padding=0,dilation=1,return_indices=False,ceil_mode=False):
    #     super(MaxPool2d_cplx, self).__init__(kernel_size,stride,padding,dilation,return_indices,ceil_mode)
    #     self._abs=AbsolutSquared()
    # def forward(self,X):
    #     N,C,H,W=X.shape
    #     X_abs=self._abs(X)
    #     _,idx=F.max_pool2d(X_abs,self.kernel_size,self.stride,self.padding,self.dilation,self.ceil_mode,return_indices=True)
    #     XX=X.split(C//2,1)
    #     Y_re=th.gather(XX[0].view(N,C//2,-1),-1,idx.view(N,C//2,-1)).view(idx.shape)
    #     Y_im=th.gather(XX[1].view(N,C//2,-1),-1,idx.view(N,C//2,-1)).view(idx.shape)
    #     Y=th.cat((Y_re,Y_im),1)
    #     return Y

class BatchNorm2d_cplx(nn.BatchNorm2d):
    '''
    Inputs a (N,2*C,H,W) complex tensor
    Outputs a whitened and parametrically rescaled (N,2*C,H,W) complex tensor
    '''
    # def __init__(self,in_channels):
    #     super(BatchNorm2d_cplx,self).__init__(in_channels)
    #     self.momentum=0.1
    #     self.running_mean=th.zeros(in_channels//2,2)
    #     self.running_var=th.eye(2)[None,:,:].repeat(in_channels//2,1,1)/(2**.5)
    #     self._gamma11=nn.Parameter(th.ones(in_channels//2)/2**.5)
    #     self._gamma22=nn.Parameter(th.ones(in_channels//2)/2**.5)
    #     self._gamma12=nn.Parameter(th.zeros(in_channels//2))
    #     self.bias=nn.Parameter(th.zeros(in_channels//2,2))
    # def forward(self,X):
    #     return functional.batch_norm2d_cplx(X,self.running_mean,self.running_var,
    #         self._gamma11,self._gamma12,self._gamma22,self.bias,self.momentum,self.training)
    pass

class BatchNorm2d_cplxSPD(nn.BatchNorm2d):
    '''
    Inputs a (N,2*C,H,W) complex tensor
    Outputs a whitened and parametrically rescaled (N,2*C,H,W) complex tensor
    '''
    def __init__(self,in_channels):
        super(BatchNorm2d_cplxSPD,self).__init__(in_channels)
        self.momentum=0.1
        self.running_mean=th.zeros(in_channels//2,2)
        self.running_var=th.eye(2)[None,:,:].repeat(in_channels//2,1,1)/(2**.5)
        self.weight_=nn.ParameterList([functional_spd.SPDParameter(th.eye(2)/(2**.5)) for _ in range(in_channels//2)])
        self.bias=nn.Parameter(th.zeros(in_channels//2,2))
    def forward(self,X):
        return functional.batch_norm2d_cplx_spd(X,self.running_mean,self.running_var,
                                                     self.weight_,self.bias,self.momentum,self.training)

class BatchNorm2d(nn.BatchNorm2d):
    pass
    # def __init__(self,in_channels):
    #     super(BatchNorm2d,self).__init__(in_channels)
    #     self.momentum=0.1
    #     self.running_mean=th.zeros(in_channels)
    #     self.running_var=th.zeros(in_channels)
    #     self.weight=nn.Parameter(th.ones(in_channels))
    #     self.bias=nn.Parameter(th.zeros(in_channels))
    # def forward(self,X):
    #     N,C,H,W=X.shape
    #     X_vec=X.transpose(0,1).contiguous().view(C,N*H*W)
    #     if(self.training):
    #         mu=X_vec.mean(1); var=X_vec.var(1)
    #         with th.no_grad():
    #             self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mu
    #             self.running_var=(1-self.momentum)*self.running_var+self.momentum*var
    #         Y=(X_vec-mu.view(-1,1))/(var.view(-1,1)**.5+self.eps)
    #     else:
    #         Y=(X_vec-self.running_mean.view(-1,1))/(self.running_var.view(-1,1)**.5+self.eps)
    #     Z=self.weight.view(-1,1)*Y+self.bias.view(-1,1)
    #     return Z.view(C,N,H,W).transpose(0,1)
    #
    # def forward(self, x):
    #     self._check_input_dim(x)
    #     y = x.transpose(0,1)
    #     return_shape = y.shape
    #     y = y.contiguous().view(x.size(1), -1)
    #     mu = y.mean(dim=1)
    #     sigma2 = y.var(dim=1)
    #     if self.training is not True:
    #         y = y - self.running_mean.view(-1, 1)
    #         y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
    #     else:
    #         if self.track_running_stats is True:
    #             with torch.no_grad():
    #                 self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
    #                 self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
    #         y = y - mu.view(-1,1)
    #         y = y / (sigma2.view(-1,1)**.5 + self.eps)
    #
    #     y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
    #     return y.view(return_shape).transpose(0,1)

class CovPool_cplx(nn.Module):
    """
    Input f: Temporal n-dimensionnal complex feature map of length T (T=1 for a unitary signal) (batch_size,2,n,T)
    Output X: Complex covariance matrix of size (batch_size,2,n,n)
    """
    def __init__(self,reg_mode='mle',N_estimates=None):
        super(__class__,self).__init__()
        self._reg_mode=reg_mode; self.N_estimates=N_estimates
    def forward(self,f):
        return functional.cov_pool_cplx(f,self._reg_mode,self.N_estimates)

class CovPoolBlock_cplx(nn.Module):
    """
    Input f: L blocks of temporal n-dimensionnal complex feature map of length T of shape (batch_size,L,2,n,T)
    Output X: L blocks of complex covariance matrix of size (batch_size,L,2,n,n)
    """
    def __init__(self,reg_mode='mle',N_estimates=None):
        super(__class__,self).__init__()
        self._reg_mode=reg_mode; self.N_estimates=N_estimates
    def forward(self,f):
        XX=[functional.cov_pool_cplx(f[:,i,:,:,:],self._reg_mode,self.N_estimates)[:,None,:,:,:]
            for i in range(f.shape[1])]
        return th.cat(XX,1)