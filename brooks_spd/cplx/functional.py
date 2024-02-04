import torch as th

def conv_cplx(X,conv_re,conv_im):
    if(isinstance(X,list)):
        XX=[x.split(x.shape[0]//2,0) for x in X]
        tmp=X[0].split(X[0].shape[0]//2,0)
        tmpp=conv_re(tmp[0][None,...])-conv_im(tmp[1][None,...])
        P_Re=[conv_re(xx[0][None,...])-conv_im(xx[1][None,...]) for xx in XX]
        P_Im=[conv_im(xx[0][None,...])+conv_re(xx[1][None,...]) for xx in XX]
        P=[th.cat((p_Re,p_Im),0) for p_Re,p_Im in zip(P_Re,P_Im)]
    else:
        XX=X.split(X.shape[1]//2,1)
        P_Re=conv_re(XX[0])-conv_im(XX[1])
        P_Im=conv_im(XX[0])+conv_re(XX[1])
        P=th.cat((P_Re,P_Im),1)
    return P

def split_signal_cplx(X,conv_re,conv_im):
    '''
    1D to 2D complex conv layer, where the weights are adequately placed zeroes and ones to split a signal
    '''
    XX=X.split(X.shape[1]//2,1)
    P_Re=conv_re(XX[0])
    P_Im=conv_re(XX[1])
    P=th.cat((P_Re[:,None,:,:],P_Im[:,None,:,:]),1)
    return P

def roll(X):
    return th.cat((X[:,:,X.shape[-2]//2:,:],X[:,:,:X.shape[-2]//2,:]),2)

def decibel(X):
    # n_fft=X.shape[-2]//2
    # X_Re=X[:,:n_fft,:]
    # X_Im=X[:,n_fft:,:]
    if(isinstance(X,list)):
        absX=AbsolutSquared()(X)
        X_db=[10*th.log(x) for x in absX]
    else:
        X_db=10*th.log(absolut_squared(X))
    return X_db

def absolut_squared(X):
    '''
    Inputs a (N,2*C,H,W) complex tensor
    Outputs a (N,C,H,W) real tensor containing the input's squared module
    '''
    XX=X.split(X.shape[1]//2,1)
    return (XX[0]**2+XX[1]**2)

def oneD2twoD_cplx(x):
    '''
    Inputs a 3D complex tensor (N,2*n,T)
    Outputs a 4D complex tensor (N,2,n,T)
    '''
    return x.view(x.shape[0],2,-1,x.shape[-1])

def batch_norm2d_cplx(X,running_mean,running_var,gamma11,gamma12,gamma22,bias,momentum,training):
    N,C,H,W=X.shape
    XX=list(X.split(C//2,1))
    X_re=XX[0].transpose(0,1).contiguous().view(C//2,N*H*W) #(C//2,NHW)
    X_im=XX[1].transpose(0,1).contiguous().view(C//2,N*H*W) #(C//2,NHW)
    X_cplx=th.cat((X_re[:,None,:],X_im[:,None,:]),1) #(C//2,2,NHW)
    if(training):
        mu_re=X_re.mean(1); mu_im=X_im.mean(1) #(C//2,)
        mu_cplx=th.cat((mu_re[:,None],mu_im[:,None]),1) #(C//2,2)
        var_re=X_re.var(1); var_im=X_im.var(1) #(C//2,)
        cov_imre=((X_re-mu_re[:,None])*(X_im-mu_im[:,None])).sum(1)/(N*H*W-1) #(C//2,)
        cov_cplx=utils.twobytwo_covmat_from_coeffs(var_re,var_im,cov_imre)
        with th.no_grad():
            running_mean=(1-momentum)*running_mean+momentum*mu_cplx #(C//2,2)
            running_var=(1-momentum)*running_var+momentum*cov_cplx #(C//2,2,2)
        # cov_cplx_sqinv=(functional_spd.SqminvEig()(cov_cplx[:,None,:,:]))[:,0,:,:]
        cov_cplx_sqinv=utils.twobytwo_sqinv(var_re,var_im,cov_imre)
        Y=cov_cplx_sqinv.matmul((X_cplx-mu_cplx[:,:,None]))
    else:
        # running_var_sqinv=(functional_spd.SqminvEig()(running_var[:,None,:,:].double())).float()[:,0,:,:]
        running_var_sqinv=utils.twobytwo_sqinv(running_var[:,0,0],running_var[:,1,1],running_var[:,1,0])
        Y=running_var_sqinv.matmul((X_cplx-running_mean[:,:,None]))
    weight=utils.twobytwo_covmat_from_coeffs(gamma11,gamma22,gamma12)
    Z=weight.matmul(Y)+bias[:,:,None]
    return Z.view(C,-1).view(C,N,H,W).transpose(0,1)

def batch_norm2d_cplx_spd(X,running_mean,running_var,weight,bias,momentum,training):
    N,C,H,W=X.shape
    XX=list(X.split(C//2,1))
    X_re=XX[0].transpose(0,1).contiguous().view(C//2,N*H*W) #(C//2,NHW)
    X_im=XX[1].transpose(0,1).contiguous().view(C//2,N*H*W) #(C//2,NHW)
    X_cplx=th.cat((X_re[:,None,:],X_im[:,None,:]),1) #(C//2,2,NHW)
    if(training):
        mu_re=X_re.mean(1); mu_im=X_im.mean(1) #(C//2,)
        mu_cplx=th.cat((mu_re[:,None],mu_im[:,None]),1) #(C//2,2)
        var_re=X_re.var(1); var_im=X_im.var(1) #(C//2,)
        cov_imre=((X_re-mu_re[:,None])*(X_im-mu_im[:,None])).sum(1)/(N*H*W-1) #(C//2,)
        cov_cplx=utils.twobytwo_covmat_from_coeffs(var_re,var_im,cov_imre)
        with th.no_grad():
            running_mean=(1-momentum)*running_mean+momentum*mu_cplx #(C//2,2)
            running_var=(1-momentum)*running_var+momentum*cov_cplx #(C//2,2,2)
        cov_cplx_sqinv=(functional_spd.SqminvEig()(cov_cplx[:,None,:,:]))[:,0,:,:]
        # cov_cplx_sqinv=utils.twobytwo_sqinv(var_re,var_im,cov_imre)
        Y=cov_cplx_sqinv.matmul((X_cplx-mu_cplx[:,:,None]))
    else:
        running_var_sqinv=(functional_spd.SqminvEig()(running_var[:,None,:,:].double())).float()[:,0,:,:]
        # running_var_sqinv=utils.twobytwo_sqinv(running_var[:,0,0],running_var[:,1,1],running_var[:,1,0])
        Y=running_var_sqinv.matmul((X_cplx-running_mean[:,:,None]))
    weight=utils.twobytwo_covmat_from_coeffs(_gamma11,_gamma22,_gamma12) ##################### CHANGE
    Z=weight.matmul(Y)+bias[:,:,None]
    return Z.view(C,-1).view(C,N,H,W).transpose(0,1)


def cov_pool_cplx(f,reg_mode='mle',N_estimates=None):
    """
    Input f: Temporal n-dimensionnal complex feature map of length T (T=1 for a unitary signal) (batch_size,2,n,T)
    Output X: Complex covariance matrix of size (batch_size,2,n,n)
    """
    N,_,n,T=f.shape
    ff=f.split(f.shape[1]//2,1)
    f_re=ff[0]; f_im=ff[1]
    if(N_estimates is not None):
        f_re=f_re.split(self._Nestimates,-1)
        if(f_re[-1].shape[-1]!=self._Nestimates):
            f_re=th.cat(f_re[:-1]+(th.cat((f_re[-1],th.zeros(N,1,n,self._Nestimates-f_re[-1].shape[-1])),-1),),1)
        else:
            f_re=th.cat(f_re,1)
        f_im=f_im.split(self._Nestimates,-1)
        if(f_im[-1].shape[-1]!=self._Nestimates):
            f_im=th.cat(f_im[:-1]+(th.cat((f_im[-1],th.zeros(N,1,n,self._Nestimates-f_im[-1].shape[-1])),-1),),1)
        else:
            f_im=th.cat(f_im,1)
    f_re-=f_re.mean(-1,True); f_im-=f_im.mean(-1,True)
    f_re=f_re.double(); f_im=f_im.double()
    X_Re=((f_re.matmul(f_re.transpose(-1,-2))+f_im.matmul(f_im.transpose(-1,-2)))/(f.shape[-1]-1))
    X_Im=((f_im.matmul(f_re.transpose(-1,-2))-f_re.matmul(f_im.transpose(-1,-2)))/(f.shape[-1]-1))
    if(reg_mode=='mle'):
        pass
    elif(self._reg_mode=='add_id'):
        X_Re=RegulEig(1e-6)(X_Re)
        X_Im=RegulEig(1e-6)(X_Im)
    elif(self._reg_mode=='adjust_eig'):
        X_Re=AdjustEig(0.75)(X_Re)
        X_Im=AdjustEig(0.75)(X_Im)
    X=(X_Re+X_Im)/2 ############## later, do cat for HPD
    # X=th.cat((X_Re,X_Im),1) #for real complex matrices
    return X
