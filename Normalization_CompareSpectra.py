#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import fitsio
from desispec.interpolation import resample_flux

def compute_mean_cont_std(wavel, meanspec, weight,coeff,eigvec,n_vec=4, lmin=1420.0, lmax=1500.0, const= 50.0):
    ### Choose the first four eigenvectors
    conti_mock = []
    for i in range(len(coeff)):
        spectram = []
        for j in range(n_vec):
            spectram.append(coeff[i][j]*eigvec[j])
        spectrasm = np.vstack(spectram)
        specm = np.sum(spectrasm,axis = 0)
        conti_mock += [specm]
    continuum_mock = np.vstack(conti_mock)
    ### The continuum is y(lambda)
    for i in range(len(continuum_mock)):
        continuum_mock[i] +=  meanspec                          # continuum = continuum + mean spec.
    ### Normalize the spectra
    #lmin=1420.0                                                # range of normalization 
    #lmax=1500.0                                                #
    #Constant = 50.0                                            # Normalization value (not neccesary to 1).
    integral_mocks = []
    mask = (wavel > lmin) & (wavel < lmax)
    for i in range(len(continuum_mock)):
        sum1 = np.sum((continuum_mock[i][mask])*(wavel[1001]-wavel[1000]))
        integral_mocks.append(sum1)
        continuum_mock[i] = (const*continuum_mock[i])/sum1
        weight[i] = (weight[i])*(((sum1)**2)/const**2)
    ### Stack
    meancont = np.average(continuum_mock, weights = weight,axis=0) # mean continuum.
    ### Get standard deviation
    for i in range(len(continuum_mock)):        
        continuum_mock[i] -= meancont                           # Now, the continuum = continuum - mean_cont. 
    
    vari = np.zeros(len(meancont))   
    for h in range(len(continuum_mock)):
        vari += continuum_mock[h]**2
    varianza = np.sqrt(vari/(len(continuum_mock)))             # standard deviation.
    #std_stack_mock = np.std(continuum_mock,axis=0)            #It's only 
    
    ### Normalization for the eigenvalues.                     
    coefficient = np.zeros((n_vec,len(coeff)))
    for k in range(n_vec):
        coefficient[k] = coeff[:,k]#/integral_mocks           # The coefficients should not be normalized
    #coefficient = coeff.T                                      # coeff[Nspec,Nvec]; coeffiicent[Nvec,Nspec]
    return meancont, varianza, coefficient

def mean_cont(mwave,mflux,dwave,dflux,stdm,stdd,maskmin,maskmax, n1, n2, mmin,mmax,zmin, zmax, name, complete = False):
    #print('MinWM = ', np.min(mwave))
    #print('MaxWM = ', np.max(mwave))
    #print('MinWD = ', np.min(dwave))
    #print('MaxWD = ', np.max(dwave))
    if complete == True:
        plt.figure(figsize=(14,15))
        plt.subplot(2,1,1)
        plt.title(r'Mean Continuum      {}$\leq$Mag$\leq${}    {}$\leq$Z$\leq${}.'.format(mmin,mmax,zmin, zmax),fontsize = 20)
        plt.plot(mwave,mflux,'-', label='{}'.format(n1),linewidth=3,alpha=1)
        plt.fill_between(mwave,mflux+stdm,mflux-stdm, label='std {}'.format(n1), color = 'k', alpha=0.4)
        plt.plot(dwave,dflux,'-', label='{}'.format(n2),linewidth=3,alpha=1)
        plt.fill_between(dwave,dflux+stdd,dflux-stdd, label='std {}'.format(n2), color = 'y', alpha=0.5)
        #axvline(1420,color='k')
        #plt.xlim(xmin,xmax)
        #plt.ylim(-2, 13)
        plt.xlabel('$\lambda_{R.F.}$', fontsize = 20)
        plt.ylabel('$\mathrm{\overline{Normalized \enspace Flux}}$', fontsize = 20)
        plt.legend(fontsize='xx-large')
        plt.grid()
    
        masks = (mwave>maskmin) & (mwave<maskmax)
        mask = (dwave>maskmin) & (dwave<maskmax)

        plt.subplot(2,1,2)
        plt.title(r'Ly-$\{}$ region'.format(name), fontsize = 20)
        plt.plot(mwave[masks],mflux[masks],linewidth=3, label='{}'.format(n1),alpha=1)
        plt.fill_between(mwave[masks],mflux[masks]+stdm[masks],mflux[masks]-stdm[masks], label='std {}'.format(n1), color = 'k', alpha=0.4)
        plt.plot(dwave[mask],dflux[mask],linewidth=3, label='{}'.format(n2),alpha=1)
        plt.fill_between(dwave[mask],dflux[mask]+stdd[mask],dflux[mask]-stdd[mask], label='std {}'.format(n2), color = 'y', alpha=0.5)
        #plt.xlim(xminzoom,xmaxzoom)
        #plt.ylim(1,3.7)
        plt.xlabel('$\lambda_{R.F.}$', fontsize = 20)
        plt.ylabel('$\mathrm{\overline{Normalized \enspace Flux}}$', fontsize = 20)
        plt.legend(fontsize='xx-large')
        plt.grid()
    else:
        masks = (mwave>maskmin) & (mwave<maskmax)
        mask = (dwave>maskmin) & (dwave<maskmax)

        plt.figure(figsize=(14,7))
        plt.title(r'Ly-$\{}$ region        {}$\leq$Mag$\leq${}    {}$\leq$Z$\leq${}'.format(name,mmin,mmax,zmin, zmax), fontsize = 20)
        plt.plot(mwave[masks],mflux[masks],linewidth=3, label='{}'.format(n1),alpha=1)
        plt.fill_between(mwave[masks],mflux[masks]+stdm[masks],mflux[masks]-stdm[masks], label='std {}'.format(n1), color = 'k', alpha=0.4)
        plt.plot(dwave[mask],dflux[mask],linewidth=3, label='{}'.format(n2),alpha=1)
        plt.fill_between(dwave[mask],dflux[mask]+stdd[mask],dflux[mask]-stdd[mask], label='std {}'.format(n2), color = 'y', alpha=0.5)
        #plt.xlim(xminzoom,xmaxzoom)
        #plt.ylim(1,3.7)
        plt.xlabel('$\lambda_{R.F.}$', fontsize = 20)
        plt.ylabel('$\mathrm{\overline{Normalized \enspace Flux}}$', fontsize = 20)
        plt.legend(fontsize='xx-large')
        plt.grid()
    
def eigvec_plot(mmwave,mmeigvec,ddwave,ddeigvec,n1,n2):
    plt.figure(figsize=(14,30))
    for i in range(4):
        plt.subplot(4,1,i+1)
        plt.title('Principal Component {}'.format(i+1),fontsize=20)
        plt.plot(mmwave,mmeigvec[i],label='{}'.format(n1))
        plt.plot(ddwave,ddeigvec[i],label='{}'.format(n2))
        #plt.xlim(xxmin,xxmax)
        #plt.ylim(-2,10)
        plt.xlabel('$\lambda_{R.F.}$', fontsize = 20)
        plt.ylabel('Flux', fontsize = 20)
        #plt.ylabel('$\mathrm{\overline{Flux}}$', fontsize = 20)
        plt.legend(fontsize='xx-large')
        plt.grid()

def coeff_hist(mcoeff,dcoeff,cmin,cmax,sizebin,n1,n2):
    plt.figure(figsize=(26,5))
    plt.title('Distributions of the coefficient',fontsize=20)
    for i in range(len(mcoeff)):
        plt.subplot(1,4,i+1)
        mweights11=np.ones_like(mcoeff[i])/len(mcoeff[i])
        dweights11=np.ones_like(dcoeff[i])/len(dcoeff[i])
        n = int((cmax-cmin)/sizebin)
        eig0bin = np.linspace(cmin,cmax, n+1)
        ii, jj, kk = plt.hist(mcoeff[i],eig0bin,histtype='step',weights=mweights11,label='{}'.format(n1))
        ll, mm, nn = plt.hist(dcoeff[i],eig0bin,histtype='step',weights=dweights11,label='{}'.format(n2))
        ax = plt.gca()
        plt.text(0.1, 0.9, 'Eigvec {}'.format(i+1),transform=ax.transAxes,fontsize = 15)
        plt.xlabel('$c_{k}$', fontsize = 20)
        plt.ylabel('Frequency', fontsize = 20)
        plt.legend(fontsize='xx-large')
        
def desviasion(wavemock1,wavemock2,wavedata,sstd,qstd,dstd,n1,n2,n3, rmin1, rmax1, rmin2, rmax2, mmin, mmax, zmin, zmax):
    ### Make an interpolation before computing  the standard deviation rate
    nwave = np.arange(rmin1,rmax1,wavemock1[1]-wavemock1[0])
    stds = np.zeros(len(sstd))
    stdq = np.zeros(len(qstd))
    stdd = np.zeros(len(dstd))
    stds = resample_flux(nwave, wavemock1, sstd)
    stdq = resample_flux(nwave, wavemock2, qstd)
    stdd = resample_flux(nwave, wavedata, dstd)
    ###
    #mas1 = (nwave>rmin1) & (nwave<rmax1)
    ww = stdd>0.01
    gg = stdq>0.01
    sq=stds[gg]/stdq[gg]
    sd=stds[ww]/stdd[ww]
    qd=stdq[ww]/stdd[ww]
    ### Waves
    wdr = nwave[ww]
    wsq = nwave[gg]
    ### Mask
    dmask = (wdr>rmin2) & (wdr<rmax2)
    mmask = (wsq>rmin2) & (wsq<rmax2)
    ###plots
    plt.figure(figsize=(14,15))
    plt.subplot(2,1,1)
    plt.title(r'Dispersion      {}$\leq$Mag$\leq${}    {}$\leq$Z$\leq${}.'.format(mmin,mmax,zmin, zmax),fontsize = 20)
    plt.plot(wsq,sq,'g', label='{}/{}'.format(n1,n2), alpha=0.7)
    plt.plot(wdr,sd,'b', label='{}/{}'.format(n1,n3), alpha=0.7)
    plt.plot(wdr,qd,'r', label='{}/{}'.format(n2,n3), alpha=0.7)
    #axhline(1, color='k')
    plt.xlabel('$\lambda_{R.F.}$', fontsize = 20)
    plt.ylabel('$\mathrm{\sigma_{i}/\sigma_{j}}$', fontsize = 20)
    plt.grid()
    plt.legend(fontsize='xx-large')
    
    plt.subplot(2,1,2)
    plt.title(r'Ly-$\alpha$ region', fontsize = 20)
    plt.plot(wsq[mmask],sq[mmask],'g', label='{}/{}'.format(n1,n2))
    plt.plot(wdr[dmask],sd[dmask],'b', label='{}/{}'.format(n1,n3))
    plt.plot(wdr[dmask],qd[dmask],'r', label='{}/{}'.format(n2,n3))
    #axhline(1, color='k')
    plt.xlabel(r'$\lambda_{R.F.}$', fontsize = 20)
    plt.ylabel(r'$\mathrm{\sigma_{i}/\sigma_{j}}$', fontsize = 20)
    plt.legend(fontsize='xx-large')
    plt.grid()