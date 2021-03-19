#!/usr/bin/env python

import numpy as np
import fitsio
from desispec.interpolation import resample_flux
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light as speed_light
from scipy.fftpack import fft
from glob import glob
import time

def build_continuums(meanspec, coeff, eigvec, n_vec = 4):
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
    return continuum_mock

def normalization(wavel, spectra, contt, lmin, lmax, const, weight = None):
    mask = (wavel > lmin) & (wavel < lmax)
    sum1 = np.sum((contt[mask])*(wavel[10] - wavel[9]))
    spectranor = (const*spectra)/sum1
    conttnor = (const*contt)/sum1
    if weight is not None:
        weightnor = (weight)*(((sum1)**2)/const**2)  
        return spectranor, conttnor, weightnor
    else:
        return spectranor, conttnor
    
def deltas(dflujo, dcont, dweight):
    F = np.zeros((dflujo.shape))                                # Transmision
    for i in range(len(dflujo)):
        F[i] = dflujo[i]/dcont[i]
    Fmean = np.average(F, weights = dweight, axis = 0)          #Transmision promedio  
    delta = F/Fmean - 1                                         #Delta
    Dmean = np.average(delta, weights = dweight, axis = 0)
    return F, Fmean, delta, Dmean
    
def deltas_from_pixels(arch, wll, pesos):
    elem = np.linspace(1,500,500).astype('int')
    ws = []
    ds = []
    wt = []
    for i in elem:
        try:
            ws.append((10**(arch[i].read()['LOGLAM']))/(1.0 + arch[i].read_header()['Z']))
            ds.append(arch[i].read()['DELTA'])
            if pesos == 'No-Pk1D':
                wt.append(arch[i].read()['WEIGHT'])
            elif pesos == 'Pk1D':
                wt.append(arch[i].read()['IVAR'])
        except IOError:
            continue
    dss  = np.zeros((len(ws), wll.size))
    wts  = np.zeros((len(ws), wll.size))

    for i in range(len(ws)):
        dss[i], wts[i] = resample_flux(wll,ws[i],ds[i], wt[i]) # interpolation 
    return dss, wts

def deltas_picca(wll,filess,peso):
    dfs = []
    wgt = []
    for i in range(len(filess)):
        data_dir = filess[i]
        try:
            dpca = fitsio.FITS(data_dir)
        except IOError:
            continue
        ddfs, weights = deltas_from_pixels(dpca, wll,peso)
        dfs.append(ddfs)
        wgt.append(weights)
    delfs = np.vstack(dfs)
    wgts  = np.vstack(wgt)
    return delfs, wgts

def pkraw(vel, deltasv):
    ####
    nb_pixels = len(deltasv[0])
    nbin_fft = nb_pixels//2 + 1
    Pk = np.zeros((len(deltasv),nbin_fft))
    delkd = []
    for i in range(len(deltasv)):
        #Hacer FFT
        fftdelta = fft(deltasv[i])
        #Calcular 1DPS
        fftdelta = fftdelta[:nbin_fft]
        Pk[i] = (fftdelta.real**2 + fftdelta.imag**2) * (vel*nb_pixels)/(nb_pixels**2)
        delkd.append(fftdelta)
    deltk = np.vstack(delkd)
    Pkm = np.average(Pk, axis=0)
    #std = np.std(Pk, axis = 0 ) 
    k = np.arange(nbin_fft,dtype=float)*2*np.pi/(vel*nb_pixels)                       #  k=2π/Δv
    return k, Pkm, Pk, deltk

def pk1noise(vel, deltasvv):
    ####
    nb_pixels = len(deltasvv)
    nbin_fft = nb_pixels//2 + 1
    #Hacer FFT
    fftdelta = fft(deltasvv)
    #Calcular 1DPS
    fftdelta = fftdelta[:nbin_fft]
    Pk = (fftdelta.real**2 + fftdelta.imag**2) * (vel*nb_pixels)/(nb_pixels**2)
    k = np.arange(nbin_fft,dtype=float)*2*np.pi/(vel*nb_pixels)                       #  k=2π/Δv
    return k, Pk, fftdelta

def pknoise(vel,noise):
    ####
    nb_pixels = len(noise[0])
    nbin_fft = nb_pixels//2 + 1
    pknois = np.zeros((len(noise),nbin_fft))
    delkiv = []
    #
    for i in range(len(noise)):
        nb_noise_exp = 10
        Pk = np.zeros(nbin_fft)
        delknoise = []
        err = np.zeros(nb_pixels)
        noises = noise[i]
        w = noises>0
        err[w] = 1.0/np.sqrt(noises[w])
        for j in range(nb_noise_exp):
            delta_exp= np.zeros(nb_pixels)
            delta_exp[w] = np.random.normal(0.,err[w])
            j,Pk_exp, deltasnoise = pk1noise(vel,delta_exp) #k_exp unused, but needed
            Pk += Pk_exp
            delknoise.append(deltasnoise)
        Pk /= float(nb_noise_exp)
        delknoise = np.vstack(delknoise)
        delknoise = np.sum(delknoise,axis=0)/float(nb_noise_exp)
        pknois[i] = Pk
        delkiv.append(delknoise)
    Pknoises = np.average(pknois, axis=0)
    delkiv = np.vstack(delkiv)
    delknoises = np.average(delkiv, axis=0)
    return Pknoises, pknois, delkiv

def window(vel, reso, kvec,clase):
    nb_bin_FFT = len(kvec)
    cor = np.ones(nb_bin_FFT)

    sinc = np.ones(nb_bin_FFT)
    sinc[kvec>0.] = np.sin(kvec[kvec>0.]*vel/2.0)/(kvec[kvec>0.]*vel/2.0)

    cor *= np.exp(-((kvec*reso)**2)/2.0)
    cor *= sinc
    if clase == 'Pk1D':
        Win = cor**2
    elif clase == 'Bk1D':
        Win = cor**3
    return Win

def Bispec(delvel,kbs, deltbs, deltbsn):
    lenforest = 2 * kbs.size - 1
    rr = np.arange(kbs.size)
    mask1 = (rr % 2 == 0)
    mask2 = (rr % 3 == 0)
    Bk1 = np.zeros((len(deltbs), kbs[mask1].size))
    Bk2 = np.zeros((len(deltbs), kbs[mask2].size))
    for i in range(len(deltbs)):
        Bk11 = deltbs[i][:kbs[mask1].size]*deltbs[i][:kbs[mask1].size]*deltbsn[i][mask1] # se hace δ(k1)δ(k1)δ(−2k1)
        Bk1[i] = Bk11.real * (delvel**2) / (lenforest**2)                    
        Bk22 = deltbs[i][mask1][:kbs[mask2].size]*deltbs[i][:kbs[mask2].size]*deltbsn[i][mask2]# se hace δ(2k1)δ(k1)δ(−3k1)
        Bk2[i] = Bk22.real * (delvel**2) / (lenforest**2)

    BS1 = np.average(Bk1,axis=0)
    BS2 = np.average(Bk2,axis=0)
    kbs1 = kbs[:kbs[mask1].size]
    kbs2 = kbs[:kbs[mask2].size]
    return kbs1, kbs2, BS1, BS2

def BispecRandom(delvel,kbs, deltbs, deltbsn):
    n_cont = 50                               # Numero de muestras aleatorias
    rr = np.arange(kbs.size)
    mask1 = (rr % 2 == 0)
    mask2 = (rr % 3 == 0)
    err_k1 = np.zeros(kbs[mask1].size)
    err_2k1 = np.zeros(kbs[mask2].size)
    deltbs_c = np.copy(deltbs)
    deltbsn_c = np.copy(deltbsn)
    for i in range(n_cont):
        deltt = []
        delttn = []
        for j in range(len(deltbs_c)):
            np.random.shuffle(deltbs_c[j])
            np.random.shuffle(deltbsn_c[j])
            deltt.append(deltbs_c[j])
            delttn.append(deltbsn_c[j])
        deltbss = np.vstack(deltt)
        deltbssn = np.vstack(delttn)
        _, __, bsm, bsm2 = Bispec(delvel,kbs, deltbss, deltbssn)
        err_k1 += bsm
        err_2k1 += bsm2
    err_k1 /= float(n_cont)
    err_2k1 /= float(n_cont)
    return err_k1, err_2k1

def stack(kvv,pwsp,sizes):
    size_elem = sizes
    mean = []
    for i in range(kvv.size):
        if size_elem*(i + 1) <= kvv.size:
            mean.append(np.average(pwsp[size_elem*(i):size_elem*(i + 1)]))
        else:
            break
    spwsp = np.hstack(mean)
    return spwsp

#### Bispectrum with Deltas from PCA ####

class MyDeltas:
    def __init__(self,dirfile):
        time1 = time.time()
        hdu = fitsio.FITS(dirfile)
        ### read data.
        wavedata   = hdu[1]['wavelength'].read()         # wavelenth of qso data
        meandata   = hdu[1]['mean_spectrum'].read()      # mean spectrum of qso data
        fluxdata   = hdu[2]['flux_spectra'].read()       # weights of qso data
        weightdata = hdu[2]['weights'].read()            # weights of qso data
        vecdata    = hdu[3]['eigvec'].read()             # empca eigvectors of qso
        coeffdata  = hdu[4]['coeff'].read()              # empca coefficients of qso.
        print('Num. BOSS DR16 spectra = ', coeffdata.shape[0])
        contdata = build_continuums(meandata,coeffdata,vecdata)
        ###### Choose only the Ly-a forest
        maskwv = (wavedata >= 1040.0) & (wavedata < 1200.0)
        wavelya      = wavedata[maskwv]
        fluxlya   = np.zeros((fluxdata.shape[0],wavelya.size))
        weightlya = np.zeros((fluxdata.shape[0],wavelya.size))
        contlya   = np.zeros((fluxdata.shape[0],wavelya.size))
        for i in range(fluxdata.shape[0]):
            fluxlya[i]   = fluxdata[i][maskwv] + meandata[maskwv]
            weightlya[i] = weightdata[i][maskwv]
            contlya[i]   = contdata[i][maskwv]

        ######## Normalization
        fluxlyan   = np.zeros((fluxdata.shape[0],wavelya.size))
        weightlyan = np.zeros((fluxdata.shape[0],wavelya.size))
        contlyan   = np.zeros((fluxdata.shape[0],wavelya.size))
        for i in range(fluxlya.shape[0]):
            fluxlyan[i], contlyan[i], weightlyan[i] = normalization(wavelya,fluxlya[i], contlya[i],1040.0, 1199.0,1.0,
                                                                    weight = weightlya[i])
        weightlyan[weightlyan<0.] = 0           #
        w = np.sum(weightlyan,axis=0)>0.        #
        self.wavelya    = wavelya[w]                 # For the noise.
        self.fluxlyan   = fluxlyan[:,w]              #
        self.weightlyan = weightlyan[:,w]            #
        self.contlyan   = contlyan[:,w]              #
        self.weightlyan[weightlyan>100.] = 100.      #
        
        ######## Compute Deltas

        self.T, self.Tprom, self.delt, self.deltprom = deltas(self.fluxlyan,self.contlyan,self.weightlyan)

        ####### Tune the wavelength 

        self.logwave = np.log10(self.wavelya)
        print('Δlog(λ) = ', self.logwave[1]-self.logwave[0])

        self.Dv = (speed_light/1000.0)*np.log(10)*(self.logwave[1]-self.logwave[0])
        #Dv = (speed_light/1000.0)*np.log(10)*len(delt[0])*(logwave[1]-logwave[0])
        print('Δv(λ) = ', self.Dv,'km/s')

        ####### Compute P_raw
        
        print('Calculando Pk_raw, Pk_noise y Pk1D')

        self.k1d, self.Pkraw_mean, self.Pkraws, self.Pdelk = pkraw(self.Dv, self.delt)          

        ####### Compute P_noise

        self.Pk_noise_mean, self.Pknoises, self.Dkiv = pknoise(self.Dv, self.weightlyan)

        ####### Resolution correction
        self.win = window(self.Dv,80,self.k1d,'Pk1D')                                  # Resolution spectrograph = 80 km/s

        ####### Compute 1D-PowerSpectrum
        self.Pk1DT = (self.Pkraw_mean - self.Pk_noise_mean)/self.win

        ####### Change deltas by less Deltas

        delmk = self.Pdelk.conjugate()     # Para las deltas de flujo
        delmkiv = self.Dkiv.conjugate()    # Para las deltas de ruido

        ####### Compute B_raw for k1 = k2 y k1 = 2k2
        
        print('Calculando Bk_raw, Bk_random y Bk1D')

        self.kb1, self.kb2, self.Bsp1, self.Bsp2 = Bispec(self.Dv, self.k1d, self.Pdelk, delmk)

        ####### Compute B_random for the same k-vector than B_raw

        self.BspR, self.BspR2 = BispecRandom(self.Dv, self.k1d, self.Pdelk, delmk)

        ####### Compute 1D-Bispectrum

        winB1 = window(self.Dv,80.0,self.kb1,'Bk1D')   
        winB2 = window(self.Dv,80.0,self.kb2,'Bk1D')

        self.Bk1DT  = (self.Bsp1)/winB1       # Para k1 = k2
        self.Bk1DT2 = (self.Bsp2)/winB2       # Para k1 = 2k2
        print('Tiempo total = {} s'.format(time.time() - time1))


#### Bispectrum with Deltas from PICCA ####

class PiccaDeltas:
    def __init__(self,dirfile,pesosD):
        time1 = time.time()
        
        ### regrid wavelength
        CRVAL1 = 2.6534
        CDELT1 = 0.0001
        NBLL   = 13637
        wwave = 10**(CRVAL1+np.arange(NBLL)*CDELT1)
        masw = (wwave > 1050) & (wwave < 1180)
        self.wwave = wwave[masw]
        
        ####### Read deltas from Picca
        
        specfile = sorted(glob(dirfile +'delta*.fits.gz'))
        
        self.deltaF, self.weigthsd = deltas_picca(self.wwave,specfile,pesosD)
        print('Num. BOSS DR16 spectra = ', self.deltaF.shape[0])
        self.deltaFprom = np.average(self.deltaF, weights=self.weigthsd, axis = 0)
        self.deltasM = self.deltaF - self.deltaFprom                                  # Ajusta para que el promedio de las deltas sea 0.
        self.deltasMprom = np.average(self.deltasM, weights=self.weigthsd, axis = 0)

        ####### Tune the wavelength 

        self.logwave = np.log10(self.wwave)
        print('Δlog(λ) = ', self.logwave[1]-self.logwave[0])

        self.Dv = (speed_light/1000.0)*np.log(10)*(self.logwave[1]-self.logwave[0])
        #Dv = (speed_light/1000.0)*np.log(10)*len(delt[0])*(logwave[1]-logwave[0])
        print('Δv(λ) = ', self.Dv,'km/s')

        ####### Compute P_raw
        
        print('Calculando Pk_raw, Pk_noise y Pk1D')

        self.k1d, self.Pkraw_mean, self.Pkraws, self.Pdelk = pkraw(self.Dv, self.deltasM)          

        ####### Compute P_noise

        self.Pk_noise_mean, self.Pknoises, self.Dkiv = pknoise(self.Dv, self.weigthsd)

        ####### Resolution correction
        self.win = window(self.Dv,80,self.k1d,'Pk1D')                                  # Resolution spectrograph = 80 km/s

        ####### Compute 1D-PowerSpectrum
        self.Pk1DT = (self.Pkraw_mean - self.Pk_noise_mean)/self.win

        ####### Change deltas by less Deltas

        delmk = self.Pdelk.conjugate()     # Para las deltas de flujo
        delmkiv = self.Dkiv.conjugate()    # Para las deltas de ruido

        ####### Compute B_raw for k1 = k2 y k1 = 2k2
        
        print('Calculando Bk_raw, Bk_random y Bk1D')

        self.kb1, self.kb2, self.Bsp1, self.Bsp2 = Bispec(self.Dv, self.k1d, self.Pdelk, delmk)

        ####### Compute B_random for the same k-vector than B_raw

        self.BspR, self.BspR2 = BispecRandom(self.Dv, self.k1d, self.Pdelk, delmk)

        ####### Compute 1D-Bispectrum

        winB1 = window(self.Dv,80.0,self.kb1,'Bk1D')   
        winB2 = window(self.Dv,80.0,self.kb2,'Bk1D')

        self.Bk1DT  = (self.Bsp1)/winB1       # Para k1 = k2
        self.Bk1DT2 = (self.Bsp2)/winB2       # Para k1 = 2k2
        print('Tiempo total = {} s'.format(time.time() - time1))