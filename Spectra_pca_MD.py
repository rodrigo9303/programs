#!/usr/bin/env python

import numpy as np
import time
import fitsio
import matplotlib.pyplot as plt
from desispec.interpolation import resample_flux
import scipy.interpolate as interpolate
import desispec.io
import sys
from astropy.io import fits
from astropy.table import Table,Column
from glob import glob

sys.path.append('/global/homes/r/rodcn25/PCA/empca/')
import empca
import argparse

############ MOCKS DATA ###############
def get_mock_spectra(dir_file, mmin,mmax,zmin,zmax,pesos):
    mflux = []
    mivar = []
    mwave = []
    for i in range(len(dir_file)): #Loop for read all pixel. [:1] mean that read only the first.
        specfilename = dir_file[i] #choose a pixel to be read.
        specobj= desispec.io.read_spectra(specfilename)
        ### Magnitude
        DM = fitsio.FITS(specfilename)
        MAG_R = 22.5 - 2.5*np.log10(DM[1]['FLUX_R'].read())
        ### Redshifts
        zfilename = specfilename.replace('spectra-16-', 'zbest-16-') 
        zs = fitsio.read(zfilename) # zbest file.
        ### Bins in Mag and Z
        mask = (MAG_R > mmin) & (MAG_R < mmax) & (zs['Z'] > zmin) & (zs['Z'] < zmax)
        zvi = zs['Z'][mask]
        wavem = specobj.wave
        fluxm = specobj.flux
        if pesos == 'ivar':
            ivarm = specobj.ivar
        elif pesos == 'constants':
            ivarm = {'b': np.ones_like(specobj.ivar['b']), 'r': np.ones_like(specobj.ivar['r']), 'z': np.ones_like(specobj.ivar['z'])}
            
        fluxm['b'] = fluxm['b'][mask]
        fluxm['r'] = fluxm['r'][mask]
        fluxm['z'] = fluxm['z'][mask]
        
        ivarm['b'] = ivarm['b'][mask]
        ivarm['r'] = ivarm['r'][mask]
        ivarm['z'] = ivarm['z'][mask]
        
        if len(fluxm['b']) & len(fluxm['r']) & len(fluxm['z']) == 0:
            continue
        else:
            qsoivars = []
            qsofluxs = []
            qsowaves = []
            for j in range(len(fluxm['b'])):
                ### Put spectra in the rest-frame
                z = zvi[j]
                brf = wavem['b']/(1+z)
                rrf = wavem['r']/(1+z)
                zrf = wavem['z']/(1+z)
                ### mask wavelength
                m = (rrf > np.max(brf))
                n = (zrf > np.max(rrf))

                ### remove the colorcut
                qsoflux = np.concatenate((fluxm['b'][j],fluxm['r'][j][m],fluxm['z'][j][n]),axis=None)
                qsoivar = np.concatenate((ivarm['b'][j],ivarm['r'][j][m],ivarm['z'][j][n]),axis=None)
                qsowave = np.concatenate((brf,rrf[m],zrf[n]),axis=None)
                qsofluxs.append(qsoflux)
                qsoivars.append(qsoivar)
                qsowaves.append(qsowave)

        mflux.append(qsofluxs)
        mivar.append(qsoivars)
        mwave.append(qsowaves)
        
    wavelength = np.vstack(mwave)
    qsofluxss = np.vstack(mflux)
    qsoivarss = np.vstack(mivar)
    print('INFO: size (total spectra)= ', qsofluxss.shape[0])
    ### new wavelenght for compute EMPCA.
    CRVAL1 = 2.6534
    CDELT1 = 0.0001
    NBLL   = 13637
    wwave = 10**(CRVAL1+np.arange(NBLL)*CDELT1)
    ### put flux and ivar with the new wavelenght
    pcafluxm  = np.zeros((len(qsofluxss), wwave.size))
    pcaivarm  = np.zeros((len(qsofluxss), wwave.size))

    for i in range(len(qsofluxss)):
        pcafluxm[i], pcaivarm[i] = resample_flux(wwave,wavelength[i],qsofluxss[i],qsoivarss[i]) # interpolation
        
    ### Normalize the spectra
    #lmin=1420.0
    #lmax=1500.0
    #integral_mocks = []
    #mask = (wwave > lmin) & (wwave < lmax)
    #for i in range(len(pcafluxm)):
    #    sum1 = np.sum((pcafluxm[i][mask])*(wwave[1001]-wwave[1000]))
    #    #integral_mocks.append(sum1)
    #    pcafluxm[i] = (53*pcafluxm[i])/sum1
    #    pcaivarm[i] = (pcaivarm[i])*(((sum1)**2)/53**2)
        
    #int_mean = np.mean(integral_mocks)
    #pcafluxm = pcafluxm*25
    #pcaivarm = pcaivarm/(25**2)
    ###
    pcaivarm[pcaivarm<0.] = 0              #
    w = np.sum(pcaivarm,axis=0)>0.         #
    wwave = wwave[w]                       # For the noise.
    pcafluxm = pcafluxm[:,w]               #
    pcaivarm = pcaivarm[:,w]               #
    pcaivarm[pcaivarm>100.] = 100.         #
    ### Get mean spectrum 
    mock_mean_spec = np.average(pcafluxm,weights=pcaivarm,axis=0) # average weighted 
    #std = np.std(pcafluxm,axis=0)        # It's not necessary, because the weighted average isn't use.
    #for i in range(len(pcafluxm)):        #
    #    w = pcaivarm[i]>0.                 # subtracting the mean for each spectrum
    #    pcafluxm[i,w] -= mock_mean_spec[w] # Now, the flux = flux - mean. 
    ### Get standard deviation
    #vari = np.zeros(wwave.size)   
    #for h in range(len(pcafluxm)):
    #    vari += pcafluxm[h]**2
    #varianza = np.sqrt(vari/(len(pcafluxm)-1)) 
        
    return wwave, pcafluxm, pcaivarm, mock_mean_spec

############### BOSS DR14 DATA ###################

# Lyman-alpha from eqn 5 of Calura et al. 2012 (Arxiv: 1201.5121)
# Other from eqn 1.1 of Irsic et al. 2013 , (Arxiv: 1307.3403)
# Lyman-limit from abstract of Worseck et al. 2014 (Arxiv: 1402.4154)
Lyman_series = {
    'Lya'     : { 'line':1215.67,  'A':0.0023,          'B':3.64, 'var_evol':3.8 },
    'Lyb'     : { 'line':1025.72,  'A':0.0023/5.2615,   'B':3.64, 'var_evol':3.8 },
    'Ly3'     : { 'line':972.537,  'A':0.0023/14.356,   'B':3.64, 'var_evol':3.8 },
    'Ly4'     : { 'line':949.7431, 'A':0.0023/29.85984, 'B':3.64, 'var_evol':3.8 },
    'Ly5'     : { 'line':937.8035, 'A':0.0023/53.36202, 'B':3.64, 'var_evol':3.8 },
    #'LyLimit' : { 'line':911.8,    'A':0.0023,          'B':3.64, 'var_evol':3.8 },
}

def transmission_Lyman(zObj,lObs):
    '''Calculate the transmitted flux fraction from the Lyman series
    This returns the transmitted flux fraction:
        1 -> everything is transmitted (medium is transparent)
        0 -> nothing is transmitted (medium is opaque)
    Args:
        zObj (float): Redshift of object
        lObs (array of float): wavelength grid
    Returns:
        array of float: transmitted flux fraction
    '''

    lRF = lObs/(1.+zObj)
    T   = np.ones(lObs.size)

    for l in list(Lyman_series.keys()):
        w      = lRF<Lyman_series[l]['line']
        zpix   = lObs[w]/Lyman_series[l]['line']-1.
        tauEff = Lyman_series[l]['A']*(1.+zpix)**Lyman_series[l]['B']
        T[w]  *= np.exp(-tauEff)

    return T

def lines_list(path):

    lines = []
    fileLines = open(path)
    for l in fileLines:
        l = l.split()
        if l[0]=='#': continue
        lines += [ [float(l[1]),float(l[2])] ]
    fileLines.close()
    lines = np.asarray(lines)

    return lines

def unred(wave, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
    '''
    https://github.com/sczesla/PyAstronomy
    in /src/pyasl/asl/unred
    '''

    x = 10000./wave # Convert to inverse microns
    curve = x*0.

    # Set some standard values:
    x0 = 4.596
    gamma = 0.99
    c3 = 3.23
    c4 = 0.41
    c2 = -0.824 + 4.717/R_V
    c1 = 2.030 - 3.007*c2

    if LMC2:
        x0    =  4.626
        gamma =  1.05
        c4   =  0.42
        c3    =  1.92
        c2    = 1.31
        c1    =  -2.16
    elif AVGLMC:
        x0 = 4.596
        gamma = 0.91
        c4   =  0.64
        c3    =  2.73
        c2    = 1.11
        c1    =  -1.28

    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and
    # R-dependent coefficients
    xcutuv = np.array([10000.0/2700.0])
    xspluv = 10000.0/np.array([2700.0,2600.0])

    iuv = np.where(x >= xcutuv)[0]
    N_UV = iuv.size
    iopir = np.where(x < xcutuv)[0]
    Nopir = iopir.size
    if N_UV>0:
        xuv = np.concatenate((xspluv,x[iuv]))
    else:
        xuv = xspluv

    yuv = c1 + c2*xuv
    yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
    yuv = yuv + c4*(0.5392*(np.maximum(xuv,5.9)-5.9)**2+0.05644*(np.maximum(xuv,5.9)-5.9)**3)
    yuv = yuv + R_V
    yspluv = yuv[0:2]  # save spline points

    if N_UV>0:
        curve[iuv] = yuv[2::] # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR
    xsplopir = np.concatenate(([0],10000.0/np.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])))
    ysplir = np.array([0.0,0.26469,0.82925])*R_V/3.1
    ysplop = np.array((np.polyval([-4.22809e-01, 1.00270, 2.13572e-04][::-1],R_V ),
            np.polyval([-5.13540e-02, 1.00216, -7.35778e-05][::-1],R_V ),
            np.polyval([ 7.00127e-01, 1.00184, -3.32598e-05][::-1],R_V ),
            np.polyval([ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1],R_V ) ))
    ysplopir = np.concatenate((ysplir,ysplop))

    if Nopir>0:
        tck = interpolate.splrep(np.concatenate((xsplopir,xspluv)),np.concatenate((ysplopir,yspluv)),s=0)
        curve[iopir] = interpolate.splev(x[iopir], tck)

    #Now apply extinction correction to input flux vector
    curve *= ebv
    corr = 1./(10.**(0.4*curve))

    return corr

def plot_spplate(path_to_data,plate,mjd,fiber):

    ###
    path = path_to_data+str(plate)+'/spPlate-' +str(plate)+'-'+str(mjd)+'.fits'

    cat = fitsio.FITS(path)
    psflux = cat[0].read()
    psivar = cat[1].read()
    end  = cat[2].read()
    h    = cat[4].read_header()
    fid = cat[5].read()
    cat.close()

    pswave = h['CRVAL1'] + h['CD1_1']*np.arange(h['NAXIS1'])
    if h['DC-FLAG']:
        pswave = 10**pswave

    ###
    cut = (psivar[fiber-1,:]>0.) & (end[fiber-1,:]==0)
    pswave = pswave[cut]
    psflux = psflux[fiber-1,:][cut]
    psivar = psivar[fiber-1,:][cut]
    #qsoid = np.where(fid['OBJTYPE'][:] == b'QSO             ')[0]#I only choose the QSO's. 
    #if (fiber-1) in qsoid:
    #    pswave = pswave[cut]
    #    psflux = psflux[fiber-1,:][cut]
    #    psivar = psivar[fiber-1,:][cut]
    #else: 
        #print('No QSO')
    #    pswave = np.array([0])
    #    psflux = np.array([0])
    #    psivar = np.array([0])

    return pswave, psflux, psivar

def get_pca(dir_data, catalog_data, sky_lines, dmmin,dmmax,dzmin,dzmax,nniter,nnvec,vboss,ppesos):
                              #'/project/projectdirs/cosmo/data/sdss/dr16/eboss/spectro/redux/v5_13_0/'
    ### Parameters
    path_spec      = dir_data #'/project/projectdirs/cosmo/data/sdss/dr14/eboss/spectro/redux/v5_10_0/'
    path_drq       = catalog_data #'/global/homes/r/rodcn25/PCA/DR14Q_v4_4.fits'
    path_lines     = sky_lines #'/global/homes/r/rodcn25/PCA/dr14-line-sky-mask.txt'
    nbObj  = 10000
    CRVAL1 = 2.6534
    CDELT1 = 0.0001
    NBLL   = 13637

    ### Get lines to veto
    lines = lines_list(path_lines)

    ### Get list qso
    data      = fitsio.FITS(path_drq)
    w  = np.ones(data[1]['PLATE'][:].size).astype(bool)
    print('INFO: init              : ', w.sum())
    w &= data[1]['THING_ID'][:]>0.
    print('INFO: removed THID>=0   : ', w.sum())
    #print('INFO: removed THID<=0   : ', w.sum())
    w &= data[1]['Z_PCA'][:]>0.
    print('INFO: removed zvi>=0.   : ', w.sum())
    #print('INFO: removed zvi<=0.   : ', w.sum())
    plate   = data[1]['PLATE'][:][w]
    mjd     = data[1]['MJD'][:][w]
    fiberid = data[1]['FIBERID'][:][w]
    zvi     = data[1]['Z_PCA'][:][w]
    AMag_r  = data[1]['PSFMAG'][:][:,2][w] # choose the 'r-band'  
    rvextinction=3.793
    if vboss == 'dr14':
        G_EXTINCTION = data[1]['GAL_EXT'][:][:,1]/rvextinction # Galactic extinction for the g band.
    elif vboss == 'dr16':
        G_EXTINCTION = data[1]['EXTINCTION'][:][:,1]/rvextinction
    extr = G_EXTINCTION[w]
    data.close()
    ### BINS
    maskd = (AMag_r>dmmin) & (AMag_r<dmmax) & (zvi>dzmin) & (zvi<dzmax)
    plate   = plate[maskd]
    mjd     = mjd[maskd]
    fiberid = fiberid[maskd]
    zvi     = zvi[maskd]
    AMag_r  = AMag_r[maskd]
    extr    = extr[maskd]
    ### Random number of object
    nsamples=31200
    np.random.seed(10)
    listQSO = np.arange(plate.size)
    randSelec = np.array([])
    for i in np.arange(0.,10.,0.1):
        w = (zvi>=i) & (zvi<i+0.1)
        if listQSO[w].size==0: continue
        r = np.random.choice(listQSO[w], size=min(listQSO[w].size,nsamples), replace=False)
        randSelec = np.append(randSelec,r)
    #randSelec = sp.append(randSelec,sp.random.choice(listQSO, size=nbObj, replace=False))
    randSelec = np.unique(randSelec)
    randSelec = randSelec.astype(int)
    nbObj = randSelec.size
    print('INFO: size (inside bin)= ', nbObj)

    ### Get spectra
    rrwave = []
    rrflux = []
    rrivar = []
    for i in range(nbObj): 
        #if i%10==0: print(i)
        r = randSelec[i]
        try:
            TMPwave, TMPflux, TMPivar = plot_spplate(path_spec,plate=plate[r],mjd=mjd[r],fiber=fiberid[r])
        except IOError:
            continue
        if len(TMPflux) & len(TMPivar) & len(TMPwave) == 0:
            continue
        else:
            if ppesos == 'ivar':
                TMPivar = TMPivar
            elif ppesos == 'constants':
                TMPivar = np.ones_like(TMPivar)
            TMPwaveRF = TMPwave/(1.+zvi[r])
            if TMPwave.size<500:
                #print('INFO: removing size<10: {}'.format(TMPwave.size))
                continue        
            ### Remove sky lines and CCD edge
            w = np.ones_like(TMPwave).astype(bool)
            for l in lines:
                w &= (TMPwave<l[0]) | (TMPwave>l[1])
            w &= TMPwave>3550.0                #3600. Dedault from Helion code. The new values are by DESI
            w &= TMPwave<10000.0               #7235. The real coverage is 3650-10,400 Å.
                                               #See https://www.sdss.org/dr14/spectro/spectro_basics/
            TMPwave = TMPwave[w]
            TMPflux = TMPflux[w]
            TMPivar = TMPivar[w]
            TMPwaveRF = TMPwaveRF[w]
            if TMPwave.size<500:
                #print('INFO: removing size<10: {}'.format(TMPwave.size))
                continue

            ### Correct for Lyman
            T = transmission_Lyman(zvi[r],TMPwave)
            TMPflux /= T
            TMPivar *= T**2
            ### Remove Galactic extinction for g, r and z bands
            tunred_g = unred(TMPwave,extr[r])
            TMPflux /= tunred_g
            TMPivar *= tunred_g**2
            
            ### Normalize
            #lmin=1420.0
            #lmax=1500.0
            #sum1=0
            #for j in range(len(TMPflux)):
            #    if lmin <= TMPwaveRF[j] < lmax :
            #        sum1+=(TMPflux[j])*(TMPwaveRF[j+1]-TMPwaveRF[j])
            #    elif (TMPwaveRF[j] > lmax):
            #        break
            #if sum1 == 0:
            #    continue
            #else:
            #    TMPflux/=(sum1/53.0)
            #    TMPivar*=(sum1/53.0)**2
            
            ### Store
            rrwave += [TMPwaveRF]
            rrflux += [TMPflux]
            rrivar += [TMPivar]
    
    nbObj = len(rrwave)
    print('INFO: size (total spectra)= ', nbObj)

    ###
    pcawave    = 10**(CRVAL1+np.arange(NBLL)*CDELT1)
    pcaflux    = np.zeros((nbObj, pcawave.size))
    pcaivar    = np.zeros((nbObj, pcawave.size))
    ### On same grid
    for i in range(nbObj):
        pcaflux[i],pcaivar[i] = resample_flux(pcawave, rrwave[i], rrflux[i], rrivar[i])
        
    ### Normalize the spectra
    #lmin=1420.0
    #lmax=1500.0
    #sum1 = 0
    #integral_mocks = []
    #for i in range(len(pcaflux)):
    #    if lmin <= pcawave <= lmax:
            
    #mask = (pcawave > lmin) & (pcawave < lmax)
    #for i in range(len(pcaflux)):
    #    sum1 = np.sum((pcaflux[i][mask])*(pcawave[1801]-pcawave[1800]))
        #integral_mocks.append(sum1)
   #     if sum1==0:
   #         continue
   #     else:
   #         pcaflux[i] = (53*pcaflux[i])/sum1
   #         pcaivar[i] = (pcaivar[i])*(((sum1)**2)/53**2)
        
    ### Remove if all measured bins are zero
    pcaivar[pcaivar<0.] = 0.
    w    = np.sum(pcaivar,axis=0)>0.
    pcawave = pcawave[w]
    pcaflux = pcaflux[:,w]
    pcaivar = pcaivar[:,w]
    ### Cap the ivar
    pcaivar[pcaivar>100.] = 100.

    ### Get the mean
    data_meanspec = np.average(pcaflux,weights=pcaivar,axis=0) # Here, I get the mean spectrum.
    for i in range(nbObj):       #
        w = pcaivar[i]>0.        # subtracting the mean for each spectrum
        pcaflux[i,w] -= data_meanspec[w] #

    ### PCA
    print('INFO: Starting EMPCA')
    dmodel = empca.empca(pcaflux, weights=pcaivar, niter=nniter, nvec=nnvec)

    return dmodel, pcawave, pcaflux, pcaivar, data_meanspec

##################################################

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    description=('Compute the PCA decomposition from QSO spectra\
                                                  for spectra simulated by SIMQSO or QSO and by BOSS DR14 data.'))
    parser.add_argument('--in_dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Directory to spectra files')
    parser.add_argument('--catalog', 
                        type=str, 
                        default=None, 
                        required=False, 
                        help='Catalog of objects for BOSS DR14 or DR16 data')
    parser.add_argument('--sky_mask', 
                        type=str, 
                        default=None, 
                        required=False, 
                        help='Line Sky Mask for remove from BOSS data')
    parser.add_argument('--out_dir', 
                        type=str, 
                        default=None, 
                        required=True, 
                        help='Output directory')
    parser.add_argument('--mode', 
                        type=str, 
                        default='pix', 
                        required=True, 
                        help='Open mode of the spectra files: DESI, BOSS DR14 or DR16')
    parser.add_argument('--spectra', 
                        type=str, 
                        default='SIMQSO', 
                        required=True, 
                        help='The spectra can be SIMQSO, QSO or BOSS_DR14, BOSS_DR16')
    parser.add_argument('--use_weights', 
                        type=str, 
                        default='ivar', 
                        required=True, 
                        help='Use ivar as weights or use weights to one (constants)')
    parser.add_argument('--zmin', 
                        type=float,
                        default=1.8,
                        required=True,
                        help="Min redshift")
    parser.add_argument('--zmax', 
                        type=float,
                        default=3.8,
                        required=True,
                        help="Max redshift")
    parser.add_argument('--Mmin', 
                        type=float,
                        default=16.25,
                        required=True,
                        help='Min magnitude in r band')
    parser.add_argument('--Mmax', 
                        type=float,
                        default=22.75,
                        required=True,
                        help='Max magnitude in r band')
    parser.add_argument('--Niter', 
                        type=float,
                        default=10,
                        required=True,
                        help='Number of iteration for PCA')
    parser.add_argument('--Nvec', 
                        type=float,
                        default=10,
                        required=True,
                        help='Number of eigenvectors for PCA')
    
    args = parser.parse_args()
    
    modes = ['desi', 'boss14', 'boss16']
    mode = args.mode
    pesos = ['ivar', 'constants']
    pesoss = args.use_weights
    clase = args.spectra
    Niterr = int(args.Niter)
    Nvecc = int(args.Nvec)
    if mode in modes:
        if mode == 'desi':
            print('Read spectra from Mocks')
            time1 = time.time()
            specfile = sorted(glob(args.in_dir+'spectra-16/*/*/spectra*.fits')) # Read all files spectra.fits
            if pesoss == 'ivar':
                waverf, flux, ivar, mean_spec = get_mock_spectra(specfile,args.Mmin,args.Mmax,args.zmin,args.zmax,'ivar')
            if pesoss == 'constants':
                waverf, flux, ivar, mean_spec = get_mock_spectra(specfile,args.Mmin,args.Mmax,args.zmin,args.zmax,'constants')
            ### For MOCKS
            for i in range(len(flux)):      #
                w = ivar[i]>0.              # subtracting the mean for each spectrum
                flux[i,w] -= mean_spec[w]
            print('Spectra reading time = ', time.time() - time1, 's')
            ### Make PCA decomposition
            print('Begin PCA')
            time2 = time.time()
            modelpca = empca.empca(flux, weights=ivar, niter=Niterr, nvec=Nvecc)
            print('PCA time = ', time.time() - time2, 's')
        elif mode == 'boss14':
            print('Read spectra from BOSS')
            time3 = time.time()
            modelpca, waverf, flux, ivar, mean_spec = get_pca(args.in_dir, args.catalog, args.sky_mask,
                                                              args.Mmin,args.Mmax,args.zmin,args.zmax,
                                                              Niterr, Nvecc,'dr14')
            print('Spectra reading time, taking PCA time = ', time.time() - time3, 's')
        elif mode == 'boss16':
            print('Read spectra from BOSS')
            time3 = time.time()
            if pesoss in pesos:
                if pesoss == 'ivar':
                    modelpca, waverf, flux, ivar, mean_spec = get_pca(args.in_dir, args.catalog, args.sky_mask,
                                                                  args.Mmin,args.Mmax,args.zmin,args.zmax,
                                                                  Niterr, Nvecc,'dr16','ivar')
                elif pesoss == 'constants':
                    modelpca, waverf, flux, ivar, mean_spec = get_pca(args.in_dir, args.catalog, args.sky_mask,
                                                                  args.Mmin,args.Mmax,args.zmin,args.zmax,
                                                                  Niterr, Nvecc,'dr16','constants')
            print('Spectra reading time, taking PCA time = ', time.time() - time3, 's')
            
    ### Read mags and z
    a = np.asarray([16.25, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25, 19.75, 20.25,20.75, 21.25, 21.75, 22.25, 22.75])
    b = np.asarray([1625, 1675, 1725, 1775, 1825, 1875, 1925, 1975, 2025,2075, 2125, 2175, 2225, 2275])
    c = np.asarray([1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]) 
    d = np.asarray([18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
    
    ### Read variance from PCA
    # Eigvectors PCA
    eigvecs = np.zeros(Nvecc)
    for i in range(Nvecc):
        eigvecs[i] = modelpca.R2vec(i)
    # Variance from PCA
    eigenvari = np.zeros(Nvecc + 1)
    for i in range(Nvecc + 1):
        eigenvari[i] = modelpca.R2(i)
        
    ### Save Data
    print('Save data')
    colum_data1 = [('wavelength','f8'),('mean_spectrum','f8')]
    data1 = Table(np.zeros(len(waverf),dtype=colum_data1))
    data1['wavelength'][:]=waverf
    data1['mean_spectrum'][:]=mean_spec

    colum_data2=[('flux_spectra','f8'),('weights','f8')]
    data2=Table(np.zeros((flux.shape[0],len(waverf)),dtype=colum_data2))
    data2=Table(np.zeros((ivar.shape[0],len(waverf)),dtype=colum_data2))
    data2['flux_spectra'][:][:]=flux
    data2['weights'][:][:]=ivar

    colum_data3=[('eigvec','f8')]
    data3=Table(np.zeros((modelpca.eigvec.shape[0],modelpca.eigvec.shape[1]),dtype=colum_data3))
    data3['eigvec'][:][:]=modelpca.eigvec

    colum_data4=[('coeff','f8')]
    data4=Table(np.zeros((modelpca.coeff.shape[0],modelpca.coeff.shape[1]),dtype=colum_data4))
    data4['coeff'][:][:]=modelpca.coeff

    colum_data5=[('eigvar','f8'), ('eigvarT','f8')]
    data5=Table(np.zeros((eigvecs.size),dtype=colum_data5))
    data5['eigvar'][:]=eigvecs
    data5['eigvarT'][:]=eigenvari[-10:]

    colum_data6=[('rchi2','f8')]
    data6=Table(np.zeros((1),dtype=colum_data6))
    data6['rchi2'][:]=modelpca.rchi2()

    hdata1 = fits.convenience.table_to_hdu(data1); hdata1.name='wl_mean'
    hdata2 = fits.convenience.table_to_hdu(data2); hdata2.name='flux_ivar'
    hdata3 = fits.convenience.table_to_hdu(data3); hdata3.name='eigvec'
    hdata4 = fits.convenience.table_to_hdu(data4); hdata4.name='coeff'
    hdata5 = fits.convenience.table_to_hdu(data5); hdata5.name='eigvar_eigvarT'
    hdata6 = fits.convenience.table_to_hdu(data6); hdata6.name='converg'

    hdulist = fits.HDUList([fits.PrimaryHDU(),hdata1,hdata2,hdata3,hdata4,hdata5,hdata6])
    magmin = float(args.Mmin)
    magmax = float(args.Mmax)
    minz = float(args.zmin)
    maxz = float(args.zmax)
    hdulist.writeto(args.out_dir + '{}-{}-{}-{}-{}.fits'.format(clase,b[a==magmin][0],b[a==magmax][0], d[c==minz][0], d[c==maxz][0]), overwrite=True)
    #hdulist.writeto(args.out_dir + '{}-{}-{}-{}-{}.fits'.format(args.class, b[a==args.Mmin],b[a==args.Mmax], d[c==args.zmin], d[c==args.zmax]), overwrite=True)
    
if __name__ == '__main__':
    time_complete = time.time()
    main()
    print('Complete time = ', time.time()-time_complete, 's')