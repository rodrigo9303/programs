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
import healpy
import os


sys.path.append('/global/homes/r/rodcn25/PCA/empca/')
import empca
import argparse

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

def read_cat(ddir, drq,zmin, zmax, magmin, magmax,in_nside=64):
    print('Reading catalog from ', drq)
    catalog = Table(fitsio.read(drq, ext=1))

    keep_columns = ['RA', 'DEC', 'Z']
    
    if 'TARGETID' in catalog.colnames:
        obj_id_name = 'TARGETID'
        if 'TARGET_RA' in catalog.colnames:
            catalog.rename_column('TARGET_RA', 'RA')
            catalog.rename_column('TARGET_DEC', 'DEC')
        keep_columns += ['TARGETID']
        if 'SURVEY' in catalog.colnames:
            keep_columns += ['SURVEY']
        if 'DESI_TARGET' in catalog.colnames:
            keep_columns += ['DESI_TARGET']
    ## Redshift
    if 'Z' not in catalog.colnames:
        if 'Z_VI' in catalog.colnames:
            catalog.rename_column('Z_VI', 'Z')
            print(
                "Z not found (new DRQ >= DRQ14 style), using Z_VI (DRQ <= DRQ12)"
            )
        else:
            print("ERROR: No valid column for redshift found in ",
                      drq_filename)
            return None
        
    ## Sanity checks
    print('')
    w = np.ones(len(catalog), dtype=bool)
    print(f" start                 : nb object in cat = {np.sum(w)}")
    w &= catalog[obj_id_name] > 0
    print(f" and {obj_id_name} > 0      : nb object in cat = {np.sum(w)}")
    w &= catalog['RA'] != catalog['DEC']
    print(f" and ra != dec         : nb object in cat = {np.sum(w)}")
    w &= catalog['RA'] != 0.
    print(f" and ra != 0.          : nb object in cat = {np.sum(w)}")
    w &= catalog['DEC'] != 0.
    print(f" and dec != 0.         : nb object in cat = {np.sum(w)}")

    ## Redshift range
    w &= catalog['Z'] >= zmin
    print(f" and z >= {zmin}          : nb object in cat = {np.sum(w)}")
    w &= catalog['Z'] < zmax
    print(f" and z < {zmax}           : nb object in cat = {np.sum(w)}")
    
    ## Magnitud range
    mag_r = 22.5 - 2.5*np.log10(catalog['FLUX_R'])
    w &= (catalog['Z'] >= zmin) & (catalog['Z'] < zmax) & (mag_r >= magmin) & (mag_r < magmax)
    print(f" and {zmin} < z < {zmax} with {magmin} < mag < {magmax}   : nb object in cat = {np.sum(w)}")
    
    catalog.keep_columns(keep_columns)
    w = np.where(w)[0]
    catalog = catalog[w]

    #-- Convert angles to radians
    catalog['RA'] = np.radians(catalog['RA'])
    catalog['DEC'] = np.radians(catalog['DEC'])
    return catalog

class Forest:
    def __init__(self, specss, zvi, galex):
        m = (specss['waveR'] > np.max(specss['waveB']))
        n = (specss['waveZ'] > np.max(specss['waveR']))
        qsoflux = np.concatenate((specss['fluxB'],specss['fluxR'][m],specss['fluxZ'][n]),axis=None)
        qsoivar = np.concatenate((specss['ivarB'],specss['ivarR'][m],specss['ivarZ'][n]),axis=None)
        qsowave = np.concatenate((specss['waveB'],specss['waveR'][m],specss['waveZ'][n]),axis=None)
        qsowaverf = qsowave/(1.0+zvi)
        
        ### Correct for Lyman
        T = transmission_Lyman(zvi,qsowave)
        qsoflux /= T
        qsoivar *= T**2
        ### Remove Galactic extinction for g, r and z bands
        tunred_g = unred(qsowave,galex)
        qsoflux /= tunred_g
        qsoivar *= tunred_g**2
        
        ### Compute signal to noise ratio
        snrmask     = (qsowaverf > 1050.0) & (qsowaverf < 1180.0)
        meanfluxlya = qsoflux[snrmask].mean()                      # mean flux in the Ly-A forest
        error       = 1.0 / np.sqrt(qsoivar[snrmask][qsoivar[snrmask]>0.0])
        snr         = qsoflux[snrmask][qsoivar[snrmask]>0.0] / error
        #snr         = qsoflux[snrmask][error > 0.0] / error[error > 0.0]
        msnr        = np.average(snr)
        
        self.wave = qsowaverf
        self.flux = qsoflux
        self.ivar = qsoivar
        self.msnr = msnr
        self.zvi  = zvi

def bins_paths(ddir, catt, nniter,nnvec, in_nside=64):
    ## Build the paths
    ra = catt['RA'].data
    dec = catt['DEC'].data
    in_healpixs = healpy.ang2pix(in_nside, np.pi / 2. - dec, ra, nest=True)
    print('Total healpix  = ',len(in_healpixs))
    unique_in_healpixs = np.unique(in_healpixs)
    print('Unique healpix = ',len(unique_in_healpixs))
    id_name = 'TARGETID'
    datas = []
    fluxs = []
    ivars = []
    zz = []
    filenamess = []
    galex = []
    rvextinction = 3.793
    #survey = np.unique(cath['SURVEY'])
    prefix = 'coadd-{}-dark'.format(catt['SURVEY'][0])
    for index, healpix in enumerate(unique_in_healpixs):
        filenamess.append(f"{ddir}/healpix/main/dark/{healpix//100}/{healpix}/{prefix}-{healpix}.fits")
        filenames = f"{ddir}/healpix/main/dark/{healpix//100}/{healpix}/{prefix}-{healpix}.fits"
        ## Read QSO spectra
        try:
            hdul = fitsio.FITS(filenames)
        except IOError:
            continue

        fibermap      = hdul['FIBERMAP'].read()
        targetid_spec = fibermap["TARGETID"]

        spec_data = {}
        colors = ["B", "R"]
        if "Z_FLUX" in hdul:
            colors.append("Z")
        for color in colors:
            spec = {}
            try:
                spec["WL"] = hdul[f"{color}_WAVELENGTH"].read()
                spec["FL"] = hdul[f"{color}_FLUX"].read()
                spec["IV"] = (hdul[f"{color}_IVAR"].read() * (hdul[f"{color}_MASK"].read() == 0))
                w = np.isnan(spec["FL"]) | np.isnan(spec["IV"])
                for key in ["FL", "IV"]:
                    spec[key][w] = 0.
                spec_data[color] = spec
            except OSError:
                print(f"ERROR: while reading {color} band from {filenames}")
        glex = fibermap['EBV']
        hdul.close()
        
        ## Get the quasars in this healpix pixel
        select = np.where(in_healpixs == healpix)[0]
        wwl = []
        ffl = []
        iiv = []
        z = []
        gale = []
        data = []
        #-- Loop over quasars in catalog inside this healpixel
        for entry in catt[select]:
            #-- Find which row in tile contains this quasar
            #-- It should be there by construction
            w_t = np.where(targetid_spec == entry[id_name])[0]
            if len(w_t) == 0:
                print(f"Error reading {entry[id_name]}")
                continue
            elif len(w_t) > 1:
                print(f"Warning: more than one spectrum in this file for {entry[id_name]}")
            else:
                w_t = w_t[0]
            wl = []
            fl = []
            iv = []
            #gal = []
            
            #for spec in spec_data.values():
            colors = ['B', 'R', 'Z']
            specs = {}
            for color in colors:
                specs['wave{}'.format(color)] = spec_data['{}'.format(color)]['WL']
                specs['flux{}'.format(color)] = spec_data['{}'.format(color)]['FL'][w_t]
                specs['ivar{}'.format(color)] = spec_data['{}'.format(color)]['IV'][w_t]
            
            gale1 = glex[w_t]
            gale1 = gale1 / rvextinction
            zz    = entry['Z']

            forest = Forest(specs, zz, gale1)
            data.append(forest)
        datas.append(data)
    datas = np.hstack(datas) 
    
    ## Choose only de Ly-A region for remove bad spectra.
    datos   = []
    snrc    = 0 
    shortFc = 0
    nanc    = 0
    
    for i in range(len(datas)):
        snrmask     = (datas[i].wave > 1050.0) & (datas[i].wave < 1180.0)
        ### Remove spectra due to forest too short
        if len(datas[i].wave[snrmask]) < 50:
            shortFc += 1
            continue
        ### Remove due to NaN found
        if np.isnan((datas[i].flux[snrmask]*datas[i].ivar[snrmask]).sum()):
            nanc += 1
            continue
        ### Remove spectra with low signal to noise ratio or negative mean flux
        meanfluxlya = datas[i].flux[snrmask].mean()                      # mean flux in the Ly-A forest

        if meanfluxlya <= 0.0 or datas[i].msnr <= 1.0: # the cut in snr is for spectra with snr > 2
            snrc += 1
            continue
        else:
            datos.append(datas[i])
    datos = np.hstack(datos)
    
    print('Total spectra with SNR < 1.0     = ', snrc)
    print('Total spectra with short forest  = ', shortFc)
    print('Total spectra with NaN found     = ', nanc)
    ### new wavelenght for compute EMPCA.
    CRVAL1 = 2.6534
    CDELT1 = 0.0001
    NBLL   = 13637
    wwave = 10**(CRVAL1+np.arange(NBLL)*CDELT1)
    ### put flux and ivar with the new wavelenght
    pcafluxm  = np.zeros((len(datos), wwave.size))
    pcaivarm  = np.zeros((len(datos), wwave.size))

    for i in range(len(datos)):
        pcafluxm[i], pcaivarm[i] = resample_flux(wwave,datas[i].wave,datas[i].flux,datas[i].ivar) # interpolation
        
    ### Normalization
    #lmin=1420.0
    #lmax=1500.0
    #sum1 = 0
    #integral_mocks = []
    #for i in range(len(pcaflux)):
    #    if lmin <= pcawave <= lmax:
            
    #mask = (wwave > lmin) & (wwave < lmax)
    #for i in range(len(pcafluxm)):
    #    sum1 = np.sum((pcafluxm[i][mask])*(wwave[1801]-wwave[1800]))
        #integral_mocks.append(sum1)
    #    if sum1==0.0:
    #        continue
    #    else:
    #        pcafluxm[i] = (100.0*pcafluxm[i])/sum1
    #        pcaivarm[i] = (pcaivarm[i])*(((sum1)**2)/100.0**2)
    
    print('Total spectra     = ', len(pcafluxm))
    ###
    pcaivarm[pcaivarm<0.] = 0              #
    w = np.sum(pcaivarm,axis=0)>0.         #
    wwave = wwave[w]                       # For the noise.
    pcafluxm = pcafluxm[:,w]               #
    pcaivarm = pcaivarm[:,w]               #
    pcaivarm[pcaivarm>100.] = 100.         #
    ### Redux sample
    #pcafluxmc = np.copy(pcafluxm)
    #pcaivarmc = np.copy(pcaivarm)
    redux = pcafluxm.shape[0] - (pcafluxm.shape[0]//10)
    np.random.seed(123)
    pcafluxm = pcafluxm[np.random.choice(pcafluxm.shape[0], size = redux, replace=False),:]
    np.random.seed(123)
    pcaivarm = pcaivarm[np.random.choice(pcaivarm.shape[0], size = redux, replace=False),:]
    print('Total spectra reduxed    = ', len(pcafluxm))
    ### Get mean spectrum 
    data_meanspec = np.average(pcafluxm,weights=pcaivarm,axis=0) # average weighted
    nbObj = pcafluxm.shape[0]
    for i in range(nbObj):       #
        w = pcaivarm[i]>0.        # subtracting the mean for each spectrum
        pcafluxm[i,w] -= data_meanspec[w] #

    ### PCA
    print('INFO: Starting EMPCA')
    dmodel = empca.empca(pcafluxm, weights=pcaivarm, niter=nniter, nvec=nnvec)
   
    return dmodel, wwave, pcafluxm, pcaivarm, data_meanspec


def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    description=('Compute the PCA decomposition from QSO spectra\
                                                  for spectra simulated by SIMQSO or QSO or BOSS DR14, DR16 and DESI\                                                         data.'))
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
    #parser.add_argument('--sky_mask', 
    #                    type=str, 
    #                    default=None, 
    #                    required=False, 
    #                    help='Line Sky Mask for remove from BOSS data')
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
    
    modes = ['desi', 'boss14', 'boss16', 'everest']
    mode = args.mode
    clase = args.spectra
    Niterr = int(args.Niter)
    Nvecc = int(args.Nvec)
    if mode in modes:
        if mode == 'everest':
            print('Read Catalog')
            cath = read_cat(args.in_dir, args.catalog, args.zmin, args.zmax, args.Mmin, args.Mmax)
            print('Read spectra from Everest')
            time3 = time.time()
            
            modelpca, waverf, flux, ivar, mean_spec = bins_paths(args.in_dir, cath, Niterr, Nvecc)
            print('Spectra reading time, taking PCA time = ', time.time() - time3, 's')
    ### Read mags and z
    a = np.asarray([16.25, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25, 19.75, 20.25,20.75, 21.25, 21.75, 22.25, 22.75])
    b = np.asarray([1625, 1675, 1725, 1775, 1825, 1875, 1925, 1975, 2025,2075, 2125, 2175, 2225, 2275])
    c = np.asarray([0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]) 
    d = np.asarray([4,6,8,10,12,14,16,18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
    
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

    #colum_data2=[('flux_spectra','f8'),('weights','f8'),('resolution','f8')]
    #data2=Table(np.zeros((flux.shape[0],len(waverf)),dtype=colum_data2))
    #data2['flux_spectra'][:][:] = flux
    #data2['weights'][:][:]      = ivar
    #data2['resolution'][:][:]   = reso
    
    colum_data2=[('flux_spectra','f8'),('weights','f8')]
    data2=Table(np.zeros((flux.shape[0],len(waverf)),dtype=colum_data2))
    data2['flux_spectra'][:][:] = flux
    data2['weights'][:][:]      = ivar

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
    
    #colum_data7=[('fibersid','f8')]
    #data7=Table(np.zeros(len(fibers),dtype=colum_data7))
    #data7['fibersid'][:] = fibers

    hdata1 = fits.convenience.table_to_hdu(data1); hdata1.name='wl_mean'
    hdata2 = fits.convenience.table_to_hdu(data2); hdata2.name='flux_ivar'
    hdata3 = fits.convenience.table_to_hdu(data3); hdata3.name='eigvec'
    hdata4 = fits.convenience.table_to_hdu(data4); hdata4.name='coeff'
    hdata5 = fits.convenience.table_to_hdu(data5); hdata5.name='eigvar_eigvarT'
    hdata6 = fits.convenience.table_to_hdu(data6); hdata6.name='converg'
    #hdata7 = fits.convenience.table_to_hdu(data7); hdata7.name='fiberids'

    #hdulist = fits.HDUList([fits.PrimaryHDU(),hdata1,hdata2,hdata3,hdata4,hdata5,hdata6,hdata7])
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