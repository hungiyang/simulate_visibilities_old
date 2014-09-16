import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import healpy as hp
import scipy.special as ssp
import matplotlib.pyplot as plt
import scipy.linalg as la
import os
from array import array

#############################################
#spherical special functions
##############################################
#def sphj(l,z):
    #return ssp.sph_jn(l,z)[0][-1]

def spheh(l,m,theta,phi):
    #return ssp.sph_harm(m,l,phi,theta)
    return sv.spharm(l,m,theta,phi)

################################################
##import the map produced by mathematica, use healpy to turn it into alm
#################################################
#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/nside=8_freq=110_yy.bin') as f:
    #farray = array('f')
    #farray.fromstring(f.read())
    #data = np.array(farray)

#data = data.flatten()
#beam_alm = hp.sphtfunc.map2alm(data,iter=10)  

#nside = 8
#Blm={}
#for l in range(3*nside-1):
    #for mm in range(-l,l+1):
        #if mm >= 0:
            #Blm[(l,mm)] = beam_alm[hp.sphtfunc.Alm.getidx(3*nside-1,l,abs(mm))]
        #if mm < 0:
            #Blm[(l,mm)] = (-1.0)**mm*np.conj(beam_alm[hp.sphtfunc.Alm.getidx(3*nside-1,l,abs(mm))])

##############################################
##import the directions of the healpixmap
##############################################
#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/healpix8_theta_phi.txt') as f:
    #hpdirection = np.array([np.array([float(x) for x in line.split()]) for line in f])
#hpdirection = hpdirection.flatten()
#hpdirection = np.reshape(hpdirection,(len(hpdirection)/2.0,2))

########################################
##create the projection of beam map up to l=20
############################################
#spherical_beam = np.zeros(len(hpdirection),'complex')
#for key in Blm:
    #for i in range(len(hpdirection)):
        #spherical_beam[i] += Blm[key]*spheh(key[0],key[1],hpdirection[i,0],hpdirection[i,1])

#############################################
##calculate the difference between the actual beam and the one recovered from spherical harmonics
############################################
#diff= np.zeros(len(data))
#for i in range(len(data)):
    #diff[i] = abs(data[i]-spherical_beam[i])

#print la.norm(diff)/la.norm(data)
#print la.norm(diff)/(12*nside**2)**0.5/max(data)
#print max(abs(diff))/max(data)

##################################
#plotting
#####################################
#hp.visufunc.mollview(data)
##hp.visufunc.mollview(spherical_beam)
#plt.show()

##########################################################
##########################################################
##########################################################
##########################################################


#####################################
##PAPER beam
######################################
#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/doc/PAPER_bm_120_180_x_nside4.bin') as f:
    #farray = array('f')
    #farray.fromstring(f.read())
    #rawdata = np.array(farray)
    
#nside = 4
#data = np.reshape(rawdata,(7,len(rawdata)/7))[4]

#data = data.flatten()
#beam_alm = hp.sphtfunc.map2alm(data,iter=10)  


#Blm={}
#for l in range(3*nside-1):
    #for mm in range(-l,l+1):
        #if mm >= 0:
            #Blm[(l,mm)] = beam_alm[hp.sphtfunc.Alm.getidx(3*nside-1,l,abs(mm))]
        #if mm < 0:
            #Blm[(l,mm)] = (-1.0)**mm*np.conj(beam_alm[hp.sphtfunc.Alm.getidx(3*nside-1,l,abs(mm))])

##import the directions of the healpixmap
#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/healpix4_theta_phi.txt') as f:
    #hpdirection = np.array([np.array([float(x) for x in line.split()]) for line in f])
#hpdirection = hpdirection.flatten()
#hpdirection = np.reshape(hpdirection,(len(hpdirection)/2.0,2))

##create the projection of beam map up to l=20
#spherical_beam = np.zeros(len(hpdirection),'complex')
#for key in Blm:
    #for i in range(len(hpdirection)):
        #spherical_beam[i] += Blm[key]*spheh(key[0],key[1],hpdirection[i,0],hpdirection[i,1])

##calculate the difference between the actual beam and the one recovered from spherical harmonics
#diff= np.zeros(len(data))
#for i in range(len(data)):
    #diff[i] = abs(data[i]-spherical_beam[i])

##print la.norm(diff)/la.norm(data)
##print la.norm(diff)/(12*nside**2)**0.5/max(data)
##print max(abs(diff))/max(data)
    



##################################################
##compare beam of diffirent nsides
#############################################
#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/nside4freq150_xx.bin') as f:
    #farray = array('f')
    #farray.fromstring(f.read())
    #rawdata = np.array(farray)

#nside = 4
#data = rawdata.flatten()

##compute Blm
#beam_alm = hp.sphtfunc.map2alm(data,iter=10)  
#Blm={}
#for l in range(3*nside-1):
    #for mm in range(-l,l+1):
        #if mm >= 0:
            #Blm[(l,mm)] = beam_alm[hp.sphtfunc.Alm.getidx(3*nside-1,l,abs(mm))]
        #if mm < 0:
            #Blm[(l,mm)] = (-1.0)**mm*np.conj(beam_alm[hp.sphtfunc.Alm.getidx(3*nside-1,l,abs(mm))])

##import the directions of the healpixmap
#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/healpix16_theta_phi.txt') as f:
    #hpdirection = np.array([np.array([float(x) for x in line.split()]) for line in f])
#hpdirection = hpdirection.flatten()
#hpdirection = np.reshape(hpdirection,(len(hpdirection)/2.0,2))


##create the projection of beam map up to l = 3nside-1
#spherical_beam = np.zeros(len(hpdirection),'complex')
#for key in Blm:
    #for i in range(len(hpdirection)):
        #spherical_beam[i] += Blm[key]*spheh(key[0],key[1],hpdirection[i,0],hpdirection[i,1])


##import the nside = 512 healpix beam
#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/nside16freq150_xx.bin') as f:
    #farray = array('f')
    #farray.fromstring(f.read())
    #rawdata = np.array(farray)
#data512 = rawdata.flatten()

##calculate the difference between the actual beam and the one recovered from spherical harmonics
#diff= np.zeros(len(data))
#for i in range(len(data)):
    #diff[i] = abs(data512[i]-spherical_beam[i])

#print la.norm(diff)/la.norm(data512)
#print la.norm(diff)/(12*nside**2)**0.5/max(data512)
#print max(abs(diff))/max(data512)



#######################################
#function to compare some nside to a reference nside(say 128)
#####################################
def compare_beam(nside, reference=16):
    directory = os.path.dirname(os.path.abspath(__file__))+'/'
    filename = 'nside'+str(nside)+'freq150_xx.bin'
    with open(directory+filename) as f:
        farray = array('f')
        farray.fromstring(f.read())
        rawdata = np.array(farray)

    #nside = 4
    data = rawdata.flatten()

    #compute Blm
    beam_alm = hp.sphtfunc.map2alm(data,iter=10)  
    Blm={}
    for l in range(3*nside-1):
        for mm in range(-l,l+1):
            if mm >= 0:
                Blm[(l,mm)] = beam_alm[hp.sphtfunc.Alm.getidx(3*nside-1,l,abs(mm))]
            if mm < 0:
                Blm[(l,mm)] = (-1.0)**mm*np.conj(beam_alm[hp.sphtfunc.Alm.getidx(3*nside-1,l,abs(mm))])
    
    directionfile = 'healpix' + str(reference) + '_theta_phi.txt'
    #import the directions of the healpixmap
    with open(directory + directionfile) as f:
        hpdirection = np.array([np.array([float(x) for x in line.split()]) for line in f])
    hpdirection = hpdirection.flatten()
    hpdirection = np.reshape(hpdirection,(len(hpdirection)/2.0,2))


    #create the projection of beam map up to l = 3nside-1
    spherical_beam = np.zeros(len(hpdirection),'complex')
    for key in Blm:
        for i in range(len(hpdirection)):
            spherical_beam[i] += Blm[key]*spheh(key[0],key[1],hpdirection[i,0],hpdirection[i,1])

    
    #import the nside = 512 healpix beam
    reffile = 'nside'+str(reference)+'freq150_xx.bin'
    with open(directory+reffile) as f:
        farray = array('f')
        farray.fromstring(f.read())
        rawdata = np.array(farray)
    refdata = rawdata.flatten()

    #calculate the difference between the actual beam and the one recovered from spherical harmonics
    diff= np.zeros(len(data))
    for i in range(len(data)):
        diff[i] = abs(refdata[i]-spherical_beam[i])
    
    #[norm of the diff over norm of the reference map, rms of difference over max value of the beam, max value of difference over max value of the beam]
    return [la.norm(diff)/la.norm(refdata),la.norm(diff)/(12*nside**2)**0.5/max(refdata),max(abs(diff))/max(refdata)]
    
    #print la.norm(diff)/la.norm(refdata)
    #print la.norm(diff)/(12*nside**2)**0.5/max(refdata)
    #print max(abs(diff))/max(refdata)


def estimate_error(refnside=8):
    nside_list = [2**i for i in range(2,10) if 2**i <= refnside]
    result = {}
    for nside in nside_list:
        result[nside] = compare_beam(nside, refnside)
        print '[nside, refnside] = [' + str(nside) + ', ' + str(refnside) + ']'
        print result[nside]
    return result



test = estimate_error(16)



