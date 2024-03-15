# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
import numpy as np 
import time,sys,os
import pyfftw
import scipy.integrate as si
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt,pow,sin,log10,abs,atan2,cos
from libc.stdlib cimport malloc, free
from cpython cimport bool

# This function determines the fundamental (kF) and Nyquist (kN) frequencies
# It also finds the maximum frequency sampled in the box, the maximum 
# frequency along the parallel and perpendicular directions in units of kF
def frequencies(BoxSize,dims):
    kF = 2.0*np.pi/BoxSize;  middle = dims//2;  kN = middle*kF
    kmax_par = middle
    kmax_per = int(np.sqrt(middle**2 + middle**2))
    kmax     = int(np.sqrt(middle**2 + middle**2 + middle**2))
    return kF,kN,kmax_par,kmax_per,kmax

# same as frequencies but in 2D
def frequencies_2D(BoxSize,dims):
    kF = 2.0*np.pi/BoxSize;  middle = dims//2;  kN = middle*kF
    kmax_par = middle
    kmax_per = middle
    kmax     = int(np.sqrt(middle**2 + middle**2))
    return kF,kN,kmax_par,kmax_per,kmax
    
# This function finds the MAS correction index and return the array used
def MAS_function(MAS):
    MAS_index = 0;  #MAS_corr = np.ones(3,dtype=np.float64)
    if MAS=='NGP':  MAS_index = 1
    if MAS=='CIC':  MAS_index = 2
    if MAS=='TSC':  MAS_index = 3
    if MAS=='PCS':  MAS_index = 4
    return MAS_index#,MAS_corr

# This function implement the MAS correction to modes amplitude
#@cython.cdivision(False)
#@cython.boundscheck(False)
cpdef inline double MAS_correction(double x, int MAS_index):
    return (1.0 if (x==0.0) else pow(x/sin(x),MAS_index))

# This function finds the MAS correction index and return 
# the C(k) shotnoise conveloved with the window
cpdef inline double CkNointerlaced(double x, int SN_index):
    if   SN_index == 0:  return 0.0
    elif SN_index == 1:  return 1.0
    elif SN_index == 2:  return (1.0 - 2.0/3.0*x*x)
    elif SN_index == 3:  return (1.0 - x*x + 2.0/15.0*x*x*x*x)
    elif SN_index == 4:  return (1.0 - 4.0/3.0*x*x + 2.0/5.0*x*x*x*x - 4.0/315.0*x*x*x*x*x*x)
    else:    print('ERROR: SN_index not valid');  sys.exit()

cpdef inline double Ckinterlaced_even(double x, double y, int SN_index):
## x means sin(x) and y means cos(x)
    if   SN_index == 0:  return 0.0
    elif SN_index == 1:  return y*y
    elif SN_index == 2:  return y*y*y*y*(1.0 - 2.0/3.0*x*x)
    elif SN_index == 3:  return y*y*y*y*y*y*(1.0 - x*x + 2.0/15.0*x*x*x*x)
    elif SN_index == 4:  return y*y*y*y*y*y*y*y*(1.0 - 4.0/3.0*x*x + 2.0/5.0*x*x*x*x - 4.0/315.0*x*x*x*x*x*x)
    else:    print('ERROR: SN_index not valid');  sys.exit()

cpdef inline double Ckinterlaced_odd(double x, int SN_index):
    if   SN_index == 0:  return 0.0
    elif SN_index == 1:  return x*x
    elif SN_index == 2:  return x*x*x*x*(1.0 - 2.0/3.0*x*x)
    elif SN_index == 3:  return x*x*x*x*x*x*(1.0 - x*x + 2.0/15.0*x*x*x*x)
    elif SN_index == 4:  return x*x*x*x*x*x*x*x*(1.0 - 4.0/3.0*x*x + 2.0/5.0*x*x*x*x - 4.0/315.0*x*x*x*x*x*x)
    else:    print('ERROR: SN_index not valid');  sys.exit()

# This function checks that all independent modes have been counted
def check_number_modes(Nmodes,dims):
    # (0,0,0) own antivector, while (n,n,n) has (-n,-n,-n) for dims odd
    if dims%2==1:  own_modes = 1 
    # (0,0,0),(0,0,n),(0,n,0),(n,0,0),(n,n,0),(n,0,n),(0,n,n),(n,n,n)
    else:          own_modes = 8 
    repeated_modes = (dims**3 - own_modes)//2  
    indep_modes    = repeated_modes + own_modes

    if int(np.sum(Nmodes))!=indep_modes:
        print('WARNING: Not all modes counted')
        print('Counted  %d independent modes'%(int(np.sum(Nmodes))))
        print('Expected %d independent modes'%indep_modes)
        sys.exit() 

# This function checks that all independent modes have been counted
def check_number_modes_2D(Nmodes,dims):
    # (0,0) own antivector, while (n,n) has (-n,-n) for dims odd
    if dims%2==1:  own_modes = 1 
    # (0,0),(0,n),(0,n),(n,0),(n,n)
    else:          own_modes = 4
    repeated_modes = (dims**2 - own_modes)//2  
    indep_modes    = repeated_modes + own_modes

    if int(np.sum(Nmodes))!=indep_modes:
        print('WARNING: Not all modes counted')
        print('Counted  %d independent modes'%(int(np.sum(Nmodes))))
        print('Expected %d independent modes'%indep_modes)
        sys.exit() 

# This function performs the 3D FFT of a field in single precision
def FFT3Dr_f(np.ndarray[np.float32_t,ndim=3] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')
    a_out = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in double precision
def FFT3Dr_d(np.ndarray[np.float64_t,ndim=3] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims),    dtype='float64')
    a_out = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex128')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in,a_out,axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD',threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in single precision
def IFFT3Dr_f(np.complex64_t[:,:,::1] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex64')
    a_out = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in double precision
def IFFT3Dr_d(np.complex128_t[:,:,::1] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex128')
    a_out = pyfftw.empty_aligned((dims,dims,dims),    dtype='float64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in,a_out,axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD',threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 2D FFT of a field in single precision
def FFT2Dr_f(np.float32_t[:,:] a, int threads):

    # align arrays
    grid  = len(a)
    a_in  = pyfftw.empty_aligned((grid,grid),    dtype='float32')
    a_out = pyfftw.empty_aligned((grid,grid//2+1),dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 2D FFT of a field in double precision
def FFT2Dr_d(np.float64_t[:,:] a, int threads):

    # align arrays
    grid  = len(a)
    a_in  = pyfftw.empty_aligned((grid,grid),    dtype='float64')
    a_out = pyfftw.empty_aligned((grid,grid//2+1),dtype='complex128')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in,a_out,axes=(0,1),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD',threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 2D FFT of a field in single precision
def IFFT2Dr_f(np.complex64_t[:,:] a, int threads):

    # align arrays
    grid  = len(a)
    a_in  = pyfftw.empty_aligned((grid,grid//2+1),dtype='complex64')
    a_out = pyfftw.empty_aligned((grid,grid),    dtype='float32')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 2D FFT of a field in double precision
def IFFT2Dr_d(np.complex128_t[:,:] a, int threads):

    # align arrays
    grid  = len(a)
    a_in  = pyfftw.empty_aligned((grid,grid//2+1),dtype='complex128')
    a_out = pyfftw.empty_aligned((grid,grid),    dtype='float64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in,a_out,axes=(0,1),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD',threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

################################################################################
# Notice here k1D and Pk1D are the 1D power spectrum consistent with the k3D 
# and Pk3D in Pylians.
################################################################################
# This routine computes the 1D P(k) of a density field
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# numeff ------> The effective number density of tracers [h^3/Mpc^3]
# MAS ---------> mass assignment scheme used to compute density field
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def Pk1D(delta,BoxSize,MAS='CIC',threads=1, verbose=True, 
         SNFlag=False, numeff=1.0, interlaced=False, 
         kmin=0.0, kmax=None, kbin=None):
    start = time.time()
    cdef int kxx, kyy, kzz, kx, ky, kz,dims, middle, k_index, MAS_index, SN_index
    cdef int kmax_par, k_par, i, modefactor
    cdef double k, delta2, prefact, real, imag, kmaxper, sink, cosk, dk
    cdef double MAS_corr[3]
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef double SN_corr0[3]
    cdef double SN_corr1[3]
    cdef float SN_factor
    cdef np.complex64_t[:,:,::1] delta_k
    ###############################################
    cdef np.float64_t[::1] k1D, Pk1D
    cdef np.float64_t[::1] Nmodes1D

    # find dimensions of delta: we assume is a (dims,dims,dims) array
    # determine the different frequencies and the MAS_index
    if verbose:  print('\nComputing power spectrum of the field...')
    dims = len(delta);  middle = dims//2
    kF   = 2.0*np.pi/BoxSize
    MAS_index = MAS_function(MAS)
    if SNFlag:
        SN_index = MAS_index
    else:
        SN_index = 0

    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    #################################
    if kmax is None:  kmax = middle*kF
    if kbin is None:  kbin = middle + 1

    kmax = kmax/kF; kmin = kmin/kF
    # define Bins and arrays 
    dk = (kmax-kmin)/(kbin-1)
    k1D      = np.zeros(kbin-1, dtype=np.float64)
    Pk1D     = np.zeros(kbin-1, dtype=np.float64)
    Nmodes1D = np.zeros(kbin-1, dtype=np.float64)

    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time();  prefact = np.pi/dims
    if interlaced and SNFlag:
        if verbose: print('Subtracting shotnoise from the field in the interlaced case.')
        for kxx in range(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)
            sink = sin(prefact*kx);  cosk = cos(prefact*kx)
            MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
            SN_corr0[0] = Ckinterlaced_even(sink, cosk, SN_index)
            SN_corr1[0] = Ckinterlaced_odd (sink, SN_index)
        
            for kyy in range(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)
                sink = sin(prefact*ky);  cosk = cos(prefact*ky)
                MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)
                SN_corr0[1] = Ckinterlaced_even(sink, cosk, SN_index)
                SN_corr1[1] = Ckinterlaced_odd (sink, SN_index)

                for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    modefactor = 2
                    if (kzz==middle) or (kzz==0) : modefactor = 1
                    sink = sin(prefact*kz);  cosk = cos(prefact*kz)
                    MAS_corr[2] = MAS_correction(prefact*kz,MAS_index) 
                    SN_corr0[2] = Ckinterlaced_even(sink, cosk, SN_index) 
                    SN_corr1[2] = Ckinterlaced_odd (sink, SN_index)

                    # kz=0 and kz=middle planes are special
                    # if kz==0 or (kz==middle and dims%2==0):
                    #     if kx<0: continue
                    #     elif kx==0 or (kx==middle and dims%2==0):
                    #         if ky<0.0: continue

                    # compute |k| of the mode and its integer part
                    k       = sqrt(kx*kx + ky*ky + kz*kz)
                    if (k >= kmax) or (k < kmin): continue
                    
                    k_index = int((k-kmin)/dk)
                    # correct modes amplitude for MAS
                    MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                    SN_factor  = SN_corr0[0]*SN_corr0[1]*SN_corr0[2]
                    SN_factor += SN_corr0[0]*SN_corr1[1]*SN_corr1[2]
                    SN_factor += SN_corr1[0]*SN_corr0[1]*SN_corr1[2]
                    SN_factor += SN_corr1[0]*SN_corr1[1]*SN_corr0[2]
                    SN_factor  = SN_factor / numeff

                    # compute |delta_k|^2 of the mode
                    real = delta_k[kxx,kyy,kzz].real
                    imag = delta_k[kxx,kyy,kzz].imag
                    delta2 = (real*real + imag*imag - SN_factor)*MAS_factor*MAS_factor

                    ## Here PK1D means 1D power spectrum not 1D k (in Pylians)
                    k1D[k_index]      += k * modefactor
                    Pk1D[k_index]     += delta2 * modefactor
                    Nmodes1D[k_index] += 1.0 * modefactor
    else:
        for kxx in range(dims):
        # for kxx in prange(dims, schedule='static'):
            kx = (kxx-dims if (kxx>middle) else kxx)
            sink = sin(prefact*kx)
            MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
            SN_corr0[0] = CkNointerlaced(sink, SN_index)
        
            for kyy in range(dims):
            # for kyy in prange(dims, schedule='static'):
                ky = (kyy-dims if (kyy>middle) else kyy)
                sink = sin(prefact*ky)
                MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)
                SN_corr0[1] = CkNointerlaced(sink, SN_index)

                for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                # for kzz in prange(middle+1, schedule='static'):
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    modefactor = 2
                    if (kzz==middle) or (kzz==0) : modefactor = 1
                    sink = sin(prefact*kz)
                    MAS_corr[2] = MAS_correction(prefact*kz,MAS_index) 
                    SN_corr0[2] = CkNointerlaced(sink, SN_index) 

                    # compute |k| of the mode and its integer part
                    k       = sqrt(kx*kx + ky*ky + kz*kz)
                    if (k >= kmax) or (k < kmin): continue
                    
                    k_index = int((k-kmin)/dk)

                    # correct modes amplitude for MAS
                    MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                    SN_factor  = SN_corr0[0]*SN_corr0[1]*SN_corr0[2]/numeff
                    
                    # compute |delta_k|^2 of the mode
                    real = delta_k[kxx,kyy,kzz].real
                    imag = delta_k[kxx,kyy,kzz].imag
                    delta2 = (real*real + imag*imag - SN_factor)*MAS_factor*MAS_factor

                    ## Here PK1D means 1D power spectrum not 1D k (in Pylians)
                    k1D[k_index]      += k * modefactor
                    Pk1D[k_index]     += delta2 * modefactor
                    Nmodes1D[k_index] += 1.0 * modefactor

    if verbose:  print('Time to complete loop = %.2f'%(time.time()-start2))

    # Pk1D. Check modes, discard DC mode bin and give units
    # we need to multiply the multipoles by (2*ell + 1)
    ### give units
    for i in range(len(k1D)):
        k1D[i]  = (k1D[i]/Nmodes1D[i]) * kF                     # give units
        Pk1D[i] = (Pk1D[i]/Nmodes1D[i]) * (BoxSize/dims**2)**3  #give units 

    if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))

    return np.asarray(k1D), np.asarray(Pk1D), np.asarray(Nmodes1D)
##################################################  ##############################
################################################################################

################################################################################
# Notice here k1D and Pk1D are the 1D power spectrum consistent with the k3D 
# and Pk3D in Pylians.
################################################################################
# This routine computes the 1D P(k) of a density field
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# axis --------> axis along which place the line of sight
# MAS ---------> mass assignment scheme used to compute density field
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
# SNFlag ------> if True, it subtracts the shotnoise from the field
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def XPk1D(delta0,delta1,BoxSize,MAS='CIC',threads=1, verbose=True, SNFlag=False, numeff=1.0, interlaced=False):
    start = time.time()
    cdef int kxx, kyy, kzz, kx, ky, kz,dims, middle, k_index, MAS_index
    cdef int kmax_par, k_par, i
    cdef double k, delta2, prefact, real0, imag0, real1, imag1, kmaxper, sink, cosk
    cdef double MAS_corr[3] 
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.complex64_t[:,:,::1] delta0_k
    cdef np.complex64_t[:,:,::1] delta1_k 
    ###############################################
    cdef np.float64_t[::1] k1D, Pk1D
    cdef np.float64_t[::1] Nmodes1D
    ###### sub shotnoise in the field level
    cdef double SN_corr0[3]
    cdef double SN_corr1[3]
    cdef float SN_factor
    # find dimensions of delta: we assume is a (dims,dims,dims) array
    # determine the different frequencies and the MAS_index
    if verbose:  print('\nComputing power spectrum of the field...')
    dims = len(delta0);  
    if dims!=len(delta1):
        print('ERROR: The dimensions of the two fields are different')
        sys.exit()
    middle = dims//2
    kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)
    MAS_index = MAS_function(MAS)

    ## compute FFT of the field (change this for double precision) ##
    delta0_k = FFT3Dr_f(delta0,threads)
    delta1_k = FFT3Dr_f(delta0,threads)
    #################################
    if SNFlag:
        SN_index = MAS_index
    else:
        SN_index = 0

    # define arrays containing k1D, Pk1D and Nmodes1D. We need kmax_par+1
    # bins since modes go from 0 to kmax_par
    k1D      = np.zeros(kmax+1, dtype=np.float64)
    Pk1D     = np.zeros(kmax+1, dtype=np.float64)
    Nmodes1D = np.zeros(kmax+1, dtype=np.float64)

    # do a loop over the independent modes.
    # k's are in kF units
    start2 = time.time();  prefact = np.pi/dims
    if interlaced:
        for kxx in range(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)
            sink = sin(prefact*kx);  cosk = cos(prefact*kx)
            MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
            SN_corr0[0] = Ckinterlaced_even(sink, cosk, SN_index)
            SN_corr1[0] = Ckinterlaced_odd (sink, SN_index)
            
            for kyy in range(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)
                sink = sin(prefact*ky);  cosk = cos(prefact*ky)
                MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)
                SN_corr0[1] = Ckinterlaced_even(sink, cosk, SN_index)
                SN_corr1[1] = Ckinterlaced_odd (sink, SN_index)

                for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    sink = sin(prefact*kz);  cosk = cos(prefact*kz)
                    MAS_corr[2] = MAS_correction(prefact*kz,MAS_index) 
                    SN_corr0[2] = Ckinterlaced_even(sink, cosk, SN_index)
                    SN_corr1[2] = Ckinterlaced_odd (sink, SN_index)

                    # kz=0 and kz=middle planes are special
                    if kz==0 or (kz==middle and dims%2==0):
                        if kx<0: continue
                        elif kx==0 or (kx==middle and dims%2==0):
                            if ky<0.0: continue

                    # compute |k| of the mode and its integer part
                    k       = sqrt(kx*kx + ky*ky + kz*kz)
                    k_index = <int>k

                    # correct modes amplitude for MAS
                    MAS_factor  = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                    SN_factor  = SN_corr0[0]*SN_corr0[1]*SN_corr0[2]
                    SN_factor += SN_corr0[0]*SN_corr1[1]*SN_corr1[2]
                    SN_factor += SN_corr1[0]*SN_corr0[1]*SN_corr1[2]
                    SN_factor += SN_corr1[0]*SN_corr1[1]*SN_corr0[2]
                    SN_factor /= numeff

                    # compute |delta_k|^2 of the mode
                    real0 = delta0_k[kxx,kyy,kzz].real
                    real1 = delta1_k[kxx,kyy,kzz].real
                    imag0 = delta0_k[kxx,kyy,kzz].imag
                    imag1 = delta1_k[kxx,kyy,kzz].imag
                    delta2 = (real0*real1 + imag0*imag1 - SN_factor)*MAS_factor*MAS_factor

                    ## Here PK1D means 1D power spectrum not 1D k (in Pylians)
                    k1D[k_index]      += k
                    Pk1D[k_index]     += delta2
                    Nmodes1D[k_index] += 1.0
    else:
        for kxx in range(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)
            sink = sin(prefact*kx)
            MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
            SN_corr0[0] = CkNointerlaced(sink, SN_index)
            
            for kyy in range(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)
                sink = sin(prefact*ky)
                MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)
                SN_corr0[1] = CkNointerlaced(sink, SN_index)

                for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    sink = sin(prefact*kz)
                    MAS_corr[2] = MAS_correction(prefact*kz,MAS_index) 
                    SN_corr0[2] = CkNointerlaced(sink, SN_index)

                    # kz=0 and kz=middle planes are special
                    if kz==0 or (kz==middle and dims%2==0):
                        if kx<0: continue
                        elif kx==0 or (kx==middle and dims%2==0):
                            if ky<0.0: continue

                    # compute |k| of the mode and its integer part
                    k       = sqrt(kx*kx + ky*ky + kz*kz)
                    k_index = <int>k

                    # correct modes amplitude for MAS
                    MAS_factor  = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                    SN_factor   = SN_corr0[0]*SN_corr0[1]*SN_corr0[2]/numeff

                    # compute |delta_k|^2 of the mode
                    real0 = delta0_k[kxx,kyy,kzz].real
                    real1 = delta1_k[kxx,kyy,kzz].real
                    imag0 = delta0_k[kxx,kyy,kzz].imag
                    imag1 = delta1_k[kxx,kyy,kzz].imag
                    delta2 = (real0*real1 + imag0*imag1 - SN_factor)*MAS_factor*MAS_factor

                    ## Here PK1D means 1D power spectrum not 1D k (in Pylians)
                    k1D[k_index]      += k
                    Pk1D[k_index]     += delta2
                    Nmodes1D[k_index] += 1.0

    if verbose:  print('Time to complete loop = %.2f'%(time.time()-start2))

    # Pk1D. Check modes, discard DC mode bin and give units
    # we need to multiply the multipoles by (2*ell + 1)
    k1D  = k1D[1:];  Nmodes1D = Nmodes1D[1:];  Pk1D = Pk1D[1:]
    for i in range(len(k1D)):
        k1D[i]  = (k1D[i]/Nmodes1D[i])*kF      #give units
        Pk1D[i] = (Pk1D[i]/Nmodes1D[i])*(BoxSize/dims**2)**3 #give units

    if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))

    return np.asarray(k1D), np.asarray(Pk1D), np.asarray(Nmodes1D)
################################################################################
################################################################################


################################################################################
################################################################################
# This routine computes the number of independent modes using three different
# ways. It is mainly use to make sure that we are not loosing modes in the 
# main routine
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def test_number_modes_2D(int grid):
    cdef int middle, count, kxx, kyy, kx, ky, Pylians_count, brute_force_count
    cdef int i, indep, kxi, kyi, kxj, kyj
    cdef int[:,:] modes

    # theory expectation
    if grid%2==1:  own_modes = 1 
    else:          own_modes = 4
    repeated_modes = (grid**2 - own_modes)//2  
    theory_count   = repeated_modes + own_modes

    # define the array containing all 2D modes
    middle = grid//2
    modes  = np.zeros((grid*(middle+1),2), dtype=np.int32)

    # count modes as Pylians
    count = 0
    Pylians_count = 0
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)
        
        for kyy in range(middle+1): #kyy=[0,1,..,middle] --> ky>0
            ky = (kyy-grid if (kyy>middle) else kyy)

            modes[count,0] = kx
            modes[count,1] = ky
            count += 1

            if (ky==0) or (ky==middle and grid%2==0):
                if kx<0:   continue

            Pylians_count +=1

    # count modes by bruce force
    brute_force_count = 0
    for i in range(count):
        indep = 1
        kxi,kyi = modes[i,0], modes[i,1]
        for j in range(i+1,count):
            kxj,kyj = -modes[j,0], -modes[j,1]
            if (grid%2==0) and kxj==-middle:  kxj=middle
            if (grid%2==0) and kyj==-middle:  kyj=middle
            if kxi==kxj and kyi==kyj:  indep = 0
        if indep==1:  brute_force_count += 1

    # print results
    print('Theory      modes = %d'%theory_count)
    print('Pylians     modes = %d'%Pylians_count)
    print('Brute force modes = %d'%brute_force_count)
    if theory_count!=Pylians_count or theory_count!=brute_force_count \
       or Pylians_count!=brute_force_count:
        raise Exception('Number of modes differ!!!')
################################################################################
################################################################################

################################################################################
################################################################################
# This routine takes a field in real-space, Fourier transform it to get it in
# Fourier-space and then correct the modes amplitude to account for MAS. It then
# Fourier transform back and return the field in real-space 
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# MAS ---------> mass assignment scheme used to compute density field
#                needed to correct modes amplitude
# threads -----> number of threads (OMP) used to make the FFTW
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def correct_MAS(delta,BoxSize,MAS='CIC',threads=1):

    start = time.time()
    cdef int kxx, kyy, kzz, kx, ky, kz, dims, middle, MAS_index
    cdef int kmax_par, kmax_per, kmax, i
    cdef double k, prefact
    cdef double MAS_corr[3]
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.complex64_t[:,:,::1] delta_k
    ###############################################

    # find dimensions of delta: we assume is a (dims,dims,dims) array
    # determine the different frequencies and the MAS_index
    print('\nComputing power spectrum of the field...')
    dims = len(delta);  middle = dims//2
    kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)
    MAS_index = MAS_function(MAS)
    
    ## compute FFT of the field (change this for double precision) ##
    delta_k = FFT3Dr_f(delta,threads)
    #################################

    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time();  prefact = np.pi/dims
    for kxx in range(dims):
        kx = (kxx-dims if (kxx>middle) else kxx)
        MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
        
        for kyy in range(dims):
            ky = (kyy-dims if (kyy>middle) else kyy)
            MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

            for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-dims if (kzz>middle) else kzz)
                MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)  

                # kz=0 and kz=middle planes are special
                if kz==0 or (kz==middle and dims%2==0):
                    if kx<0: continue
                    elif kx==0 or (kx==middle and dims%2==0):
                        if ky<0.0: continue

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

    print('Time to complete loop = %.2f'%(time.time()-start2))

    ## compute IFFT of the field (change this for double precision) ##
    delta = IFFT3Dr_f(delta_k,threads)
    #################################

    print('Time taken = %.2f seconds'%(time.time()-start))

    return delta
################################################################################
################################################################################

