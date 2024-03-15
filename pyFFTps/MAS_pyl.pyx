# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
import numpy as np 
import time,sys,os
import scipy.integrate as SI
from libc.math cimport sqrt,pow,sin,cos,floor,fabs
import pyFFTps.PowerSpec as PS
cimport numpy as np
cimport cython
cimport pyFFTps.MAS_c as MASC

ctypedef MASC.FLOAT FLOAT

def FLOAT_type():
    if sizeof(MASC.FLOAT)==4:  return np.float32
    else:                      return np.float64

def Interlacing(MassAssign, pos, number, BoxSize, Nmesh, threads=1):
    delta_k   = np.zeros((Nmesh,Nmesh,Nmesh//2+1),dtype=np.complex64)
    karr      = np.fft.fftfreq(Nmesh,1.0/Nmesh).astype(np.float32) * np.pi / Nmesh 
    karr[Nmesh//2] = Nmesh / 2 ## change the -Nmesh/2 to Nmesh/2 consistent with rfftfreq
    karrz     = np.fft.rfftfreq(Nmesh,1.0/Nmesh).astype(np.float32) * np.pi / Nmesh 
    karr      = np.sum(np.meshgrid(karr, karr, karrz), axis=0)
    numberint = np.zeros_like(number)
    MassAssign(pos,number,BoxSize)
    #interlacing shift start
    pos[:] = pos[:] + 0.5*BoxSize/Nmesh 
    pos[pos>=BoxSize] -= BoxSize
    pos[pos<0] += BoxSize
    print(pos.min(),pos.max())
    #interlacing shift end
    MassAssign(pos,numberint,BoxSize)
    delta_k[:]  = PS.FFT3Dr_f(numberint,threads)
    delta_k[:]  = 0.5*delta_k[:]*np.exp(1j * karr[:])
    delta_k[:]  = delta_k[:] + 0.5*PS.FFT3Dr_f(number,threads)
    number[:]   = PS.IFFT3Dr_f(delta_k,threads)
    del delta_k, numberint, karr

################################################################################
################################# ROUTINES #####################################
#### from particle positions, with optional weights, to 3D density fields ####
# MA(pos,number,BoxSize,MAS='CIC',W=None) ---> main routine
# NGP(pos,number,BoxSize)
# CIC(pos,number,BoxSize)
# TSC(pos,number,BoxSize)
# PCS(pos,number,BoxSize)
################################################################################
################################################################################

# This is the main function to use when performing the mass assignment
# pos --------> array containing the positions of the particles: 2D or 3D
# numbers ----> array containing the density field: 2D or 3D
# BoxSize ----> size of the simulation box
# MAS --------> mass assignment scheme: NGP, CIC, TSC or PCS
# W ----------> array containing the weights to be used: 1D array (optional)
# renormalize_2D ---> when computing the density field by reading multiple 
# subfiles, the normalization factor /2.0, /3.0, /4.0 should be added manually
# only at the end, otherwise the results will be incorrect!!
cpdef void MA(pos, number, BoxSize, MAS='CIC', W=None, verbose=False,
              renormalize_2D=True, interlaced=False, threads=1):
    #number of coordinates to work in 2D or 3D
    coord,coord_aux = pos.shape[1], number.ndim
    if interlaced:
        Nmesh     = number.shape[0]
    # check that the number of dimensions match
    if coord!=coord_aux:
        print('pos have %d dimensions and the density %d!!!'%(coord,coord_aux))
        sys.exit()
    if verbose:
        if W is None:  print('\nUsing %s mass assignment scheme'%MAS)
        else:          print('\nUsing %s mass assignment scheme with weights'%MAS)
    start = time.time()
    if coord==3:
        if interlaced:
            if verbose: print('Interlacing the density field')
            if   MAS=='NGP' and W is None:  
                Interlacing(NGP, pos, number, BoxSize, Nmesh, threads)
            elif MAS=='CIC' and W is None:
                Interlacing(CIC, pos, number, BoxSize, Nmesh, threads)
            elif MAS=='TSC' and W is None:  
                Interlacing(TSC, pos, number, BoxSize, Nmesh, threads)
            elif MAS=='PCS' and W is None:  
                Interlacing(PCS, pos, number, BoxSize, Nmesh, threads)
            else:
                print('option not valid!!!');  sys.exit()
        else:
            if   MAS=='NGP' and W is None:  NGP(pos,number,BoxSize)
            elif MAS=='CIC' and W is None:  CIC(pos,number,BoxSize)
            elif MAS=='TSC' and W is None:  TSC(pos,number,BoxSize)
            elif MAS=='PCS' and W is None:  PCS(pos,number,BoxSize)
            else:
                print('option not valid!!!');  sys.exit()

    if coord==2:
        number2 = np.expand_dims(number,axis=2)
        if   MAS=='NGP' and W is None:  
            NGP(pos,number2,BoxSize)
        elif MAS=='CIC' and W is None:  
            CIC(pos,number2,BoxSize);
            if renormalize_2D:  number2 /= 2.0
        elif MAS=='TSC' and W is None:  
            TSC(pos,number2,BoxSize);  
            if renormalize_2D:  number2 /= 3.0
        elif MAS=='PCS' and W is None:  
            PCS(pos,number2,BoxSize);
            if renormalize_2D:  number2 /= 4.0
        else:
            print('option not valid!!!');  sys.exit()
        number = number2[:,:,0]
    if verbose:
        print('Time taken = %.3f seconds\n'%(time.time()-start))

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef void CIC(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize):
        
    cdef int axis,dims,coord
    cdef long i,particles
    cdef float inv_cell_size,dist
    cdef float u[3]
    cdef float d[3]
    cdef int index_u[3]
    cdef int index_d[3]

    # find number of particles, the inverse of the cell size and dims
    particles = pos.shape[0];  coord = pos.shape[1];  dims = number.shape[0]
    inv_cell_size = dims/BoxSize
    
    # when computing things in 2D, use the index_ud[2]=0 plane
    for i in range(3):
        index_d[i] = 0;  index_u[i] = 0;  d[i] = 1.0;  u[i] = 1.0

    # do a loop over all particles
    for i in range(particles):

        # $: grid point, X: particle position
        # $.........$..X......$
        # ------------>         dist    (1.3)
        # --------->            index_d (1)
        # --------------------> index_u (2)
        #           --->        u       (0.3)
        #              -------> d       (0.7)
        for axis in range(coord):
            dist          = pos[i,axis]*inv_cell_size
            u[axis]       = dist - <int>dist
            d[axis]       = 1.0 - u[axis]
            index_d[axis] = (<int>dist)%dims
            index_u[axis] = index_d[axis] + 1
            index_u[axis] = index_u[axis]%dims #seems this is faster

        number[index_d[0],index_d[1],index_d[2]] += d[0]*d[1]*d[2]
        number[index_d[0],index_d[1],index_u[2]] += d[0]*d[1]*u[2]
        number[index_d[0],index_u[1],index_d[2]] += d[0]*u[1]*d[2]
        number[index_d[0],index_u[1],index_u[2]] += d[0]*u[1]*u[2]
        number[index_u[0],index_d[1],index_d[2]] += u[0]*d[1]*d[2]
        number[index_u[0],index_d[1],index_u[2]] += u[0]*d[1]*u[2]
        number[index_u[0],index_u[1],index_d[2]] += u[0]*u[1]*d[2]
        number[index_u[0],index_u[1],index_u[2]] += u[0]*u[1]*u[2]
################################################################################

################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void NGP(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize):

    cdef int axis,dims,coord
    cdef long i,particles
    cdef float inv_cell_size
    cdef int index[3]

    # find number of particles, the inverse of the cell size and dims
    particles = pos.shape[0];  coord = pos.shape[1];  dims = number.shape[0]
    inv_cell_size = dims/BoxSize

    # when computing things in 2D, use the index[2]=0 plane
    for i in range(3):  index[i] = 0

    # do a loop over all particles
    for i in range(particles):
        for axis in range(coord):
            index[axis] = <int>(pos[i,axis]*inv_cell_size + 0.5)
            index[axis] = (index[axis]+dims)%dims
        number[index[0],index[1],index[2]] += 1.0
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void TSC(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize):

    cdef int axis, dims, minimum
    cdef int j, l, m, n, coord
    cdef long i, particles
    cdef float inv_cell_size, dist, diff
    cdef float C[3][3]
    cdef int index[3][3]

    # find number of particles, the inverse of the cell size and dims
    particles = pos.shape[0];  coord = pos.shape[1];  dims = number.shape[0]
    inv_cell_size = dims/BoxSize
    
    # define arrays: for 2D set we have C[2,:] = 1.0 and index[2,:] = 0
    for i in range(3):
        for j in range(3):
            C[i][j] = 1.0;  index[i][j] = 0
            
    # do a loop over all particles
    for i in range(particles):

        # do a loop over the axes of the particle
        for axis in range(coord):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-1.5)
            for j in range(3): #only 3 cells/dimension can contribute
                index[axis][j] = (minimum+j+1+dims)%dims
                diff = fabs(minimum + j+1 - dist)
                if diff<0.5:    C[axis][j] = 0.75-diff*diff
                elif diff<1.5:  C[axis][j] = 0.5*(1.5-diff)*(1.5-diff)
                else:           C[axis][j] = 0.0

        for l in range(3):  
            for m in range(3):  
                for n in range(3): 
                    number[index[0][l],index[1][m],index[2][n]] += C[0][l]*C[1][m]*C[2][n]
################################################################################

################################################################################
# This function computes the density field of a cubic distribution of particles
# pos ------> positions of the particles. Numpy array
# number ---> array with the density field. Numpy array (dims,dims,dims)
# BoxSize --> Size of the box
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void PCS(np.float32_t[:,:] pos, np.float32_t[:,:,:] number, float BoxSize):

    cdef int axis,dims,minimum,j,l,m,n,coord
    cdef long i,particles
    cdef float inv_cell_size,dist,diff
    cdef float C[3][4]
    cdef int index[3][4]

    # find number of particles, the inverse of the cell size and dims
    particles = pos.shape[0];  coord = pos.shape[1];  dims = number.shape[0]
    inv_cell_size = dims/BoxSize
        
    # define arrays: for 2D set we have C[2,:] = 1.0 and index[2,:] = 0
    for i in range(3):
        for j in range(4):
            C[i][j] = 1.0;  index[i][j] = 0

    # do a loop over all particles
    for i in range(particles):

        # do a loop over the three axes of the particle
        for axis in range(coord):
            dist    = pos[i,axis]*inv_cell_size
            minimum = <int>floor(dist-2.0)
            for j in range(4): #only 4 cells/dimension can contribute
                index[axis][j] = (minimum + j+1 + dims)%dims
                diff = fabs(minimum + j+1 - dist)
                if diff<1.0:    C[axis][j] = (4.0 - 6.0*diff*diff + 3.0*diff*diff*diff)/6.0
                elif diff<2.0:  C[axis][j] = (2.0 - diff)*(2.0 - diff)*(2.0 - diff)/6.0
                else:           C[axis][j] = 0.0

        for l in range(4):  
            for m in range(4):  
                for n in range(4): 
                    number[index[0][l],index[1][m],index[2][n]] += C[0][l]*C[1][m]*C[2][n]
################################################################################

##################### MAS_c (openmp) functions ########################

############## NGP #################
cpdef void NGPc3D(FLOAT[:,::1] pos, FLOAT[:,:,::1] number, FLOAT BoxSize,
                  long threads):
    MASC.NGP(&pos[0,0], &number[0,0,0], NULL, pos.shape[0], number.shape[0],
             pos.shape[1], BoxSize, threads)

cpdef void NGPc2D(FLOAT[:,::1] pos, FLOAT[:,::1] number, FLOAT BoxSize,
                  long threads):
    MASC.NGP(&pos[0,0], &number[0,0], NULL, pos.shape[0], number.shape[0],
             pos.shape[1], BoxSize, threads)
####################################

############## CIC #################
cpdef void CICc3D(FLOAT[:,::1] pos, FLOAT[:,:,::1] number, FLOAT BoxSize,
                  long threads):
    MASC.CIC(&pos[0,0], &number[0,0,0], NULL, pos.shape[0], number.shape[0],
             pos.shape[1], BoxSize, threads)

cpdef void CICc2D(FLOAT[:,::1] pos, FLOAT[:,::1] number, FLOAT BoxSize,
                  long threads):
    MASC.CIC(&pos[0,0], &number[0,0], NULL, pos.shape[0], number.shape[0],
             pos.shape[1], BoxSize, threads)
####################################

############## TSC #################
cpdef void TSCc3D(FLOAT[:,::1] pos, FLOAT[:,:,::1] number, FLOAT BoxSize,
                  long threads):
    MASC.TSC(&pos[0,0], &number[0,0,0], NULL, pos.shape[0], number.shape[0],
             pos.shape[1], BoxSize, threads)

cpdef void TSCc2D(FLOAT[:,::1] pos, FLOAT[:,::1] number, FLOAT BoxSize,
                  long threads):
    MASC.TSC(&pos[0,0], &number[0,0], NULL, pos.shape[0], number.shape[0],
             pos.shape[1], BoxSize, threads)

############## PCS #################
cpdef void PCSc3D(FLOAT[:,::1] pos, FLOAT[:,:,::1] number,
                  FLOAT BoxSize, long threads):
    MASC.PCS(&pos[0,0], &number[0,0,0], NULL, pos.shape[0], number.shape[0],
             pos.shape[1], BoxSize, threads)

cpdef void PCSc2D(FLOAT[:,::1] pos, FLOAT[:,::1] number,
                   FLOAT BoxSize, long threads):
    MASC.PCS(&pos[0,0], &number[0,0], NULL, pos.shape[0], number.shape[0],
              pos.shape[1], BoxSize, threads)
####################################
