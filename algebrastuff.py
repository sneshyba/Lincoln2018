import numpy as np
try:
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    from skcuda import linalg, misc
    linalg.init()
    import cudastuff as cds; reload(cds)
    cuda_available = True
    print("Using CUDA.")
except ImportError:
    print("CUDA not available.")
    cuda_available = False

def dot(a, b):
    ''' Calculates matrix multiplication "a*b" on GPU. '''
    #print("dot "+str(a.shape)+" "+str(b.shape))
    
    if (cuda_available):
        return cds.dot(a, b)
    else:
        return np.dot(a, b)
    
def dot2(b, A):
    ''' Calculates matrix multiplication "b.T*A" on GPU. '''
    #print("dot2 "+str(b.shape)+" "+str(A.shape))
    
    if (cuda_available):
        return cds.dot2(b, A)
    else:
        return np.dot(b.T, A)

    
def dot3(A, b):
    ''' Calculates matrix multiplication "b.T*A*b" on GPU. 
        A has to be nxn. '''
    #print("dot3 "+str(A.shape)+" "+str(b.shape))
    
    if (cuda_available):
        return cds.dot3(A, b)
    else:
        return np.dot(np.dot(b.T, A), b)
    
def T(a):
    ''' Transposes matrix "y" on the GPU. '''
    if (cuda_available):
        return cds.T(a)
    else:
        return np.matrix(a.T, copy=False)
    
def substract(a, b):
    ''' Calculates matrix substraction "a-b".'''
    if (cuda_available):
        return cds.substract(a, b)
    else:
        return a-b    