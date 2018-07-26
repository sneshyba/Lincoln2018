import numpy as np

def makelittleblocky(nx,ny):
    block = np.zeros((ny-1,ny)).astype(int)
    row = np.zeros(ny).astype(int)
    row[0] = -1; row[1] = 1
    for i in range(ny-1):
        block[i,:] = np.roll(row,i)
    #print "little block", block
    return block

def makegrady(nx,ny):
    littleblock = makelittleblocky(nx,ny)
    bigblock = np.kron(np.eye(nx).astype(int),littleblock)
    #print bigblock
    return np.matrix(bigblock)

def oldmakeMxMy(nx,ny):
    # Usage: 
    # Operates on grid "A" which has been flattened according to Along = np.matrix(np.reshape(A.T,nx*ny,1)).T
    import time
    
    # axis = 0 direction
    t = time.time()
    Myp = makegrady(ny,nx); #print Myp
    arow = np.zeros((1,nx*ny)).astype(int); arow[0,0]=1
    T = np.empty((0,nx*ny), int)
    for i in range(ny):
        for j in range(nx):
            T = np.append(T,np.roll(arow,j*ny+i),axis=0)
    print ("time for T is", time.time()-t)

    t = time.time()
    arow = np.zeros((1,(nx-1)*ny)).astype(int); arow[0,0]=1        
    P = np.empty((0,(nx-1)*ny)).astype(int)
    for i in range(nx-1):
        for j in range(ny):
            P = np.append(P,np.roll(arow,j*(nx-1)+i),axis=0)
    print ("time for P is", time.time()-t)
    t = time.time()
    
    #Mx = P*Myp*T; #print "Mx is \n", Mx
    Mx = P*Myp*T; #print "Mx is \n", Mx
    print ("time for Mx is", time.time()-t)
    t = time.time()
    
    # axis = 1 direction
    #t = time.time()
    My = makegrady(nx,ny)
    #print "time for My is", time.time()-t

    return Mx, My


def makeMxMy(nx,ny):
    # Usage: 
    # Operates on grid "A" which has been flattened according to Along = np.matrix(np.reshape(A.T,nx*ny,1)).T
    #import time
    
    # axis = 0 direction
    #t = time.time()
    Myp = makegrady(ny,nx); #print Myp
    arow = np.zeros((1,nx*ny)).astype(int); arow[0,0]=1
    #T = np.empty((0,nx*ny), int)
    T = np.matrix(np.zeros((nx*ny,np.size(arow))))
    irow = 0
    for i in range(ny):
        for j in range(nx):
            #T = np.append(T,np.roll(arow,j*ny+i),axis=0)
            T[irow,:] = np.roll(arow,j*ny+i)
            irow += 1
    #print "time for T is", time.time()-t

    #t = time.time()
    arow = np.zeros((1,(nx-1)*ny)).astype(int); arow[0,0]=1        
    #P = np.matrix(np.empty((0,(nx-1)*ny)).astype(int))
    P = np.matrix(np.zeros(((nx-1)*ny,(nx-1)*ny)).astype(int))
    irow = 0
    for i in range(nx-1):
        for j in range(ny):
            #P = np.append(P,np.roll(arow,j*(nx-1)+i),axis=0)
            P[irow,:] = np.roll(arow,j*(nx-1)+i)
            irow += 1
    #print "time for P is", time.time()-t
    #t = time.time()
    
    #Mx = P*Myp*T; #print "Mx is \n", Mx
    Mx = (P.astype(float)*Myp.astype(float)*T.astype(float)).astype(int); #print "Mx is \n", Mx
    #print "time for Mx is", time.time()-t
    #t = time.time()

    # axis = 1 direction
    #t = time.time()
    My = makegrady(nx,ny)
    #print "time for My is", time.time()-t

    return Mx, My

def oldmakeNxNy(nx,ny):
    Mx,My = oldmakeMxMy(nx,ny)
    Ny = My[0:-(ny-1),:-1]; #print "Ny is \n", np.shape(Ny) # Cuts out the last ny-1 rows, and the last column
    a,b = np.shape(Mx)
    Nx = np.empty((0,b)).astype(int); #print "initial Nx is \n", np.shape(Nx) # Need to cut out the last nx-1 rows
    for i in range(a):
        if (i+1)%ny != 0:
            Nx = np.append(Nx,Mx[i,:],axis=0); #print "Nx is \n", np.shape(Nx) 
    Nx = Nx[:,:-1]
    return Nx, Ny

def makeNxNy(nx,ny):
    Mx,My = makeMxMy(nx,ny)
    Ny = My[0:-(ny-1),:-1]; #print "Ny is \n", np.shape(Ny) # Cuts out the last ny-1 rows, and the last column
    Nx = np.matrix(np.zeros(np.shape(Ny)).astype(int)); #print "initial Nx is \n", np.shape(Nx)
    a,b = np.shape(Mx)
    j = 0
    for i in range(a):
        if (i+1)%ny != 0:
            test = Mx[i,:-1];
            Nx[j,:] = test; j+= 1; #print "Nx is \n", np.shape(Nx) 
    return Nx, Ny


'''
# Testing
nx = 3
ny = 2
print ("nx, ny =", nx,ny)

Nx, Ny = makeNxNy(nx,ny)
print ("Nx is \n", Nx)
print ("Ny is \n", Ny)

A = np.random.rand(nx,ny); print ("A = \n", A)
Along = np.matrix(np.reshape(A.T,nx*ny,1)).T
Along = Along[:-1]; print (Along)

print ("Grad(x) \n", Nx*Along)
print ("numpy's result \n", np.diff(A,axis=0))

print ("Grad(y) \n", Ny*Along)
print ("numpy's result \n", np.diff(A,axis=1))
'''