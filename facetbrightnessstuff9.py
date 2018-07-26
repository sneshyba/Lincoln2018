import numpy as np
import scipy
from numpy import matrix, cross, zeros, reshape, vstack, hstack, empty, roll, shape
#import scipy.interpolate
import copy
from numpy.linalg.linalg import norm

''' A copy of fcb4, but CUDA stuff is elective. '''

import algebrastuff as ags; #reload(ags)
import gradstuff as gds

def rdfbfile(filename):

    import pandas
    testp = pandas.DataFrame().from_csv(filename)
    testpdict = testp.to_dict()

    tiltlist = []
    for key, value in testpdict['tilt'].iteritems():
        tiltlist.append(value)

    facetlist = []
    for key, value in testpdict['facet'].iteritems():
        facetlist.append(value)

    detectorlist = []
    for key, value in testpdict['detector'].iteritems():
        detectorlist.append(value)

    BSlist = []
    for key, value in testpdict['BS'].iteritems():
        BSlist.append(value)

    ndblist = []
    for key, value in testpdict['ndb'].iteritems():
        ndblist.append(value)

    nddlist = []
    for key, value in testpdict['ndd'].iteritems():
        nddlist.append(value)

    x1list = []
    for key, value in testpdict['x1'].iteritems():
        x1list.append(value)

    x2list = []
    for key, value in testpdict['x2'].iteritems():
        x2list.append(value)

    y1list = []
    for key, value in testpdict['y1'].iteritems():
        y1list.append(value)

    y2list = []
    for key, value in testpdict['y2'].iteritems():
        y2list.append(value)

    Filenamelist = []
    for key, value in testpdict['Filename'].iteritems():
        Filenamelist.append(value)

    refxlist = []
    for i in range(4):
        value = testpdict['refx'][i]
        refxlist.append(value)

    refylist = []
    for i in range(4):
        value = testpdict['refy'][i]
        refylist.append(value)

    ABCDangle = testpdict['ABCDangle'][0]
    return tiltlist, facetlist, detectorlist, BSlist, ndblist, nddlist, \
        x1list, x2list, y1list, y2list, \
        Filenamelist, refxlist, refylist, ABCDangle

# This defines a rotation matrix
def rotation_matrix(axis, theta_deg):
    """
    Return the rotation matrix associated with clockwise rotation of an object
    (clockwise as seen by looking along the rotation axis toward the origin)
    about the given axis by theta degrees
    """
    import math
    theta = theta_deg*np.pi/180
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.asmatrix(np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]]))

def solveforabc(xa,ya,xb,yb,xc,yc):

    from sympy import S, Eq, solve
    from numpy import linalg as LA
    from numpy import abs  

    # This solves for the missing z-component of the vectors
    zc, cmag = S('zc cmag'.split())
    za, amag = S('za amag'.split())
    zb, bmag = S('zb bmag'.split())
    equations = [
        Eq(xc*xa+yc*ya+zc*za, 0),
        Eq(xc*xb+yc*yb+zc*zb, 0),
        Eq(xa*xb+ya*yb+za*zb, -amag*bmag/2),
        Eq(xa**2+ya**2+za**2,amag**2),
        Eq(xb**2+yb**2+zb**2,bmag**2),
        Eq(xc**2+yc**2+zc**2,cmag**2)]
    solution = solve(equations)

    # Print which one looks most physically reasonable
    print("found", len(solution), "solutions")
    for i in range(len(solution)):
        if  solution[i][bmag] >= 0 and \
            solution[i][cmag] >= 0 and \
            solution[i][amag] >= 0 and \
            solution[i][zb] <= 0:
                print ("physically reasonable solution is #", i)
                #print ("solution=",solution[i])
                isol = i
                break

    # Create the vectors (matrices)
    zaval = solution[isol][za]
    zbval = solution[isol][zb]
    zcval = solution[isol][zc]
    avec = matrix([xa,ya,zaval]).astype('float64').T
    bvec = matrix([xb,yb,zbval]).astype('float64').T
    cvec = matrix([xc,yc,zcval]).astype('float64').T

    # This normalizes them
    avec = avec/LA.norm(avec)
    bvec = bvec/LA.norm(bvec)
    cvec = cvec/LA.norm(cvec)

    # This gets normals for the "a" and "b" facets
    navec = matrix(cross(cvec.T,avec.T)).T; #print navec
    nbvec = matrix(cross(bvec.T,cvec.T)).T; #print nbvec

    # Return
    return avec, bvec, cvec, navec, nbvec

def getdetectorvectors(ABCDangle_deg):
    ABCDangle = ABCDangle_deg*np.pi/180
    dAvec = matrix([-np.sin(ABCDangle),0,np.cos(ABCDangle)]).astype('float64').T
    dBvec = matrix([0,-np.sin(ABCDangle),np.cos(ABCDangle)]).astype('float64').T
    dCvec = matrix([+np.sin(ABCDangle),0,np.cos(ABCDangle)]).astype('float64').T
    dDvec = matrix([0,+np.sin(ABCDangle),np.cos(ABCDangle)]).astype('float64').T
    return dAvec, dBvec, dCvec, dDvec

def rBS(diffterm,specterm, ks):
    """
    Returns a modeled backscatter intensity
    """
#     # according to Blinn-Phong
#     return normal_dot_detector, normal_dot_beam, \
#         ((normal_dot_detector+normal_dot_beam)/2)**2

    # Phong or Blinn-Phong
    return (1-ks)*diffterm + ks*specterm**2

def getK(nxy_start,
         Arule, Brule, Crule, Drule,
         KAxrule, KAyrule,
         KBxrule, KByrule,
         KCxrule, KCyrule,
         KDxrule, KDyrule):

    # Unpack the vector
    #print nxy_start[0,0]
    nx_start = nxy_start.item(0)
    ny_start = nxy_start.item(1)
    #print "incoming", nx_start, ny_start

    # Get the starting amplitudes
    cA_start = Arule(nx_start,ny_start)
    cB_start = Brule(nx_start,ny_start)
    cC_start = Crule(nx_start,ny_start)
    cD_start = Drule(nx_start,ny_start)
    c_start = np.matrix([cA_start,cB_start,cC_start,cD_start], copy=False)

    # Construct the starting K-vectors
    KAx_start = KAxrule(nx_start,ny_start)
    KAy_start = KAyrule(nx_start,ny_start)
    KBx_start = KBxrule(nx_start,ny_start)
    KBy_start = KByrule(nx_start,ny_start)
    KCx_start = KCxrule(nx_start,ny_start)
    KCy_start = KCyrule(nx_start,ny_start)
    KDx_start = KDxrule(nx_start,ny_start)
    KDy_start = KDyrule(nx_start,ny_start)
    Kx = np.matrix([KAx_start,KBx_start,KCx_start,KDx_start], copy=False)
    Ky = np.matrix([KAy_start,KBy_start,KCy_start,KDy_start], copy=False)
    K = np.hstack((Kx,Ky))
    # print Kx
    # print Ky
    #print K
    return K, c_start

def getdots(nvec,dvec,beamvec,hvec,stagetheta,axis,method,detector,facet):

    # Rotate the crystal stage
    if stagetheta == 0:
        nvec_new = nvec
    else:
        rotmat = rotation_matrix(axis, stagetheta)
        nvec_new = np.dot(rotmat, nvec)

    # Get normal * detector dot product
    nvec_dot_dvec = np.inner(nvec_new.T,dvec.T)

    # Get normal * beam dot products
    nvec_dot_beamvec = np.inner(nvec_new.T,beamvec.T)

    # Get reflection vector
    rvec = -(beamvec - 2*nvec_new*nvec_dot_beamvec)

    # Get reflection * detector dot product
    rvec_dot_dvec = np.inner(rvec.T,dvec.T)

    # Get normal * h dot product
    nvec_dot_hvec = np.inner(nvec_new.T,hvec.T)

    # Append to the backscatter list
    if method == 'blinn-phong':
        ks = .2
        BSnext = rBS(nvec_dot_beamvec,nvec_dot_hvec,ks)
    elif method == 'phong':
        ks = .2
        BSnext = rBS(nvec_dot_beamvec,rvec_dot_dvec,ks)
    elif method == 'butterfield':
        eta = 1/(1+nvec_dot_beamvec)**2.1
        BSnext = eta*nvec_dot_hvec**(1.0/nvec_dot_beamvec)+1
    elif method == 'hitachiBS':
        s = nvec_dot_dvec-nvec_dot_beamvec
        #BSnext = 120*s+48.5
        #BSnext = 100*s+135
        BSnext = 194*s + 90
    elif method == 'hitachiBS2':
        s = nvec_dot_dvec-nvec_dot_beamvec
        if detector == 'A':
            BSnext = 192*s + 97 + 229*s**2
        if detector == 'B':
            BSnext = 188*s + 78 + 56*s**2
        if detector == 'C':
            BSnext = 204*s + 81 + 231*s**2
        if detector == 'D':
            BSnext = 193*s + 94 + 17*s**2
    elif method == 'hitachiBS25':
        s = (nvec_dot_dvec-nvec_dot_beamvec)
        c1A = 200; c2A = 100 
        c1C = 200; c2C = 100 
        c1B = 200; c2B = 100
        c1D = 200; c2D = 100
        c0 = 80
        if detector == 'A':
            BSnext = 97 + c1A*s + c2A*s**2
        elif detector == 'B':
            BSnext = 78 + c1B*s + c2B*s**2
        elif detector == 'C':
            BSnext = 81 + c1C*s + c2C*s**2
        elif detector == 'D':
            BSnext = 94 + c1D*s + c2D*s**2
        else:
            print ('Should not have gotten here')
    else:
        print ('Not implemented ...')
    return nvec_dot_dvec, nvec_dot_beamvec, rvec_dot_dvec, nvec_dot_hvec, BSnext


def getBSgrids(npts, backoff, dvecs,beamvec,hvecs,method):
# Response functions at detectors


    nxi, nyi = np.linspace(-1+backoff,1-backoff,npts), np.linspace(-1+backoff,1-backoff,npts)
    dnx = nxi[1]-nxi[0]
    dny = nyi[1]-nyi[0]
    nxigrid, nyigrid = np.meshgrid(nxi, nyi)
    nzigrid_squared = 1 - (nxigrid**2+nyigrid**2)
    mask = nzigrid_squared>0
    nzigrid = np.zeros(np.shape(nxigrid))
    nzigrid[mask]=np.sqrt(nzigrid_squared[mask])
    nvec = matrix([nxigrid[mask],nyigrid[mask],nzigrid[mask]]).astype('float64').T
    detectors = ['A','B','C','D']
    
    '''
    # Set aside space to put the dot products and BS response
    nvec_dot_dXvec = np.zeros(len(nvec))
    nvec_dot_beamvec = np.zeros(len(nvec))
    rvec_dot_dXvec = np.zeros(len(nvec))
    nvec_dot_hXvec = np.zeros(len(nvec))
    BS_of_n = np.zeros(len(nvec))

    # Calculate dot products and BS response
    detectors = ['A','B','C','D']
    for j in range(4):
        i = 0
        for nveci in nvec:
            a, b, c, d, e = getdots(nveci.T,dvecs[j],beamvec,hvecs[j],0,0,method,detectors[j],'a')
            #a, b, c, d, e = getdots(nveci.T,dvecs[1],beamvec,hvecs[1],0,0,method,'B','a')
            #a, b, c, d, e = getdots(nveci.T,dvecs[2],beamvec,hvecs[2],0,0,method,'C','a')
            #a, b, c, d, e = getdots(nveci.T,dvecs[3],beamvec,hvecs[3],0,0,method,'D','a')
            nvec_dot_dXvec[i] = a
            nvec_dot_beamvec[i] = b
            rvec_dot_dXvec[i] = c
            nvec_dot_hXvec[i] = d 
            if nvec_dot_dXvec[i] <= 0 or nvec_dot_beamvec[i]<= 0: # The detector can't see the surface
                BS_of_n[i] = 0
            else:
                BS_of_n[i] = e    
            i+=1
            
        if j == 0:
            BSgridA = BSgrid.T
        if j == 1:
            BSgridB = BSgrid.T
        if j == 2:
            BSgridC = BSgrid.T
        if j == 3:
            BSgridD = BSgrid.T        
    '''        
    for j in range(4):
        filename = 'BSgrid' + detectors[j] + '_' + method +'.txt'
        if j == 0:
            BSgridA = np.loadtxt(filename)
        if j == 1:
            BSgridB = np.loadtxt(filename)
        if j == 2:
            BSgridC = np.loadtxt(filename)
        if j == 3:
            BSgridD = np.loadtxt(filename)
    
    return BSgridA, BSgridB, BSgridC, BSgridD, nxi, nyi, dnx, dny, nxigrid, nyigrid


def getBSgrids_asymmetric(nptsx, nptsy, nmax, dvecs, beamvec, hvecs, method):
# Response functions at detectors w/ different number of pts in x and y

    nxi, nyi = np.linspace(-nmax,nmax,nptsx), np.linspace(-nmax,nmax,nptsy)
    dnx = nxi[1]-nxi[0]
    dny = nyi[1]-nyi[0]
    nxigrid, nyigrid = np.meshgrid(nxi, nyi)
    nzigrid_squared = 1 - (nxigrid**2+nyigrid**2)
    mask = nzigrid_squared>0
    nzigrid = np.zeros(np.shape(nxigrid)); print (shape(nzigrid))
    nzigrid[mask]=np.sqrt(nzigrid_squared[mask])
    nvec = matrix([nxigrid[mask],nyigrid[mask],nzigrid[mask]]).astype('float64').T; print (shape(nvec))

    # Set aside space to put the dot products and BS response
    nvec_dot_dXvec = np.zeros(len(nvec))
    nvec_dot_beamvec = np.zeros(len(nvec))
    rvec_dot_dXvec = np.zeros(len(nvec))
    nvec_dot_hXvec = np.zeros(len(nvec))
    BS_of_n = np.zeros(len(nvec))
    print (shape(BS_of_n))

    # Calculate dot products and BS response
    detectors = ['A','B','C','D']
    for j in range(4):
        i = 0
        for nveci in nvec:
            a, b, c, d, e = getdots(nveci.T,dvecs[j],beamvec,hvecs[j],0,0,method,detectors[j],'a')
            nvec_dot_dXvec[i] = a
            nvec_dot_beamvec[i] = b
            rvec_dot_dXvec[i] = c
            nvec_dot_hXvec[i] = d 
            if nvec_dot_dXvec[i] <= 0 or nvec_dot_beamvec[i]<= 0: # The detector can't see the surface
                BS_of_n[i] = 0
            else:
                BS_of_n[i] = e    
            i+=1
    
        # Interpolate; there's also method='cubic'
        print (shape((nxigrid,nyigrid)))
        BSgrid = scipy.interpolate.griddata((nxigrid[mask],nyigrid[mask]), 
                                    BS_of_n, (nxigrid, nyigrid), method='linear')
        nanmask = np.isnan(BSgrid)
        BSgrid[nanmask] = 0
    
        if j == 0:
            BSgridA = BSgrid.T
        if j == 1:
            BSgridB = BSgrid.T
        if j == 2:
            BSgridC = BSgrid.T
        if j == 3:
            BSgridD = BSgrid.T

    return BSgridA, BSgridB, BSgridC, BSgridD, nxi, nyi, dnx, dny, nxigrid, nyigrid
    



def getBSgrids_asymmetricgrad(nptsx, nptsy, nmax, dvecs,beamvec,hvecs,method):
# Response functions at detectors w/ different number of pts in x and y, normalized as gradient (nz=1)

    nxi, nyi = np.linspace(-nmax,nmax,nptsx), np.linspace(-nmax,nmax,nptsy)
    dnx = nxi[1]-nxi[0]
    dny = nyi[1]-nyi[0]
    nxigrid, nyigrid = np.meshgrid(nxi, nyi)
    nzigrid = np.ones(np.shape(nxigrid)); 
    mask = nzigrid>0
    temp = [nxigrid[mask],nyigrid[mask],nzigrid[mask]]
    nvec = matrix(temp).astype('float64').T; 

    # Set aside space to put the dot products and BS response
    nvec_dot_dXvec = np.zeros(len(nvec))
    nvec_dot_beamvec = np.zeros(len(nvec))
    rvec_dot_dXvec = np.zeros(len(nvec))
    nvec_dot_hXvec = np.zeros(len(nvec))
    BS_of_n = np.zeros(len(nvec))

    # Calculate dot products and BS response
    detectors = ['A','B','C','D']
    for j in range(4):
        i = 0
        for nveci in nvec:
            a, b, c, d, e = getdots(nveci.T,dvecs[j],beamvec,hvecs[j],0,0,method,detectors[j],'a')
            nvec_dot_dXvec[i] = a
            nvec_dot_beamvec[i] = b
            if nvec_dot_dXvec[i] <= 0 or nvec_dot_beamvec[i]<= 0: # The detector can't see the surface
                BS_of_n[i] = 0
            else:
                BS_of_n[i] = e    
            i+=1
    
        # Interpolate; there's also method='cubic'
        BSgrid = scipy.interpolate.griddata((nxigrid[mask],nyigrid[mask]), 
                                    BS_of_n, (nxigrid, nyigrid), method='linear')
        nanmask = np.isnan(BSgrid)
        BSgrid[nanmask] = 0
    
        # Copy into the appropriate backscatter holding arrays
        if j == 0:
            BSgridA = BSgrid.T
        if j == 1:
            BSgridB = BSgrid.T
        if j == 2:
            BSgridC = BSgrid.T
        if j == 3:
            BSgridD = BSgrid.T

    # Implement a little bit of bleeding of one detector onto another
    sidebleed = 0.2
    tempBSgridA = (1-2*sidebleed)*BSgridA + sidebleed*BSgridB + sidebleed*BSgridD 
    tempBSgridB = (1-2*sidebleed)*BSgridB + sidebleed*BSgridC + sidebleed*BSgridA 
    tempBSgridC = (1-2*sidebleed)*BSgridC + sidebleed*BSgridD + sidebleed*BSgridB 
    tempBSgridD = (1-2*sidebleed)*BSgridD + sidebleed*BSgridA + sidebleed*BSgridC
    
    BSgridA = tempBSgridA
    BSgridB = tempBSgridB
    BSgridC = tempBSgridC
    BSgridD = tempBSgridD
    
    return BSgridA, BSgridB, BSgridC, BSgridD, nxi, nyi, dnx, dny, nxigrid, nyigrid
    

    
def setupdetectorresponse(\
         BSgridA, BSgridB, BSgridC, BSgridD,
         nxi, nyi, dnx, dny):

    # These are rules for interpolating the response functions
    Arule = scipy.interpolate.interp2d(nxi, nyi, BSgridA.T, kind='linear')
    Brule = scipy.interpolate.interp2d(nxi, nyi, BSgridB.T, kind='linear')
    Crule = scipy.interpolate.interp2d(nxi, nyi, BSgridC.T, kind='linear')
    Drule = scipy.interpolate.interp2d(nxi, nyi, BSgridD.T, kind='linear')

    # Calculating the gradients of the response functions
    KAx, KAy = np.gradient(BSgridA); KAy=KAy/dny; KAx = KAx/dnx
    KBx, KBy = np.gradient(BSgridB); KBy=KBy/dny; KBx = KBx/dnx
    KCx, KCy = np.gradient(BSgridC); KCy=KCy/dny; KCx = KCx/dnx
    KDx, KDy = np.gradient(BSgridD); KDy=KDy/dny; KDx = KDx/dnx

    # These are rules for interpolating the response functions
    KAxrule = scipy.interpolate.interp2d(nxi, nyi, KAx.T, kind='linear')
    KAyrule = scipy.interpolate.interp2d(nxi, nyi, KAy.T, kind='linear')
    KBxrule = scipy.interpolate.interp2d(nxi, nyi, KBx.T, kind='linear')
    KByrule = scipy.interpolate.interp2d(nxi, nyi, KBy.T, kind='linear')
    KCxrule = scipy.interpolate.interp2d(nxi, nyi, KCx.T, kind='linear')
    KCyrule = scipy.interpolate.interp2d(nxi, nyi, KCy.T, kind='linear')
    KDxrule = scipy.interpolate.interp2d(nxi, nyi, KDx.T, kind='linear')
    KDyrule = scipy.interpolate.interp2d(nxi, nyi, KDy.T, kind='linear')

    return \
    Arule, Brule, Crule, Drule,\
    KAxrule, KAyrule, KBxrule, KByrule, KCxrule, KCyrule, KDxrule, KDyrule


def setupdetectorresponse2(\
         BSgridA, BSgridB, BSgridC, BSgridD,
         nxi, nyi, dnx, dny):

    # Same as the old setupdetectorresponse, but with a transposed BSgrid coming in
    
    # These are rules for interpolating the response functions
    Arule = scipy.interpolate.interp2d(nxi, nyi, BSgridA, kind='linear')
    Brule = scipy.interpolate.interp2d(nxi, nyi, BSgridB, kind='linear')
    Crule = scipy.interpolate.interp2d(nxi, nyi, BSgridC, kind='linear')
    Drule = scipy.interpolate.interp2d(nxi, nyi, BSgridD, kind='linear')

    # Calculating the gradients of the response functions
    KAx, KAy = np.gradient(BSgridA.T); KAy=KAy/dny; KAx = KAx/dnx
    KBx, KBy = np.gradient(BSgridB.T); KBy=KBy/dny; KBx = KBx/dnx
    KCx, KCy = np.gradient(BSgridC.T); KCy=KCy/dny; KCx = KCx/dnx
    KDx, KDy = np.gradient(BSgridD.T); KDy=KDy/dny; KDx = KDx/dnx

    # These are rules for interpolating the response functions
    KAxrule = scipy.interpolate.interp2d(nxi, nyi, KAx.T, kind='linear')
    KAyrule = scipy.interpolate.interp2d(nxi, nyi, KAy.T, kind='linear')
    KBxrule = scipy.interpolate.interp2d(nxi, nyi, KBx.T, kind='linear')
    KByrule = scipy.interpolate.interp2d(nxi, nyi, KBy.T, kind='linear')
    KCxrule = scipy.interpolate.interp2d(nxi, nyi, KCx.T, kind='linear')
    KCyrule = scipy.interpolate.interp2d(nxi, nyi, KCy.T, kind='linear')
    KDxrule = scipy.interpolate.interp2d(nxi, nyi, KDx.T, kind='linear')
    KDyrule = scipy.interpolate.interp2d(nxi, nyi, KDy.T, kind='linear')

    return \
    Arule, Brule, Crule, Drule,\
    KAxrule, KAyrule, KBxrule, KByrule, KCxrule, KCyrule, KDxrule, KDyrule


def GaussNewton2(\
             c_obs, Se_inv, Sa_inv,
             Arule, Brule, Crule, Drule,
             KAxrule, KAyrule,
             KBxrule, KByrule,
             KCxrule, KCyrule,
             KDxrule, KDyrule):


    # Find a good starting point
    error_least = 100.
    for r in np.linspace(0,.8,3):
        nphis = max(1,round(7*r))
        for phi in np.linspace(.1,2*np.pi,nphis,endpoint=False):
            nx_start = r*np.cos(phi)
            ny_start = r*np.sin(phi)
            nxy_start = np.matrix([nx_start,ny_start]).T
            K_start, c_start = getK(nxy_start,
                 Arule, Brule, Crule, Drule,
                 KAxrule, KAyrule,
                 KBxrule, KByrule,
                 KCxrule, KCyrule,
                 KDxrule, KDyrule)
            #plt.plot(nxy_start.item(0),nxy_start.item(1),'kx')
            error = norm(c_start-c_obs)
            if error<error_least:
                error_least = copy.copy(error)
                nxy_best = copy.copy(nxy_start)
    nxy_last = nxy_best
    K_last, c_last = getK(nxy_last,
             Arule, Brule, Crule, Drule,
             KAxrule, KAyrule,
             KBxrule, KByrule,
             KCxrule, KCyrule,
             KDxrule, KDyrule)
    K_i = K_last
    #plt.plot(nxy_last.item(0),nxy_last.item(1),'ro')


    # Iterate to retrieve nx, ny
    I_converged = False
    for i in range(10):

        # Construct a delta-c vector
        delta_c = c_obs-c_last; #print delta_c

        # This implements optimal method as above, but w/o matrix inversion
        term0 = K_i.T*Se_inv*K_i
        term2 = Sa_inv+term0
        xa = 0
        term3 = K_i.T*Se_inv*(delta_c+K_i*(nxy_last-xa))
        nxy_next = xa + np.linalg.solve(term2,term3) # same as term2\term3

        # Test for convergence (Rodgers p. 90, Eqn 5.31)
        dnx = nxy_next - nxy_last
        di2 = dnx.T * term2 * dnx

        # Update normals and K- and c-values
        nxy_last = copy.copy(nxy_next)
        K_last, c_last = getK(nxy_last,
             Arule, Brule, Crule, Drule,
             KAxrule, KAyrule,
             KBxrule, KByrule,
             KCxrule, KCyrule,
             KDxrule, KDyrule)
        K_i = K_last # Just a notational convenience

        # Decide to continue or not
        if di2 < 0.01:
            nxy_last.item(0), nxy_last.item(1), \
            nxy_last.item(0)**2+nxy_last.item(1)**2
            I_converged = True
            break

    return nxy_last.item(0), nxy_last.item(1), I_converged


def getcrossindices(nx,ny):
    # note that what gets called nx & ny depends on caller definitions
    if ny==nx:
        NV = 2*nx-3
        Vacross = [0 for i in range(NV)]
        for index in range(NV):
            if index<nx-1:
                nindex = index+1
                Vacross[index] = zeros((nindex,2),dtype=int)
                for j in range(nindex):
                    iy = j; ix = index-iy
                    Vacross[index][j][1]=iy
                    Vacross[index][j][0]=ix
            else:
                nindex = NV-index
                Vacross[index] = zeros((nindex,2),dtype=int)
                for j in range(nindex):
                    ix = nx-j-2; iy = index-ix
                    Vacross[index][j][1]=iy
                    Vacross[index][j][0]=ix

    elif ny>nx:
        NV = 2*nx-3+(ny-nx)
        Vacross = [0 for i in range(NV)]
        for index in range(NV):
            if index<nx-1:
                nindex = index+1
                Vacross[index] = zeros((nindex,2),dtype=int)
                for j in range(nindex):
                    iy = j; ix = index-iy
                    Vacross[index][j][1]=iy
                    Vacross[index][j][0]=ix
            elif index >= ny-1:
                nindex = NV-index
                Vacross[index] = zeros((nindex,2),dtype=int)
                for j in range(nindex):
                    ix = nx-j-2; iy = index-ix
                    Vacross[index][j][1]=iy
                    Vacross[index][j][0]=ix
            else:
                nindex = nx-1
                Vacross[index] = zeros((nindex,2),dtype=int)
                for j in range(nindex):
                    ix = nx-j-2; iy = index-ix
                    Vacross[index][j][1]=iy
                    Vacross[index][j][0]=ix


    else:
        NV = 2*nx-3-(nx-ny)
        Vacross = [0 for i in range(NV)]
        for index in range(NV):
            if index<ny-1:
                nindex = index+1
                Vacross[index] = zeros((nindex,2),dtype=int)
                for j in range(nindex):
                    iy = j; ix = index-iy
                    Vacross[index][j][1]=iy
                    Vacross[index][j][0]=ix
            elif index >= nx-1:
                nindex = NV-index
                Vacross[index] = zeros((nindex,2),dtype=int)
                for j in range(nindex):
                    ix = nx-j-2; iy = index-ix
                    Vacross[index][j][1]=iy
                    Vacross[index][j][0]=ix
            else:
                nindex = ny-1
                Vacross[index] = zeros((nindex,2),dtype=int)
                for j in range(nindex):
                    ix = index-j; iy = index-ix
                    Vacross[index][j][1]=iy
                    Vacross[index][j][0]=ix
    return Vacross

def getbigK(\
            nx, ny, nXobs, surf_dzgrid_dx_retrieved, surf_dzgrid_dy_retrieved,
            Arule, Brule, Crule, Drule,
            KAxrule, KAyrule,
            KBxrule, KByrule,
            KCxrule, KCyrule,
            KDxrule, KDyrule):
    bigKx = matrix(empty((4,0)), copy=False)
    bigKy = matrix(empty((4,0)), copy=False)
    bigc_last = matrix(empty((4,0)), copy=False)
    for iy in range(ny-1):
        for ix in range(nx-1):
            dzdx = surf_dzgrid_dx_retrieved[iy,ix]
            dzdy = surf_dzgrid_dy_retrieved[iy,ix]
            nxy_last = vstack((dzdx,dzdy))
            K_i, c_last = getK(nxy_last,
            Arule, Brule, Crule, Drule,
            KAxrule, KAyrule,
            KBxrule, KByrule,
            KCxrule, KCyrule,
            KDxrule, KDyrule)
            bigKx = hstack((bigKx,K_i[:,0]))
            bigKy = hstack((bigKy,K_i[:,1]))
            bigc_last = hstack((bigc_last,c_last))
    vbigKx = matrix(zeros((nXobs*4,nXobs)), copy=False)
    vbigKy = matrix(zeros((nXobs*4,nXobs)), copy=False)
    vbigKx[0:4,:] = bigKx
    vbigKy[0:4,:] = bigKy
    for col in range(nXobs):
        vbigKx[:,col] = roll(vbigKx[:,col],4*col,axis=0)
        vbigKy[:,col] = roll(vbigKy[:,col],4*col,axis=0)
    vbigK = hstack((vbigKx, vbigKy))
    return vbigK, bigc_last




def getnextz(\
            z_last,za, c_obs_long,\
            Nx,Ny,nx,ny,nXobs,\
            Se_inv,Sa_inv,\
            Arule, Brule, Crule, Drule,\
            KAxrule, KAyrule, \
            KBxrule, KByrule, \
            KCxrule, KCyrule, \
            KDxrule, KDyrule):


    nx_last_long = ags.dot(Nx,z_last)
    ny_last_long = ags.dot(Ny,z_last)
    nx_last = matrix(reshape(nx_last_long,(ny-1,nx-1)), copy=False);
    ny_last = matrix(reshape(ny_last_long,(ny-1,nx-1)), copy=False);

    vbigK, bigc_last = getbigK(\
            nx,ny,nXobs,
            nx_last, ny_last,
            Arule, Brule, Crule, Drule,
            KAxrule, KAyrule,
            KBxrule, KByrule,
            KCxrule, KCyrule,
            KDxrule, KDyrule)

    NxNy = vstack((Nx,Ny))

    KN = ags.dot(vbigK, NxNy)


    nobs = nXobs*4   # int*int
    bigc_last_long = matrix(reshape(ags.T(bigc_last),(nobs,1)), copy=False)
    delta_c = ags.substract(c_obs_long, bigc_last_long)

#     # This is the Gauss method
#     dz = linalg.inv(KN.T*KN)*KN.T*delta_c
#     z_next = z_last + dz
#     di2 = squeeze(dot(dz.T,dz))

    # This is the optimal estimation method

    # nasledujici dva radky by se daly mergnout
    
    # term0 = KN.T * Se^-1 * KN
    term0 = ags.dot3(Se_inv, KN)
    
    # term2 = Sa^-1 + term0
    term2 = Sa_inv+term0
    
    # dz = z_last - za
    # dc = c_obs - c_last
    # term5 = dc + KN*dz
    # term4 = Se^-1 * term5
    # term3 = KN.T * term4
    term3 = ags.dot2(KN, ags.dot(Se_inv, (delta_c+ags.dot(KN,(z_last-za)))))
    z_next = za + np.linalg.solve(term2,term3) # same as term2\term3
    dz = z_next - z_last
    di2 = ags.dot3(term2, dz)[0,0]
    
    # Some error estimates
    print ('<diff>, std(diff), di2 =', np.mean(delta_c), np.std(delta_c), di2)

    # Get out
    return z_next, di2

def getnextz_iterate(\
            z_start, za, c_obs_long, maxiter, tolerance, \
            Nx,Ny,nx,ny,nXobs,\
            Se_inv,Sa_inv,\
            Arule, Brule, Crule, Drule,\
            KAxrule, KAyrule, \
            KBxrule, KByrule, \
            KCxrule, KCyrule, \
            KDxrule, KDyrule):

    z_last = copy.copy(z_start)
    for i in range(maxiter):
        z_next, di2 = getnextz(
            z_last, za, c_obs_long,\
            Nx,Ny,nx,ny,nXobs,\
            Se_inv,Sa_inv,
            Arule, Brule, Crule, Drule,\
            KAxrule, KAyrule, \
            KBxrule, KByrule, \
            KCxrule, KCyrule, \
            KDxrule, KDyrule)
        z_last = copy.copy(z_next)
        #print ("iteration, di2", i, di2)
        if di2<tolerance:
             break
    return z_next


def retrievesegment(\
    nx1,ny1,nx2,ny2,cA,cB,cC,cD,\
    Sa,Se,z_start,maxiter,tolerance,\
    Nx,Ny,\
    Arule, Brule, Crule, Drule,\
    KAxrule, KAyrule, \
    KBxrule, KByrule, \
    KCxrule, KCyrule, \
    KDxrule, KDyrule):

    # Make the starting value the a priori
    za = copy.copy(z_start)

    # Extract the image data of the subset of interest
    cA_obs = cA[ny1:ny2,nx1:nx2].astype('float')
    cB_obs = cB[ny1:ny2,nx1:nx2].astype('float')
    cC_obs = cC[ny1:ny2,nx1:nx2].astype('float')
    cD_obs = cD[ny1:ny2,nx1:nx2].astype('float')

    # Pack these into a vector
    nXobs = np.size(cA_obs)
    #nobs = nXobs*4; print "# of observations", nobs
    caxis = 0 # This was 1
    cA_obs_long = reshape(cA_obs,nXobs,caxis)
    cB_obs_long = reshape(cB_obs,nXobs,caxis)
    cC_obs_long = reshape(cC_obs,nXobs,caxis)
    cD_obs_long = reshape(cD_obs,nXobs,caxis)
    c_obs_stacked = vstack((cA_obs_long,cB_obs_long,cC_obs_long,cD_obs_long))
    c_obs_long = matrix(reshape(c_obs_stacked.T,(nXobs*4,1)), copy=False)

    # Print some statistics of the observations
    print ("Observed intensities (detector B):")
    print ("mean, max, min =", np.mean(cB_obs_long),np.max(cB_obs_long),np.min(cB_obs_long))

    # Set-up for the inversion
    Sa_inv = np.diag(1/np.diag(Sa))
    Se_inv = np.diag(1/np.diag(Se))

    # Height retrieval
    nx = nx2-nx1+1
    ny = ny2-ny1+1
    z_next = getnextz_iterate(\
            z_start, za, c_obs_long, maxiter, tolerance,\
            Nx,Ny,nx,ny,nXobs,\
            Se_inv,Sa_inv,\
            Arule, Brule, Crule, Drule,\
            KAxrule, KAyrule, \
            KBxrule, KByrule, \
            KCxrule, KCyrule, \
            KDxrule, KDyrule)
    z_retrieved_long = vstack((z_next,z_next[-1]))
    z_retrieved = reshape(z_retrieved_long,(ny,nx))

    # Done
    return z_retrieved

#def getnextz_iteratewithpseudoinverse(\
#            z_start, c_obs_long, maxiter, tolerance, \
#            Nx,Ny,nx,ny,nXobs,\
#            Se_inv,Sa_inv,\
#            Arule, Brule, Crule, Drule,\
#            KAxrule, KAyrule, \
#            KBxrule, KByrule, \
#            KCxrule, KCyrule, \
#            KDxrule, KDyrule):
#
#    # Housekeeping
#    nobs = nXobs*4
#    NxNy = vstack((Nx,Ny))
#
#    # Copy of the latest solution vector
#    z_last = copy.copy(z_start)
#
#    # Solve for the pseudoinverse
#    nx_last_long = Nx*z_last
#    ny_last_long = Ny*z_last
#    nx_last = matrix(reshape(nx_last_long,(ny-1,nx-1)));
#    ny_last = matrix(reshape(ny_last_long,(ny-1,nx-1)));
#    vbigK, bigc_last = getbigK(\
#            nx,ny,nXobs,
#            nx_last, ny_last,
#            Arule, Brule, Crule, Drule,
#            KAxrule, KAyrule,
#            KBxrule, KByrule,
#            KCxrule, KCyrule,
#            KDxrule, KDyrule)
#    NxNy = vstack((Nx,Ny))
#    KN = vbigK*NxNy
#    term0 = KN.T*Se_inv*KN
#    term2 = Sa_inv+term0
#    term2_pinv = np.linalg.pinv(term2)
#
#    # Now iterate
#    for i in range(maxiter):
#
#        nx_last_long = Nx*z_last
#        ny_last_long = Ny*z_last
#        nx_last = matrix(reshape(nx_last_long,(ny-1,nx-1)));
#        ny_last = matrix(reshape(ny_last_long,(ny-1,nx-1)));
#
#        vbigK, bigc_last = getbigK(\
#                nx,ny,nXobs,
#                nx_last, ny_last,
#                Arule, Brule, Crule, Drule,
#                KAxrule, KAyrule,
#                KBxrule, KByrule,
#                KCxrule, KCyrule,
#                KDxrule, KDyrule)
#
#        KN = vbigK*NxNy
#        bigc_last_long = matrix(reshape(bigc_last.T,(nobs,1)))
#        delta_c = c_obs_long - bigc_last_long
#
#        # This is the other method
#        term3 = KN.T*Se_inv*(delta_c+KN*z_last)
#        z_next = term2_pinv*term3
#        #z_next = np.linalg.solve(term2,term3) # same as term2\term3
#        dz = z_next - z_last
#        di2 = dz.T * term2 * dz
#
#        # Get out
#        z_last = copy.copy(z_next)
#        print "iteration, di2", i, di2
#        if di2<tolerance:
#             break
#
#    return z_next
#

def retrieveall(\
        nmax,maxiter,tolerance,rootnoiseamp,rootapriorivar0,\
        nsegments,nptsx,nptsy,nx1list,ny1list,nx2list,ny2list,nyxgrid,dx,dy,\
        Calibration,cA,cB,cC,cD):
    
    # Extract calibration data
    pA=Calibration['Calibration']['pA']
    pB=Calibration['Calibration']['pB']
    pC=Calibration['Calibration']['pC']
    pD=Calibration['Calibration']['pD']

    
    # Set up a grid of surface normal vectors and the backscatter response on them
    nxmid = int(nptsx/2); #print nxmid
    nymid = int(nptsy/2); #print nymid
    nxi = np.linspace(-nmax,nmax,nptsx); dnx = nxi[1]-nxi[0]
    nyi = np.linspace(-nmax,nmax,nptsy); dny = nyi[1]-nyi[0]
    nxigrid,nyigrid = np.meshgrid(nxi,nyi)
    theta = 15*np.pi/180
    sA = (-nxigrid*np.sin(theta)+np.cos(theta)-1)/(1+nxigrid**2+nyigrid**2)**.5
    sB = (-nyigrid*np.sin(theta)+np.cos(theta)-1)/(1+nxigrid**2+nyigrid**2)**.5
    sC = (+nxigrid*np.sin(theta)+np.cos(theta)-1)/(1+nxigrid**2+nyigrid**2)**.5
    sD = (+nyigrid*np.sin(theta)+np.cos(theta)-1)/(1+nxigrid**2+nyigrid**2)**.5

    # Set up the grids     
    BSgridA = np.polyval(pA,sA)
    BSgridB = np.polyval(pB,sB)
    BSgridC = np.polyval(pC,sC)
    BSgridD = np.polyval(pD,sD)
                
    # Generating the response function for each detector
    BSgridN = [BSgridA, BSgridB, BSgridC, BSgridD]
    BSgridL = ['A', 'B', 'C', 'D']
    BSmax = 150 # this for display purposes
    print('nxigrid.shape=',nxigrid.shape)

    # Set up interpolators for detector responses
    Arule, Brule, Crule, Drule, \
    KAxrule, KAyrule, KBxrule, KByrule, KCxrule, KCyrule, KDxrule, KDyrule =\
    setupdetectorresponse2(BSgridA, BSgridB, BSgridC, BSgridD, nxi, nyi, dnx, dny)
                
    # Create a blank slate
    solutionshape = cA.shape
    solution = np.zeros(solutionshape)
                
    # Define the variance in the observations (BS units^2)
    #print('Std deviation in input signal is', rootnoiseamp)
    noiseamp = rootnoiseamp**2

    # Define parameters determining the variance in the a priori (microns^2)
    #print('Std deviation in a priori is', rootapriorivar0)
    apriorivar0 = rootapriorivar0**2

    # Create the initial a priori variance
    apriorivar = np.ones(cA.shape)*apriorivar0

    # Create the initial a priori set
    aprioriset = np.zeros(cA.shape)
                
    # Loop to retrieve each segment
    for isegment in range(nsegments):

        # Choose the particular location of the dataset to analyze
        nx1=nx1list[isegment]; nx2=nx2list[isegment]; nx = nx2-nx1+1
        ny1=ny1list[isegment]; ny2=ny2list[isegment]; ny = ny2-ny1+1

        # Construct gradients
        Ny_unscaled, Nx_unscaled = gds.makeNxNy(ny,nx)
        Ny = Ny_unscaled/dy
        Nx = -Nx_unscaled/dx #fixing x inversion

        # Number of observations
        nobs = (nx-1)*(ny-1)*4

        # Number of desired points (heights)
        nzpts = ny*nx-1

        # Extract the a priori variance
        vartemp = apriorivar[ny1:ny2+1,nx1:nx2+1]
        vartemp_long = np.reshape(vartemp,nzpts+1,0)
        Sa = np.diag(vartemp_long[:-1]); #print "apriorivar", shape(Sa)

        # Extract the starting z
        settemp = solution[ny1:ny2+1,nx1:nx2+1]
        settemp_long = np.reshape(settemp,nzpts+1,0)
        settemp_longminus1 = settemp_long[:-1]
        z_start = np.matrix(settemp_longminus1).T; #print "aprioriset", shape(z_start)
        z_start = z_start*0.0; #print "aprioriset", shape(z_start)

        # Construct the variance in observation + model
        Se = np.matrix(np.eye(nobs))*noiseamp # Variance in observation + model (c)

        # Do the retrieval
        print('')
        print("for", nx1, ny1)
        print("Segment:", isegment+1, "of", nsegments)
        z_retrieved = retrievesegment(\
            nx1,ny1,nx2,ny2,cA,cB,cC,cD,\
            Sa,Se,z_start,maxiter,tolerance,\
            Nx,Ny,\
            Arule, Brule, Crule, Drule,\
            KAxrule, KAyrule, \
            KBxrule, KByrule, \
            KCxrule, KCyrule, \
            KDxrule, KDyrule)

        if isegment == 0:
            solution[ny1:ny2+1,nx1:nx2+1] = copy.copy(z_retrieved)
        else:
            nextsolution = np.zeros(cA.shape)
            nextsolution[ny1:ny2+1,nx1:nx2+1] = copy.copy(z_retrieved)
            overlap = []
            for i in range(isegment):
                nextoverlap = list( set(nyxgrid[i])&set(nyxgrid[isegment]) )
                overlap = overlap + nextoverlap
                Noverlap = len(overlap); 
            print("Noverlap =", Noverlap)
            diff = 0.0
            for j in range(Noverlap):
                diff += nextsolution[overlap[j]] - solution[overlap[j]]
            diffavg = diff/Noverlap
            z_retrieved -= diffavg
            solution[ny1:ny2+1,nx1:nx2+1] = copy.copy(z_retrieved)
            
    return solution