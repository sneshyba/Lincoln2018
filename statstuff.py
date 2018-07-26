# Code for roughness statistics
import numpy as np

def pWeibull(r, sigma, eta):
    ''' Weibull function '''
    from numpy import exp
    mu = 1-r
    ret = 2*eta/sigma**2/mu**3 * \
        (((mu**(-2)-1)/sigma**2)**(eta-1)) * \
        exp(-((mu**(-2)-1)/sigma**2)**eta)
    return ret

def pWeibullr(r, sigma, eta):
    ''' Weibull function times r '''
    return pWeibull(r, sigma, eta)*r

def pGaussian(r, sigma):
    ''' Gaussian function '''
    return pWeibull(r, sigma, 1)    

def pGaussianr(r, sigma):
    ''' Gaussian function times r '''
    return pWeibullr(r, sigma, 1)

def bimodal(r, sigma1, sigma2, N):
    ''' Bimodal Gaussian function '''
    pdf1 = pWeibull(r,sigma1,1.0)
    pdf2 = pWeibull(r,sigma2,1.0)
    return (1-N)*pdf1 + N*pdf2 

def bimodalr(r, sigma1, sigma2, N):
    ''' Bimodal Gaussian function times r'''
    pdf1 = pWeibullr(r,sigma1,1.0)
    pdf2 = pWeibullr(r,sigma2,1.0)
    return (1-N)*pdf1 + N*pdf2 


def bimodalfunc(r, sigma1, sigma2, N):
    ''' Bimodal Gaussian function '''
    pdf1 = pWeibullr(r,sigma1,1.0)
    pdf2 = pWeibullr(r,sigma2,1.0)
    return (1-N)*pdf1 + N*pdf2 


def sigma2meanr(sigma):
    ''' Converting sigma to <r> 
        Usage: 
        
        sigmalist = np.linspace(.01,.9)
        meanr = sigma2meanr(sigmalist)
        plt.figure()
        plt.plot(sigmalist,meanr,'o')
        plt.grid(True)        
    '''
    p = np.array([ 4.57899291e-01, -2.27236062e+00,  4.72080621e+00, -5.09338608e+00,
        2.57626515e+00,  1.77811151e-01, -8.38705493e-01,  1.49765369e-02,
        4.98822342e-01,  3.87112620e-05, -3.41914362e-07])
    meanr = np.polyval(p,sigma)
    return meanr

def R_squar(y,yfit):
    SS_res = np.sum((y-yfit)**2)
    SS_tot = np.sum((y-np.mean(y))**2)
    return 1-SS_res/SS_tot

