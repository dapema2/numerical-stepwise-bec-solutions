import numpy as np
import os
import tworeg
from tqdm import tqdm

def load_data(path:str, xlist) :
    data = np.load(path)

    wlist = data['wlist']
    klist = np.moveaxis(data['klist'], 1, 2) # shape (reg, k, w)
    dlist = np.moveaxis(data['dlist'], 1, 2) # shape (reg, k, w)
    elist = np.moveaxis(data['elist'], 1, 2) # shape (reg, k, w)
    sMatrix = data['scattering_matrix']

    print(wlist.shape, klist.shape, dlist.shape, elist.shape, sMatrix.shape)

    return wlist, klist, dlist, elist, sMatrix

def kde_in_out(xlist, klist, dlist, elist) :
    
    V, U, P, M = 0, 1, 2, 3
    L, R = 0, -1
    
    klist = np.transpose(klist, (1, 0, 2))

    thetaL = np.heaviside(-xlist, 0)
    thetaR = np.heaviside( xlist, 1) 
    theta = np.array([thetaL, thetaR]) # shape (reg, x)
    del thetaL, thetaR

    dpe = dlist + elist
    print(dpe.shape)
    print(theta.shape)
    dpe = np.einsum('rkw, rx -> krwx', dpe, theta)
    del dlist, elist

    k_isreal = tworeg.k_isreal(klist)

    if k_isreal[L] and k_isreal[R]:
        indices = tworeg.supersuper_indices()
    elif k_isreal[L]:
        indices = tworeg.supersub_indices()
    elif k_isreal[R]:
        indices = tworeg.subsuper_indices()
    else:
        indices = tworeg.subsub_indices()

    IN, OUT = 0, -1
        
    kIn   = np.array([klist[r,c] for r,c in indices[IN]]).T # (w, k)
    dpeIn = np.array([  dpe[r,c] for r,c in indices[IN]])
    dpeIn = np.moveaxis(dpeIn, 1, 0) # (w, k, x)

    kOut   = np.array([klist[r,c] for r,c in indices[OUT]]).T # (w, k)
    dpeOut = np.array([  dpe[r,c] for r,c in indices[OUT]])
    dpeOut = np.moveaxis(dpeOut, 1, 0) # (w, k, x)

    return kIn, kOut, dpeIn, dpeOut

def kde_in_out_subsub(xlist, klist, dlist, elist) :
    
    V = 0
    U = 1
    P = 2
    M = 3

    L = 0
    R = 1

    thetaL = np.heaviside(-xlist, 0)
    thetaR = np.heaviside( xlist, 1) 
    theta = np.array([thetaL, thetaR]) # shape (reg, x)
    del thetaL, thetaR

    dpe = dlist + elist
    dpe = np.einsum('rkw, rx -> rkwx', dpe, theta)
    del dlist, elist

    kIn   = np.array([klist[R,V], klist[L,U]]).T # (w, k)
    dpeIn = np.array([  dpe[R,V],   dpe[L,U]])
    dpeIn = np.moveaxis(dpeIn, 1, 0) # (w, k, x)

    kOut   = np.array([klist[R,U], klist[L,V], klist[R,P], klist[L,M]]).T # (w, k)
    dpeOut = np.array([  dpe[R,U],   dpe[L,V],   dpe[R,P],   dpe[L,M]])
    dpeOut = np.moveaxis(dpeOut, 1, 0) # (w, k, x)

    return kIn, kOut, dpeIn, dpeOut

def iterated_integral(xlist, dwlist, kIn, kOut, dpeIn, dpeOut, sMatrix):
    
    integral = 0

    for iw, dw in enumerate(tqdm(dwlist, desc="Progress", unit=" omega", mininterval=1)):

        # CALCULATE (u_phi + u_varphi)

        exponential = np.exp(1j * np.einsum('o,x -> ox', kOut[iw], xlist))
        u_inmode = np.einsum('oi,ox,ox -> xi', sMatrix[iw], dpeOut[iw], exponential)
        u_inmode += np.einsum('ix,ix -> xi', dpeIn[iw], np.exp(1j * np.outer(kIn[iw], xlist)))

        # MULTIPLY BY c.c(x') AND SUM OVER THE IN MODES

        integral += np.real(np.einsum('xi, yi -> ixy', u_inmode, np.conjugate(u_inmode))) * dw

    return integral

def main(xlist, load_path, save_path:str = "", include_zero=False) -> None:
    """
     Calculates the two point correlation function for a three region stepwise BEC model
    # Parameters
    --------\n
    **`xlist` : float**\n
        Sound speed of the condensate at the left region \n
    **`load_path` : str**\n
        Path where the results calculated with `three-regions.py` are located\n
    **`save_path` : str**\n
        Name of the file where the data should be saved. Leave empty to not save.\n
    **'include_zero` : bool**\n
        True/False to include the zero as the lower bound of the integral.
    """
    
    wlist, klist, dlist, elist, sMatrix = load_data(load_path, xlist)

    kIn, kOut, dpeIn, dpeOut = kde_in_out(xlist, klist, dlist, elist)
    del klist, dlist, elist

    if include_zero:
        dwlist  = np.diff(wlist, prepend=0)
    else:
        dwlist  = np.diff(wlist)

    integral = iterated_integral(xlist, dwlist, kIn, kOut, dpeIn, dpeOut, sMatrix)
    
    if save_path != "":
        np.savez(save_path, correlator = integral, xlist = xlist)
        print(f"Saved to {save_path}")



if __name__ == "__main__":
    xlist     = np.linspace(-100, 100, num=1001, endpoint=True)
    LOAD_PATH = ''
    SAVE_PATH = ''

    print(xlist)

    main(xlist=xlist, load_path=LOAD_PATH, save_path=SAVE_PATH, include_zero=True)
