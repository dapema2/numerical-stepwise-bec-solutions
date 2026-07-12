import numpy as np
from timing import function_timer
import math
import os


def calculate_wmax(c:float, v:float) -> float:
    """
    Returns the wmax value for a supersonic region
    """
    if c < abs(v):
        kmax = -c * math.sqrt(-2 + v**2 / (2 * c**2) + abs(v) * math.sqrt(8 + v**2 / c**2) /(2 * c))
        wmax = v * kmax - c * math.sqrt(kmax**2 + kmax**4 / (4 * c**2))
        return wmax
    else:
        raise ValueError("wmax not defined by a non-supersonic region")
    

def create_wrange(wmin:float, wmax:float, wnum:int, spacing = 'log'):
    if spacing == 'lin':
        return np.linspace(wmin, wmax, wnum, endpoint=True)
    elif spacing == 'log':
        return np.logspace(np.log10(wmin), np.log10(wmax), wnum)
    else:
        raise ValueError("Variable 'spacing' needs to be either string 'lin' or string 'log'")


def dispersion_relation(clist, v:float, mass:float, wlist):
    x4 = np.array([[1/(4*mass**2) for w in wlist] for c in clist])
    x3 = np.array([[0 for w in wlist] for c in clist])
    x2 = np.array([[c**2 - v**2 for w in wlist] for c in clist])
    x1 = np.array([[2 * v * w for w in wlist] for c in clist])
    x0 = np.array([[-w**2 for w in wlist] for c in clist])

    return np.stack([x4, x3, x2, x1, x0], axis=-1)


def sort_roots(roots):
    def sort_function(arr):
        real_mask = np.isreal(arr)
        complex_mask = np.logical_not(real_mask)

        if np.all(real_mask):
            ascending_order = np.sort(arr)
            return ascending_order[[2,1,3,0]]
        else:
            real_vals = arr[real_mask]
            complex_vals = arr[complex_mask]

            real_sorted = real_vals[np.argsort(real_vals.real)]
            complex_sorted = complex_vals[np.argsort(-complex_vals.imag)]

            return np.concatenate([real_sorted, complex_sorted])
        
    return np.apply_along_axis(sort_function, -1, roots)


def create_klist(clist, v:float, mass:float, wlist):

    coeffs = dispersion_relation(clist, v, mass, wlist)

    roots = np.array([[np.roots(poly) for poly in c_group] for c_group in coeffs])

    return sort_roots(roots)


def k_derivative(k, c, v, m, w):
    return (w - v * k) / (v * w + k * (c**2 - v**2) + (k**3 / (2 * m **2)))


def create_delist(klist, clist, v:float, m:float, n:float, wlist):
    c = clist[:,None,None]
    w = wlist[None,:,None]
    k = klist

    dk = k_derivative(k, c, v, m, w)

    prefactor = np.sqrt(m / (4 * np.pi * n))

    denominator = np.sqrt( np.abs(w * (k.real**2 - k.imag**2) - v * (k.real**2 + k.imag**2) * k.real) )
    
    dlist =   prefactor * np.sqrt(np.abs(dk)) * (w - v*k + k**2/(2*m)) / denominator
    elist = - prefactor * np.sqrt(np.abs(dk)) * (w - v*k - k**2/(2*m)) / denominator

    return dlist, elist

def k_isreal(klist):
    """
    Takes a NDArray of shape (mode, region, w) and returns an array of shape (region,) with bool values saying if k is real. It also asserts that all k are the same type inside the same region.
    """
    k_inf_real = np.all(np.isreal(klist[:, :,  0]), axis=0)
    k_sup_real = np.all(np.isreal(klist[:, :, -1]), axis=0)
    for region in range(len(k_inf_real)):
        if k_inf_real[region] != k_sup_real[region]:
            # print(klist[:, region,  0], klist[:, region,  -1])
            raise ValueError(f"k values in region {region} mix complex and real values")
        
    return k_inf_real

def model_teller(klist):
    all_k_isreal = k_isreal(klist)

    k_inf_real = np.all(np.isreal(klist[:, :,  1]), axis=0)
    k_sup_real = np.all(np.isreal(klist[:, :, -2]), axis=0)
    for region in range(len(k_inf_real)):
        if k_inf_real[region] != k_sup_real[region]:
            # print(klist[:, region,  1], klist[:, region,  -2])
            raise ValueError(f"k values in region {region} mix complex and real values")
        
    all_k_isreal = k_inf_real
    del k_inf_real, k_sup_real

    if all_k_isreal[L] and all_k_isreal[R]:
        return 'supersuper'
    elif all_k_isreal[R]:
        return 'subsuper'
    elif all_k_isreal[L]:
        return 'supersub'
    else:
        return 'subsub'

V, U, P, M = 0, 1, 2, 3
L, R = 0, -1

def subsub_indices() -> None:
    in_indices  = [(V,R), (U,L)]
    out_indices = [(U,R), (V,L), (P,R), (M,L)]
    return in_indices, out_indices
def supersub_indices() -> None:
    in_indices  = [(V,R), (P,L), (M,L)]
    out_indices = [(U,R), (V,L), (U,L), (P,R)]
    return in_indices, out_indices
def subsuper_indices() -> None:
    in_indices  = [(V,R), (U,R), (U,L)]
    out_indices = [(P,R), (M,R), (V,L), (M,L)]
    return in_indices, out_indices
def supersuper_indices() -> None:
    in_indices  = [(V,R), (U,R), (P,L), (M,L)]
    out_indices = [(P,R), (M,R), (V,L), (U,L)]
    return in_indices, out_indices

def inout_indexes(model):
    if model == 'subsub':
        return subsub_indices()
    elif model == 'supersub':
        return supersub_indices()
    elif model == 'subsuper':
        return subsuper_indices()
    elif model == 'supersuper':
        return supersuper_indices()
    else:
        raise ValueError(f"{model} is not a valid value for variable model")

# def build_region_matching_matrices(klist, dlist, elist, disc_coords) -> :
#     """
#     Returns the matrices in the equation ML @ AL == MCL @ AC, MCR @ AC == MR @ AR
#     ML = output[0], MC = output[1], MR = output[2]
#     """
#     k = np.moveaxis(klist, -1, 0)
#     d = np.moveaxis(dlist, -1, 0)
#     e = np.moveaxis(elist, -1, 0)

#     # shapes are now (mode, c, w)

#     modes = range(4)

#     output = []

#     for i, a in enumerate(disc_coords):
#         j = i + 1
#         auxL = [[d[mode, i] * np.exp(1j * k[mode, i] * a)              for mode in modes],
#                 [k[mode, i] * d[mode, i] * np.exp(1j * k[mode, i] * a) for mode in modes],
#                 [e[mode, i] * np.exp(1j * k[mode, i] * a)              for mode in modes],
#                 [k[mode, i] * e[mode, i] * np.exp(1j * k[mode, i] * a) for mode in modes]]
#         auxR = [[d[mode, j] * np.exp(1j * k[mode, j] * a)              for mode in modes],
#                 [k[mode, j] * d[mode, j] * np.exp(1j * k[mode, j] * a) for mode in modes],
#                 [e[mode, j] * np.exp(1j * k[mode, j] * a)              for mode in modes],
#                 [k[mode, j] * e[mode, j] * np.exp(1j * k[mode, j] * a) for mode in modes]]
#         output.append([auxL, auxR])

#     return np.moveaxis(np.array(output), -1, -3) # shape (disc, LR, w, row, col)

def create_delist_disc(klist, dlist, elist, disc):
    '''
    Returns D*exp(1j*k*a) and E*exp(1j*k*a) in an array of shape (discontinuity, region, w, mode)

    If in a discontinuity a region is not there, the array is valued 0 there
    Example: we're working in a 3 region model (regions L, C, R) with two discontinuities (x1, x2). At discontinuity x1, regions L and C meet. if we check the array in discontinuity x1 region R (arr[0,2,:,:]), we get 0
    '''
    delist = np.array([dlist, elist])
    disc = np.array(disc)
    i, j = 0, 1
    ans = []
    for a in disc:
        exps = np.exp(1j * klist * a)
        delist_a = delist * exps[None]
        for k in range(delist_a.shape[1]):
            if k != i and k != j:
                delist_a[:,k] = np.zeros((np.shape(disc)[0], np.shape(delist)[2], np.shape(delist)[3]))
        ans.append(delist_a)
        i += 1
        j += 1

    ans = np.array(ans)

    return np.array(ans[:,0]), np.array(ans[:,1])

def create_matching_matrix_dim8(dlist, elist, klist, model):
    """
    Inputs:
        dlist, elist : shapes (disc, region, w, mode). Contains D*exp(1j*k*a) for each region and discontinuity
        klist : shape (region, w, mode). Contains the k's.
        model : 
    """

    dlist = np.moveaxis(dlist, -1, -3)
    elist = np.moveaxis(elist, -1, -3)
    klist = np.moveaxis(klist, -1, -3)

    aux = np.moveaxis(np.array([dlist, dlist * klist[None], elist, elist * klist[None]]), 0, 1) # shape (disc, matrix_element, mode, region, w)

    cols = np.moveaxis(np.concatenate((aux[0], aux[1]), axis=0), 0, -1) # shape (mode, region, w, matrix_element)

    inout_ixs = inout_indexes(model)
    print(inout_ixs)
    matching_matrix = []
    for ixs in inout_ixs[1]:
        matching_matrix.append(-cols[ixs])
        print(cols[ixs].shape)
    for central_region in range(1, np.shape(cols)[1] - 1):
        for mode in range(np.shape(cols)[0]):
            matching_matrix.append(cols[mode, central_region])
    matching_matrix = np.moveaxis(np.array(matching_matrix), 0, 1)
    matching_matrix = np.transpose(matching_matrix, (0, 2, 1))

    indep_matrix = []
    for ixs in inout_ixs[0]:
        indep_matrix.append(cols[ixs])
    indep_matrix = np.moveaxis(np.array(indep_matrix), 0, 1)
    indep_matrix = np.transpose(indep_matrix, (0, 2, 1))

    return matching_matrix, indep_matrix

@function_timer
def main(clist, v:float, mass:float, density:float, 
         wlist, 
         discontinuities_coordinates,
         save_path : str ="") -> None:
    """
    Calculates the scattering matrix of a two region stepwise BEC
    # Parameters
    --------\n
    **`cL` : float**\n
        Sound speed of the condensate at the left region \n
    **`cR` : float**\n
        Sound speed of the condensate at the right region\n
    **`v` : float**\n
        Velocity of the condensate flow\n
    **`mass` : float**\n
        Mass of the atoms that form the condensate\n
    **`density` : float**\n
        Density of the atoms in the condensate (should be constant in all regions)\n
    **`wlist`**\n
        List of (real) frequencies for solving the model
    **`discontinuities_coordinates`**\n
        List of (two) values of space where the discontinuities are located.\n
    **`save_path` : str**\n
        Name of the file where the data should be saved. Leave empty to not save.\n
    
    """

    clist = np.array(clist)

    klist = create_klist(clist, v, mass, wlist)
    dlist, elist = create_delist(klist, clist, v, mass, density, wlist) # shapes (c, w, mode)
    model = model_teller(np.transpose(klist, (2, 0, 1))) # takes in (mode, region, w)
    
    dlista, elista = create_delist_disc(klist, dlist, elist, discontinuities_coordinates)

    matching_matrix_dim8, indep_matrix_dim8 = create_matching_matrix_dim8(dlista, elista, klist, model)

    scattering_matrix_dim8 = np.linalg.inv(matching_matrix_dim8) @ indep_matrix_dim8

    if save_path != '':
        BASE_PATH = os.path.dirname(os.path.abspath(__file__))
        SAVE_PATH = os.path.join(BASE_PATH, save_path)
        np.savez(SAVE_PATH,
                 clist = clist,
                 v = v,
                 mass = mass,
                 density = density,
                 discontinuities = discontinuities_coordinates,
                 wlist = wlist,
                 klist = klist,
                 dlist = dlist,
                 elist = elist,
                 scattering_matrix = scattering_matrix_dim8)
        print(f"Saved to {save_path}")


if __name__ == "__main__":

    SAVE_PATH = ''

    cleft, cright, speed = 0.5, 2.0, -1.0
    wrange = np.logspace(-4, np.log10(calculate_wmax(cleft, speed)), 10**4)
    a = 5

    main(clist=[cleft, -speed, cright], 
         v=speed, 
         mass=1, 
         density=1/(4*np.pi), 
         discontinuities_coordinates = [0,a],
         wlist=wrange[:], 
         save_path=SAVE_PATH)




