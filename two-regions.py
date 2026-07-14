import numpy as np
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

def build_region_matching_matrices(klist, dlist, elist):
    """
    Returns the matrices in the equation ML @ AL == MR @ AR
    ML = output[0], MR = output[1]
    """
    k = np.moveaxis(klist, -1, 0)
    d = np.moveaxis(dlist, -1, 0)
    e = np.moveaxis(elist, -1, 0)

    # shapes are now (mode, c, w)

    modes = range(4)

    region_matrices = np.array(
        [[d[mode]           for mode in modes],
         [k[mode] * d[mode] for mode in modes],
         [e[mode]           for mode in modes],
         [k[mode] * e[mode] for mode in modes]]
    )

    return np.moveaxis(region_matrices, 2, 0)

def k_isreal(klist):
    """
    Takes a NDArray of shape (mode, region, w) and returns an array of shape (region,) with bool values saying if k is real. It also asserts that all k are the same type inside the same region.
    """
    k_inf_real = np.all(np.isreal(klist[:, :,  0]), axis=0)
    k_sup_real = np.all(np.isreal(klist[:, :, -1]), axis=0)
    for region in range(len(k_inf_real)):
        if k_inf_real[region] != k_sup_real[region]:
            print(klist[:, region,  0], klist[:, region,  -1])
            raise ValueError(f"k values in region {region} mix complex and real values")
        
    return k_inf_real

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

def build_in_out_matching_matrices(region_matrices, klist):
    """
    Returns a list of matrices in the equations Min @ Ain == Mout @ Aout. Modes are written in "canonical order", defined below.

    ## Canonical Order

    The order of the modes in Ain, Aout vectors should be as follows:
        We write first all propagating modes in this order:
            Right modes first, Left modes last
            In a same region, modes follow the order "v, u, 3, 4"
        Then, we write exponential (decaying) modes in this order
            Right modes first, Left modes last
            In a same region, modes follow the order "+,-"
    """

    IN, OUT = range(2)

    def subsub(cols):
        in_matrix  = [cols[m,r] * (-1)**(r+1) for m,r in subsub_indices()[IN]]
        out_matrix = [cols[m,r] * (-1)**r     for m,r in subsub_indices()[OUT]]
        return in_matrix, out_matrix
    def subsuper(cols):
        in_matrix  = [cols[m,r] * (-1)**(r+1) for m,r in subsuper_indices()[IN]]
        out_matrix = [cols[m,r] * (-1)**r     for m,r in subsuper_indices()[OUT]]
        return in_matrix, out_matrix
    def supersub(cols):
        in_matrix  = [cols[m,r] * (-1)**(r+1) for m,r in supersub_indices()[IN]]
        out_matrix = [cols[m,r] * (-1)**r     for m,r in supersub_indices()[OUT]]
        return in_matrix, out_matrix
    def supersuper(cols):
        in_matrix  = [cols[m,r] * (-1)**(r+1) for m,r in supersuper_indices()[IN]]
        out_matrix = [cols[m,r] * (-1)**r     for m,r in supersuper_indices()[OUT]]
        return in_matrix, out_matrix

    cols = np.transpose(region_matrices, (2, 0, 1, 3))
    klist = np.transpose(klist, (2, 0, 1))

    all_k_isreal = k_isreal(klist)

    k_inf_real = np.all(np.isreal(klist[:, :,  1]), axis=0)
    k_sup_real = np.all(np.isreal(klist[:, :, -2]), axis=0)
    for region in range(len(k_inf_real)):
        if k_inf_real[region] != k_sup_real[region]:
            print(klist[:, region,  1], klist[:, region,  -2])
            raise ValueError(f"k values in region {region} mix complex and real values")
        
    all_k_isreal = k_inf_real
    del k_inf_real, k_sup_real

    if all_k_isreal[L] and all_k_isreal[R]:
        in_matrix, out_matrix = supersuper(cols)
    elif all_k_isreal[R]:
        in_matrix, out_matrix = subsuper(cols)
    elif all_k_isreal[L]:
        in_matrix, out_matrix = supersub(cols)
    else:
        in_matrix, out_matrix = subsub(cols)

    return np.array(in_matrix).T, np.array(out_matrix).T


@function_timer
def main(clist, v:float, mass:float, density:float, 
         wlist, 
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
    
    region_matching_matrices = build_region_matching_matrices(klist, dlist, elist)

    in_matrix, out_matrix = build_in_out_matching_matrices(region_matching_matrices, klist)

    scattering_matrix = np.linalg.inv(out_matrix) @ in_matrix

    if save_path != '':
        BASE_PATH = os.path.dirname(os.path.abspath(__file__))
        SAVE_PATH = os.path.join(BASE_PATH, save_path)
        np.savez(SAVE_PATH,
                 clist = clist,
                 v = v,
                 mass = mass,
                 density = density,
                 wlist = wlist,
                 klist = klist,
                 dlist = dlist,
                 elist = elist,
                 region_matching_matrices = region_matching_matrices,
                 in_matrix = in_matrix,
                 out_matrix = out_matrix,
                 scattering_matrix = scattering_matrix)
        print(f"Saved to {save_path}")


if __name__ == "__main__":

    SAVE_PATH = ''

    cleft, cright, speed = 0.5, 2.0, -1.0
    log_wmax = np.log10(calculate_wmax(cleft, speed))

    wrange = np.logspace(-4, log_wmax, 10**4)

    main(clist=[cleft, cright], 
         v=speed, 
         mass=1, 
         density=1/(4*np.pi), 
         wlist=wrange[:], 
         save_path=SAVE_PATH)




