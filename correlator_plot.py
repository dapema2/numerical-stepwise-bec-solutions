import matplotlib.pyplot as plt
import numpy as np
import os



def main(data_paths, vmax, vmin, save_path, title="") -> None:

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path"
    })

    N_PATHS = len(data_paths)

    correlators_list = []

    first = True
    for path in data_paths:
        with np.load(path) as f:
            if first:
                xlist = f['xlist']
                first = False
            elif np.all(xlist != f['xlist']):
                raise ValueError("xlist does not coincide in paths provided")
            correlators_list.append(f['correlator'])
    
    cL, cR = 0.5, 2.0
    v = -1.0
    mlist = [(v+cL)/(v+cR), (v-cL)/(v+cR), (v-cL)/(v+cL)]
    colors = ['red', 'blue', 'green']

    correlator_list_allmodes = np.array([np.sum(correlator, axis=0) for correlator in correlators_list])

    total_correlator = np.sum(correlator_list_allmodes, axis=0)

    if vmax is None and vmin is None:
        plt.imshow(total_correlator)
    elif vmax is None:
        plt.imshow(total_correlator, vmin=vmin)
    elif vmin is None:
        plt.imshow(total_correlator, vmax=vmax)
    else:
        plt.imshow(total_correlator, vmax=vmax, vmin=vmin)

    plt.gca().invert_yaxis()

    NUMBER_OF_TICKS = 5
    X_SIZE = len(xlist)
    tick_indexs = [int(((X_SIZE - 1) / (NUMBER_OF_TICKS - 1)) * n) for n in range(NUMBER_OF_TICKS)]
    tick_labels = [str(int(xlist[i])) for i in tick_indexs]
    plt.xticks(tick_indexs, tick_labels)
    plt.yticks(tick_indexs, tick_labels)
        
    plt.xlabel(r"$x$")
    plt.ylabel(r"$x^\prime$")

    plt.title(title)
    
    color_bar = plt.colorbar()
    color_bar.set_label(r"$G^{(2)}(x,x^\prime)$")

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(BASE_PATH, save_path)
    plt.savefig(SAVE_PATH)
    plt.clf()

    print(xlist.shape)

if __name__ == "__main__":
    val = 0.25
    vmax =  val
    vmin = -val
    LOAD_PATH_1 = ''
    LOAD_PATH_2 = ''
    SAVE_PATH = ''
    title=r"$G^{(2)}(x,x^{\prime})$"
    main([LOAD_PATH_1, LOAD_PATH_2], vmax, vmin, SAVE_PATH, title)
