import matplotlib.pyplot as plt
import numpy as np
import os



def main(data_paths, vmax, vmin, save_path, title="") -> None:

    # --- Global LaTeX Font Configuration ---
    # This forces Matplotlib to use LaTeX to render all text.
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

    # corr_1 = correlators_list[0]
    # corr_2 = correlators_list[1]
    
    # correlator_1 = (corr_1[0] + corr_1[1] + corr_1[2])
    # correlator_2 = (corr_2[0] + corr_2[1])

    # total_correlator = correlator_1 + correlator_2

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

    # plt.plot([0,1000], [500, 500], color='red')
    # plt.plot([500,500], [0,1000], color='red')

    # for im, m in enumerate(mlist):
    #     plt.axline([500,500], slope=m, color=colors[im])

    # origin = [500, 500]
    # colors  = ['red', 'blue', 'white']
    # plt.plot([origin[0],  999], [origin[1], 250], color=colors[0])
    # plt.plot([origin[0],  833], [origin[1],   0], color=colors[1])
    # plt.plot([origin[0],  333], [origin[1],   0], color=colors[2])
    # # plt.axline([750,375], slope=-1/mlist[0], color='black')

    # plt.plot([origin[0],  250], [origin[1], 999], color=colors[0])
    # plt.plot([origin[0],    0], [origin[1], 833], color=colors[1])
    # plt.plot([origin[0],    0], [origin[1], 333], color=colors[2])

    # letters = [r'$a$', r'$b$', r'$c$']
    # x_positions = [900, 700, 325]  # replace with your actual values
    # y_positions = [250, 100, 150]  # replace with your actual values

    # for letter, color, x, y in zip(letters, colors, x_positions, y_positions):
    #     plt.text(x, y, letter, color=color, fontsize=14, fontweight='bold',
    #             ha='center', va='center')

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
