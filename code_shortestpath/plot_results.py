import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',**{'size': 20})
mpl.rc('legend', **{'fontsize': 18})
mpl.rc('text', usetex=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', nargs='?', const=1, type=str, default="./results/")
args = parser.parse_args()


def accuracy_plot(arr, fig_path):
    mpl.rc('xtick', **{'labelsize': 16})
    mpl.rc('ytick', **{'labelsize': 16})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)
    spacing = 4
    plt.errorbar(np.arange(len(arr[0][0]))[::spacing], arr[3][0][::spacing], yerr=arr[3][1][::spacing], color='forestgreen', ls='-', linewidth=2.8, label="Cur", elinewidth=1.0)
    plt.errorbar(np.arange(len(arr[0][0]))[::spacing], arr[2][0][::spacing], yerr=arr[2][1][::spacing], color='dodgerblue', linewidth=2.5, ls='-.', label="Cur-T", elinewidth=1.0)
    plt.errorbar(np.arange(len(arr[0][0]))[::spacing], arr[1][0][::spacing], yerr=arr[1][1][::spacing], color='mediumpurple', linewidth=2.5, ls='--', label="Cur-L", elinewidth=1.0)
    plt.errorbar(np.arange(len(arr[0][0]))[::spacing], arr[0][0][::spacing], yerr=arr[0][1][::spacing], color='orangered', linewidth=3, ls=':', label="Agn", elinewidth=1.0)
    plt.ylabel("Expected Reward")
    plt.legend()
    plt.xlabel("Time t")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.34),ncol=2, fancybox=False, shadow=False)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return


def read_test_performance_step(direc):
    test = []
    for f in os.listdir(direc):
        if f[0:1] != 'r':
            continue
        arr = np.load(os.path.join(direc, f))
        test.append(arr["test_performance_step"])
    test = np.vstack(test)
    return [np.mean(test, axis=0), np.std(test, axis=0)/np.sqrt(len(test))]


def equate_lengths(arr_list):
    min_len = min(len(arr[0]) for arr in arr_list)
    arr_list = [[arr[0][10:min(min_len,111)], arr[1][10:min(min_len,111)]] for arr in arr_list]
    return arr_list


def main():
    curriculum_strategies = os.listdir(args.result_dir)
    curriculum_strategies.sort()
    test_performance = []
    for curr in curriculum_strategies:
        if curr[-3:] == "pdf":
            continue
        data_dir = os.path.join(args.result_dir, curr)
        test_performance.append(read_test_performance_step(data_dir))

    accuracy_plot(equate_lengths(test_performance), args.result_dir+"test_reward_graph.pdf")
    return



if __name__=="__main__":
    main()
