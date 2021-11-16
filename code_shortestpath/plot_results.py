import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',**{'family': 'serif', 'serif': ['Times'], 'size': 25})
mpl.rc('legend', **{'fontsize': 17})
mpl.rc('text', usetex=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', nargs='?', const=1, type=str, default="./results/")
parser.add_argument('--task_type', nargs='?', const=1, type=str, default="tsp")
args = parser.parse_args()


def accuracy_plot(arr, fig_path):
    mpl.rc('xtick', **{'labelsize': 18})
    mpl.rc('ytick', **{'labelsize': 18})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)
    spacing = 5
    plt.errorbar(np.arange(len(arr[0][0]))[::spacing], arr[3][0][::spacing], yerr=arr[3][1][::spacing], color='forestgreen', linewidth=3, label=r'$\textsc{Cur}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(arr[0][0]))[::spacing], arr[2][0][::spacing], yerr=arr[2][1][::spacing], color='dodgerblue', ls='-.', linewidth=2.5, label=r'$\textsc{Cur-T}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(arr[0][0]))[::spacing], arr[1][0][::spacing], yerr=arr[1][1][::spacing], color='mediumpurple', ls='--', linewidth=2.5, marker='^', markersize=6, label=r'$\textsc{Cur-L}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(arr[0][0]))[::spacing], arr[0][0][::spacing], yerr=arr[0][1][::spacing], color='orangered', ls=':', linewidth=3.5, label=r'$\textsc{Agn}$', elinewidth=1.0)
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
    result_dir = args.result_dir + args.task_type + "/"
    curriculum_strategies = os.listdir(result_dir)
    curriculum_strategies.sort()
    test_performance = []
    for curr in curriculum_strategies:
        if curr[-3:] == "pdf":
            continue
        data_dir = os.path.join(result_dir, curr)
        test_performance.append(read_test_performance_step(data_dir))

    accuracy_plot(equate_lengths(test_performance), result_dir+"test_reward_graph.pdf")
    return



if __name__=="__main__":
    main()
