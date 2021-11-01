import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',**{'family':'serif','serif':['Times'], 'size': 25})
mpl.rc('legend', **{'fontsize': 17})
mpl.rc('text', usetex=True)

import numpy as np
import os
import argparse


def plot_reward_all(reward_curves, path):
    mpl.rc('xtick', **{'labelsize': 18})
    mpl.rc('ytick', **{'labelsize': 18})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)
    spacing=10
    plt.errorbar(np.arange(len(reward_curves[3][0]))[::spacing], np.mean(reward_curves[3], axis=0)[::spacing], yerr=(np.std(reward_curves[3], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='forestgreen', linewidth=3, label=r'$\textsc{Cur}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[4][0]))[::spacing], np.mean(reward_curves[4], axis=0)[::spacing], yerr=(np.std(reward_curves[4], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='dodgerblue', ls='-.', linewidth=2.5, label=r'$\textsc{Cur-T}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[5][0]))[::spacing], np.mean(reward_curves[5], axis=0)[::spacing], yerr=(np.std(reward_curves[5], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='mediumpurple', ls='--', marker='^', markersize=6, linewidth=2.5, label=r'$\textsc{Cur-L}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[2][0]))[::spacing], np.mean(reward_curves[2], axis=0)[::spacing], yerr=(np.std(reward_curves[2], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='dimgray', ls='-.', marker='v', markersize=6, linewidth=2.5, label=r'$\textsc{Scot}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[0][0]))[::spacing], np.mean(reward_curves[0], axis=0)[::spacing], yerr=(np.std(reward_curves[0], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='orangered', ls=':', linewidth=3.5, label=r'$\textsc{Agn}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[1][0]))[::spacing], np.mean(reward_curves[1], axis=0)[::spacing], yerr=(np.std(reward_curves[1], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='gold', ls='--', marker='s', markersize=6, linewidth=3, label=r'$\textsc{Omn}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[6][0]))[::spacing], np.mean(reward_curves[6], axis=0)[::spacing], yerr=(np.std(reward_curves[6], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='firebrick', ls='--', linewidth=2, label=r'$\textsc{BBox}$', elinewidth=1.0)
    plt.xlabel('Time t')
    plt.ylabel('Expected reward', labelpad=(5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.56), ncol=2, fancybox=False, shadow=False)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return


def plot_curriculum(curriculum_array, path):
    mpl.rc('xtick', **{'labelsize': 20})
    mpl.rc('ytick', **{'labelsize': 20})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)
    ytick_list = []
    tasks = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in tasks:
        ytick_list.append('T'+str(i))
    plt.pcolor(curriculum_array, cmap='Greys')
    plt.xlabel('Time t')
    plt.ylabel('Task picked')
    plt.yticks(list(np.arange(len(tasks)) + 0.5), ytick_list)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_lanes',nargs='?', const=1, type=int, default=4)
    args = parser.parse_args()


    plot_path = "./results/init_lanes={}/".format(args.init_lanes)
    reward_path = plot_path + "reward/"
    curr_path = plot_path + "curriculum/"

    for i,f in enumerate(os.listdir(curr_path)):
        curr_array = np.load(curr_path + f)[2]
        plot_curriculum(curr_array, curr_path + "curriculum_{}.pdf".format(i))

    files = os.listdir(reward_path)
    if "git.keep" in files:
        files.remove("git.keep")

    reward_curve = np.load(reward_path + files[0])[:, np.newaxis, :]
    for arr in files[1:]:
        reward_curve = np.append(reward_curve, np.load(reward_path + arr)[:, np.newaxis, :], axis=1)
    plot_reward_all(reward_curve, plot_path + "expected_reward_graph.pdf")
    return


if __name__=="__main__":
    main()
