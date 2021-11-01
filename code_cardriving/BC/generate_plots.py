import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('legend', **{'fontsize': 16.6})
mpl.rc('text', usetex=True)
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_lanes',nargs='?', const=1, type=int, default=1)
args = parser.parse_args()


def final_plot_all(reward_curves, path):
    mpl.rc('xtick', **{'labelsize': 18})
    mpl.rc('ytick', **{'labelsize': 18})
    mpl.rc('font',**{'family':'serif','serif':['Times'], 'size': 25})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)
    spacing=10
    plt.errorbar(np.arange(len(reward_curves[1][0]))[::spacing], np.mean(reward_curves[1], axis=0)[::spacing], yerr=(np.std(reward_curves[1], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='forestgreen', linewidth=3, label=r'$\textsc{Cur}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[2][0]))[::spacing], np.mean(reward_curves[2], axis=0)[::spacing], yerr=(np.std(reward_curves[2], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='dodgerblue', ls='-.', linewidth=2.5, label=r'$\textsc{Cur-T}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[3][0]))[::spacing], np.mean(reward_curves[3], axis=0)[::spacing], yerr=(np.std(reward_curves[3], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='mediumpurple', marker='^', markersize=6, markevery=2, ls='--', linewidth=2.5, label=r'$\textsc{Cur-L}$', elinewidth=1.0)
    plt.errorbar(np.arange(len(reward_curves[0][0]))[::spacing], np.mean(reward_curves[0], axis=0)[::spacing], yerr=(np.std(reward_curves[0], axis=0)/np.sqrt(len(reward_curves)))[::spacing], color='orangered', ls=':', linewidth=3.5, label=r'$\textsc{Agn}$', elinewidth=1.0)
    plt.ylabel('Expected reward')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.31),ncol=2, fancybox=False, shadow=False)
    plt.xlabel('Time t')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return


def plot_curriculum(curriculum_array, path, tasks):
    mpl.rc('xtick', **{'labelsize': 20})
    mpl.rc('ytick', **{'labelsize': 20})
    mpl.rc('font',**{'family':'serif','serif':['Times'], 'size': 25})
    plt.figure()
    fig_size = (5.5 / 1.3, 3.4 / 0.95)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(*fig_size)
    ytick_list = []
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
    lanes = 8
    tasks = np.arange(8)

    plot_path = "results/init_lanes={}/".format(args.init_lanes)
    learner_names = ["Agn", "Cur", "Cur-T", "Cur-L"]
    reward_path = ["/expected_reward/"]

    for learner in learner_names[1:2]:
        curr_path = plot_path + learner + "/curriculum/"
        i = 0 
        for f in os.listdir(curr_path):
            curr_array = np.load(curr_path + f)
            plot_curriculum(curr_array, curr_path + "curriculum_{}.pdf".format(i), tasks)
            i += 1

    for l, reward_type in enumerate(reward_path):
        reward_curve = list()
        for i, name in enumerate(learner_names):
            file_path = plot_path + name + reward_type
            files = os.listdir(file_path)
            reward_curve.append(np.load(file_path + files[0])[np.newaxis, :])
            for arr in files[1:]:
                reward_curve[i] = np.append(reward_curve[i], np.load(file_path + arr)[np.newaxis, :], axis = 0)

        final_plot_all(reward_curve, plot_path + reward_type[:-1] + "_graph.pdf")

    return



if __name__=="__main__":
    main()
