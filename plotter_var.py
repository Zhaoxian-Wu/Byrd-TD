import pickle
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load_file_in_cache(file_name):
    file_path = os.path.join('record', file_name)
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def plot_var(args):
    
    plt.style.use('seaborn')
    plt.rc('axes', labelsize=20, titlesize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=19)
    # linestyles = ('-','--','-','--')
    linestyles = ('-','--','-.')
    # colors = ('royalblue', 'darkorange', 'darkgreen', 'darkmagenta')
    cmap = plt.cm.coolwarm
    # colors = cmap(np.linspace(0, 1, len(args.lams)))
    # colors = cmap([0, 0.3, 1])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # fig = plt.figure(figsize=(7.,7.5))
    fig = plt.figure(figsize=(8.,5.5))
    fig.subplots_adjust(left=0.05, right=0.98,top=0.95, bottom=0.16, hspace=0.3,wspace=0.31)
    
    vars_2 = [v**2 for v in args.vars]
    for i in range(len(args.lams)):
        asym_err_trim = []
        asym_err_trim_att = []
        asym_err_local = []
        file_name = 'sbe' + '_' + args.network \
            + '_a' + str(args.attack) \
            + '_lam' + str(args.lams[i]) \
            + '_vars' + '.pkl'
        for index_var, variation in enumerate(args.vars):
            asbe = np.array(load_file_in_cache(file_name))
            avg_asym_sbe = np.mean(asbe[:,-1,:], 0)
            # the 1th method is trimmed mean
            asym_err_trim.append(avg_asym_sbe[5*index_var+1])
            # the 3th method is trimmed mean under attack
            asym_err_trim_att.append(avg_asym_sbe[5*index_var+3])
            # the 4th method is local
            asym_err_local.append(avg_asym_sbe[5*index_var+4])
        plt.plot(vars_2, asym_err_trim, linestyles[0], color=colors[i],
                 linewidth=2.25, label=f'trim: $\lambda={args.lams[i]}$')
        plt.plot(vars_2, asym_err_trim_att, linestyles[1], color=colors[i],
                 linewidth=2.25, label=f'trim-att: $\lambda={args.lams[i]}$')
        plt.plot(vars_2, asym_err_local, linestyles[2], color=colors[i],
                 linewidth=2.25, label=f'local: $\lambda={args.lams[i]}$')
    
    plt.ylabel('Asymtotic MSBE')
    plt.xlabel('$\delta^2_0$')
    plt.ylim(top=0.2, bottom=0)
    plt.xlim(right=1.5)
    custom_lines = [
        Line2D([0], [0], color=colors[0], linestyle=linestyles[0], lw=2.25),
        Line2D([0], [0], color=colors[0], linestyle=linestyles[1], lw=2.25),
        Line2D([0], [0], color=colors[0], linestyle=linestyles[2], lw=2.25)
    ] \
    + [Line2D([0], [0], color=colors[i], linestyle=linestyles[0], lw=8) for i in range(len(args.lams))]
    custom_title = [
        'trim', 'trim-att', 'local'
    ] + [f'$\lambda={lam}$' for lam in args.lams]
    
    plt.legend(custom_lines, custom_title, ncol=2)
    # fig.tight_layout()
    figure_path = os.path.join('figure', 'var_'+args.network+'_a'+str(args.attack)+'.pdf')
    fig.savefig(figure_path, dpi=fig.dpi, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plotter for robust TD')
    
    parser.add_argument('--network', type=str, default='complete',
    help='name of the network (h1b1, h3b1, b3b2, renyi, complete)')
    parser.add_argument('--attack', type=int, default=2,
    help='Type of attack')
    parser.add_argument('--lams', type=float, nargs='+', 
                        help='lambda list', default=[0.0,0.3,0.6,0.9])
    parser.add_argument('--vars', type=float, nargs='+', 
                        help='lambda list', default=[0.0,0.5,1.0,1.5])
    parser.add_argument('--lnk', action='store_true',
    help='whether to divide aggregate ce by ln(epoch) or epoch')
    args = parser.parse_args()
    
    plot_var(args)