import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

markers = [
    'o', 'p', 'v', 's', '*'
]
SCALE = 4.5
# SCALE = 2.5
FONT_SIZE = 24
LINE_STYLE = '-'
MARKER_SIZE = 10
TICK_SIZE = 16

def subplotall(args):   
    
    def define_style(rcParams):
        rcParams['figure.figsize'] = SCALE*2, SCALE*1.6
        rcParams['figure.subplot.wspace'] = 0.2
        rcParams['figure.subplot.hspace'] = 0.3
        rcParams['lines.markersize'] = MARKER_SIZE
        rcParams['xtick.labelsize'] = TICK_SIZE
        rcParams['ytick.labelsize'] = TICK_SIZE
        rcParams['axes.labelsize'] = FONT_SIZE
        rcParams['axes.titlesize'] = FONT_SIZE
        rcParams['legend.fontsize'] = FONT_SIZE
        rcParams['font.sans-serif'] = 'Microsoft YaHei'
        rcParams['axes.grid'] = True
        
    define_style(rcParams)
    
    # plt.style.use('seaborn')
    # plt.rc('axes', labelsize=20, titlesize=20)
    # plt.rc('xtick', labelsize=15)
    # plt.rc('ytick', labelsize=15)
    # plt.rc('legend', fontsize=19)
    
    linestyles = ('d--', 'o--', 'd-', 'o-', '*--')
    fig, ax_matrix = plt.subplots(2, 2)
    ax_array = [ax_matrix[0][0], ax_matrix[0][1], ax_matrix[1][0], ax_matrix[1][1]]
    # fig.subplots_adjust(left=0.05, right=0.98,top=0.95, bottom=0.16, hspace=0.3,wspace=0.31)
    for i, axes in zip(range(len(args.lams)), ax_array):
        sbe_name = 'sbe'+'_'+args.network+'_a'+str(args.attack)+'_lam'+str(args.lams[i])+'.pkl'
        ace_name = 'ce'+'_'+args.network+'_a'+str(args.attack)+'_lam'+str(args.lams[i])+'.pkl'
        
        file_dir = os.path.dirname(os.path.abspath(__file__))
        dir_png_path = os.path.join(file_dir, 'record')
        sbe_path = os.path.join(dir_png_path, sbe_name)
        ace_path = os.path.join(dir_png_path, ace_name)
        
        f = open(sbe_path, 'rb')
        asbe = np.array(pickle.load(f))
        f.close()
        f = open(ace_path, 'rb')
        ace = np.array(pickle.load(f))
        f.close()
        
        # fig.add_subplot(2, len(args.lams), i+1, xlabel='Step', ylabel='MSBE', title=r'$\lambda=$%.1g' %args.lams[i])
        for k in range(np.shape(asbe)[-1]):
            y = asbe[:,:,k]
            y_mean = np.mean(y, 0)
            x = np.arange(len(y_mean))
            axes.set_title(r'$\lambda=$%.1g' %args.lams[i])
            axes.plot(x, y_mean, linestyles[k], linewidth=2.25, markevery=30)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        
    ax_matrix[1][0].set_xlabel('迭代次数')
    ax_matrix[1][1].set_xlabel('迭代次数')
    ax_matrix[0][0].set_ylabel('估计误差')
    ax_matrix[1][0].set_ylabel('估计误差')
    # ax_matrix[1][0].set_ylabel('一致误差')
    fig.legend(
        (r'TD($\lambda$)（无攻击）', 
         r'Byrd-TD($\lambda$)（无攻击）', 
         r'TD($\lambda$)（有攻击）', 
         r'Byrd-TD($\lambda$)（有攻击）',
         r'无通信'),
        loc='lower left', bbox_to_anchor=(0.95, 0.4), fancybox=False, shadow=False, 
        ncol=1, borderaxespad=0., frameon=True)
    # fig.tight_layout()
    
    pic_name = 'all_'+args.network+'_a'+str(args.attack)+'_slides'
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    dir_png_path = os.path.join(file_dir, 'thesis-figure', 'png')
    dir_pdf_path = os.path.join(file_dir, 'thesis-figure', 'pdf')

    if not os.path.isdir(dir_pdf_path):
        os.makedirs(dir_pdf_path)
    if not os.path.isdir(dir_png_path):
        os.makedirs(dir_png_path)
    pic_png_path = os.path.join(dir_png_path, pic_name + '.png')
    pic_pdf_path = os.path.join(dir_pdf_path, pic_name + '.pdf')
    plt.savefig(pic_png_path, format='png', bbox_inches='tight')
    plt.savefig(pic_pdf_path, format='pdf', bbox_inches='tight')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plotter for robust TD')
    
    parser.add_argument('--network', type=str, default='complete',
    help='name of the network (h1b1, h3b1, b3b2, renyi, complete)')
    parser.add_argument('--attack', type=int, default=2,
    help='Type of attack')
    parser.add_argument('--lams', type=float, nargs='+', 
                        help='lambda list', default=[0.0,0.3,0.6,0.9])
    parser.add_argument('--lnk', action='store_true',
    help='whether to divide aggregate ce by ln(epoch) or epoch')
    args = parser.parse_args()
    
    subplotall(args)