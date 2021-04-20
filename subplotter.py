def subplotall(args):   
    import pickle
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn')
    plt.rc('axes', labelsize=20, titlesize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=19)
    
    linestyles = ('-','--','-','--')
    fig = plt.figure(figsize=(18.,7.5))
    fig.subplots_adjust(left=0.05, right=0.98,top=0.95, bottom=0.16, hspace=0.3,wspace=0.31)
    for i in range(len(args.lams)):
        
        f = open('sbe'+'_'+args.network+'_a'+str(args.attack)+'_lam'+str(args.lams[i])+'.pkl', 'rb')
        asbe = np.array(pickle.load(f))
        f.close()
        f = open('ce'+'_'+args.network+'_a'+str(args.attack)+'_lam'+str(args.lams[i])+'.pkl', 'rb')
        ace = np.array(pickle.load(f))
        f.close()
        
        fig.add_subplot(2, len(args.lams), i+1, xlabel='Step', ylabel='MSBE', title=r'$\lambda=$%.1g' %args.lams[i])
        for k in range(np.shape(asbe)[-1]):
            y = asbe[:,:,k]
            y_mean = np.mean(y, 0)
            x = np.arange(len(y_mean))
            plt.plot(x, y_mean, linestyles[k], linewidth=2.25)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        
        if args.lnk:
            fig.add_subplot(2, len(args.lams), i+1+len(args.lams), xlabel='Step', ylabel=r'MCE$\times k/\ln(k)$')
        else:
            fig.add_subplot(2, len(args.lams), i+1+len(args.lams), xlabel='Step', ylabel='MCE')
        for k in range(np.shape(ace)[-1]):
            y = ace[:,:,k] 
            y_mean = np.mean(y, 0)
            try:
                if args.lnk:
                    for j in np.arange(1,len(y_mean)):
                        y_mean[j] = y_mean[j]*(j+1)/np.log(j+1)
            except:
                pass
            x = np.arange(len(y_mean))
            plt.plot(x, y_mean, linestyles[k], linewidth=2.5)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))


    fig.legend((r'TD($\lambda$)-mean', r'TD($\lambda$)-trim', r'TD($\lambda$)-mean-attack', r'TD($\lambda$)-trim-attack'),
               loc='upper center', bbox_to_anchor=(0.5, 0.07), fancybox=False, shadow=False, ncol=4, borderaxespad=0., frameon=True)
    # fig.tight_layout()
    fig.savefig('all_'+args.network+'_a'+str(args.attack)+'.pdf', dpi=fig.dpi, bbox_inches="tight")
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
    parser.add_argument('--lnk', action='store_true',
    help='whether to divide aggregate ce by ln(epoch) or epoch')
    args = parser.parse_args()
    
    subplotall(args)