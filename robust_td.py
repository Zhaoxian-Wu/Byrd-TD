import pickle
import plotter
import argparse
import numpy as np
from scipy import stats
from make_env import make_env
import RenyiGraph as renyigraph
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import cdist, euclidean
        
# def action_sampler(policy, state):
#     logits = policy @ state
#     logits = np.reshape(logits, (1,len(logits)))
#     return np.squeeze(tf.random.categorical(logits, 1), -1)  

class decent_network:
    
    def __init__(self, env, w, trainers):
        self.env = env
        self.trainers = trainers
        self.policy_n = [np.random.normal(0, 0.1, (env.action_space[0].n, env.observation_space[0].shape[0])) for i in range(env.n)]
        self.w = np.array(w)

    def train(self, max_epoch=300, itr_num=32, lr=1e-1, lam=0, diminish=False, render=False, reset_interval=1):
        lr0 = lr
        env = self.env
        gam = self.trainers[0].gam
        asbe, epoch_sbe, ace, epoch_ce = [], [], [], []
        next_obs_n = np.array(env.reset())
        z = np.zeros(env.observation_space[0].shape[0])
        for epoch in range(max_epoch):
            sbe_ce = []
            if (epoch+1) % reset_interval == 0:
                z = np.zeros(env.observation_space[0].shape[0])
                next_obs_n = np.array(env.reset())
            for itr in range(itr_num):
                # sampling
                obs_n = next_obs_n.copy()
                obs_n = np.array([obs_n[0]] * env.n)
#                actions_n = np.array([action_sampler(self.policy_n[i], obs_n[i]) for i in range(env.n)])
                actions_n = np.random.randint(env.action_space[0].n, size=(env.n,1))
                onehot_n = to_categorical(actions_n, env.action_space[0].n)
                next_obs_n, rew_n, _, _ = [np.array(ret) for ret in env.step(onehot_n)]
                next_obs_n = np.array([next_obs_n[0]] * env.n)
                if render:
                    env.render(mdoe=None)
                
                z = gam * lam * z + obs_n[0]
                pair = (obs_n, next_obs_n, rew_n, z)
                self.w = (self.w.T/np.sum(self.w, 1)).T
                if diminish:
                    lr = lr0/np.sqrt((epoch + 1))
                sbe_ce.append([trainer.train(pair, lr, self.w) for trainer in self.trainers])
            self.write_summary(asbe, epoch_sbe, ace, epoch_ce, sbe_ce)       
        return asbe, ace
    
    def write_summary(self, asbe, epoch_sbe, ace, epoch_ce, sbe_ce):
        sbe_ce = np.array(sbe_ce)
        epoch_sbe.append(np.mean(sbe_ce[:,:,0], 0))
        epoch_ce.append(np.mean(sbe_ce[:,:,1], 0))
        # asbe.append(np.mean(np.array(msbe)[-runtime:], 0))
        # ace.append(np.mean(np.array(mce)[-runtime:], 0))
        if len(asbe) == 0:
            asbe.append(sbe_ce[0,:,0])
            ace.append(sbe_ce[0,:,1])
        asbe.append(np.mean(epoch_sbe, 0))
        ace.append(np.mean(epoch_ce, 0))
        string = ''
        for i in range(len(self.trainers)):
            string += self.trainers[i].name + ': %.3f  '
        tmp = tuple(asbe[len(asbe)-1])
        #            tmp = tuple(ace[len(ace)-1])
        print('Epoch: %d, ' %(len(asbe)-1), string %tmp)    
                    

class trainer_mean:
    
    def __init__(self, env, name=None, attacker=[], attack_mode=None, gam=0.95, watch=[]):
        
        self.n = env.n
        self.p_n = np.array([np.zeros(env.observation_space[0].shape) for i in range(env.n)])
        self.p_tmp = np.array([np.zeros(env.observation_space[0].shape) for i in range(env.n)])
        
        self.att = attacker
        self.att_mode = attack_mode
        self.good = np.setdiff1d(np.arange(env.n), self.att)
        if len(watch) != 0:
            self.watch = watch
        else:
            self.watch = self.good
        
        self.name = name
        self.gam = gam
       
    def train(self, pair, lr, w):
        
        obs_n, next_obs_n, rew_n, z = [np.copy(pair[i]) for i in range(len(pair))]
        rew = np.mean(rew_n[self.good])
        
        # record sbe of watched agents
        sbe_watch = []
        for j in self.watch:
            sbe_watch.append(np.mean([(obs_n[i] @ self.p_n[j] - (rew + self.gam * next_obs_n[i] @ self.p_n[j]))**2 for i in range(self.n)]))
               
        # record consensus error of watched agents
        ce_watch = np.mean(np.square(self.p_n[self.watch] - np.mean(self.p_n[self.watch], axis=0)))
        
        # neighborhood update
        for i in self.good:
            self.p_tmp[i] = np.mean(self.p_n[w[i] != 0], 0)
        
        # local update
        for i in self.good:
            grad = (rew_n[i] + (self.gam * next_obs_n[i] - obs_n[i]) @ self.p_n[i]) * z      
            self.p_tmp[i] += lr * grad
        
        if len(self.att) != 0:
            for i in self.att:
                if self.att_mode == 0:
                    pass
                elif self.att_mode == 1:
                    noise = np.random.normal(0, 1, self.p_n[i].shape)    
                    self.p_n[i] = self.p_tmp[np.random.choice(self.good)] + noise  
                elif self.att_mode == 2:
                    grad = (rew_n[i] + (self.gam * next_obs_n[i] - obs_n[i]) @ self.p_n[i]) * z    
                    self.p_n[i] += lr * grad
                    self.p_n[i] = -self.p_n[i]
        
        # # neighborhood update
        # for i in self.good:
        #     self.p_n[i] = np.mean(self.p_tmp[w[i] != 0], 0)
        
        for i in self.good:
            self.p_n[i] = np.copy(self.p_tmp[i])  
        # for i in self.good:
        #     self.p_tmp[i] = np.copy(self.p_n[i])     
        return np.mean(sbe_watch), ce_watch


class trainer_trim:
    
    def __init__(self, env, name=None, attacker=[], attack_mode=None, b=0, gam=0.95, watch=[]):
        
        self.n = env.n
        self.p_n = np.array([np.zeros(env.observation_space[0].shape) for i in range(env.n)])
        self.p_tmp = np.array([np.zeros(env.observation_space[0].shape) for i in range(env.n)])
        
        self.att = attacker
        self.good = np.setdiff1d(np.arange(env.n), self.att)
        if len(watch) != 0:
            self.watch = watch
        else:
            self.watch = self.good
        self.att_mode = attack_mode
        
        self.name = name
        self.gam = gam
        self.b = b
       
    def train(self, pair, lr, w):
        
        obs_n, next_obs_n, rew_n, z = [np.copy(pair[i]) for i in range(len(pair))]
        rew = np.mean(rew_n[self.good])
        b = self.b
        
        # record sbe of watched agents
        sbe_watch = []
        for j in self.watch:
            sbe_watch.append(np.mean([(obs_n[i] @ self.p_n[j] - (rew + self.gam * next_obs_n[i] @ self.p_n[j]))**2 for i in range(self.n)]))
                   
        # record consensus error of watched agents
        ce_watch = np.mean(np.square(self.p_n[self.watch] - np.mean(self.p_n[self.watch], axis=0)))
        
        # neighborhood update
        for i in self.good:
            v = np.copy(w[i])
            v[i] = 0
            N = np.sum(v != 0)
            prop = np.min((b/N, 0.5))
            if (N-2*np.floor(N*prop)) >= 1:
                self.p_tmp[i] = (stats.trim_mean(self.p_n[v != 0], prop, axis=0) * N + self.p_n[i])/(N+1)
            else:
                self.p_tmp[i] = np.copy(self.p_n[i])
            
        # local update
        for i in self.good:
            grad = (rew_n[i] + (self.gam * next_obs_n[i] - obs_n[i]) @ self.p_n[i]) * z      
            self.p_tmp[i] += lr * grad
        
        if len(self.att) != 0:
            for i in self.att:
                if self.att_mode == 0:
                    pass
                elif self.att_mode == 1:
                    noise = np.random.normal(0, 1, self.p_n[i].shape)    
                    self.p_n[i] = self.p_tmp[np.random.choice(self.good)] + noise    
                elif self.att_mode == 2:
                    grad = (rew_n[i] + (self.gam * next_obs_n[i] - obs_n[i]) @ self.p_n[i]) * z     
                    self.p_n[i] += lr * grad
                    self.p_n[i] = -self.p_n[i]
        
        # # neighborhood update
        # for i in self.good:
        #     N = np.sum(w[i] != 0)
        #     prop = np.min((b/N, 0.5))
        #     if (N-2*np.floor(N*prop)) >= 1:
        #         self.p_n[i] = stats.trim_mean(self.p_tmp[w[i] != 0], prop, axis=0)
        #     else:
        #         self.p_n[i] = np.copy(self.p_tmp[i])
  
        for i in self.good:
            self.p_n[i] = np.copy(self.p_tmp[i])           
        # for i in self.good:
        #     self.p_tmp[i] = np.copy(self.p_n[i])
        return np.mean(sbe_watch), ce_watch


def main(args):
    mc = args.mc
    asbes, aces = [], []
    for i in range(mc):
        # configs
        if args.network in ('h1b1', 'h3b1', 'h2b1'):
            env = make_env('simple_spread_custom_local_12')
            attacker = [1,3,5,7,9,11]
            b = 1  
        elif args.network == 'h3b2':
            env = make_env('simple_spread_custom_local_18')
            attacker = [1,2,4,5,7,8,10,11,13,14,16,17]
            b = 2
        
        try:
            w = np.loadtxt(args.network+'.txt')
        except:
            if args.network == 'complete':
                env = make_env('simple_spread_custom_local_9')
                w = np.ones((env.n,env.n))
                attacker = [0,1]
                b = len(attacker) 
            elif args.network == 'renyi':
                env = make_env('simple_spread_custom_local_9')
                _, _, attacker, connectivity = renyigraph.Renyi(9, 0.7, 0.2)
                while len(attacker) > 1 or len(attacker) < 1:
                    _, _, attacker, connectivity = renyigraph.Renyi(9, 0.7, 0.2)
                w = connectivity
                b = int(len(attacker))
            else:
                raise Exception("Implemented network: h1b1, h2b1, h3b1, h3b2, renyi, complete.")        
        mode = args.attack
        
        watch = np.setdiff1d(np.arange(env.n), attacker)
        trainers = [trainer_mean(env, 'Mean', watch=watch), 
                    trainer_trim(env, 'Trim', b=b, watch=watch),
                    trainer_mean(env, 'Mean_att', attacker, mode),
                    trainer_trim(env, 'Trim_att', attacker, mode, b)]
        network = decent_network(env, w, trainers)
        asbe, ace = network.train(args.epoch, args.horizon, args.lr, 
                                    args.lam, args.diminish, args.render)
        asbes.append(asbe)
        aces.append(ace)
        
    # write result
    f = open('sbe'+'_'+args.network+'_a'+str(args.attack)+'_lam'+str(args.lam)+'.pkl', 'wb')
    pickle.dump(asbes, f)
    f.close()
    f = open('ce'+'_'+args.network+'_a'+str(args.attack)+'_lam'+str(args.lam)+'.pkl', 'wb')
    pickle.dump(aces, f)
    f.close()
    
    if args.plot:
        plotter.plot(args)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Robust Temporal Difference Learning')
    
    # Arguments
    parser.add_argument('--network', type=str, default='complete',
    help='name of the network (h1b1, h2b1, h3b1, h3b2, renyi, complete)')
    parser.add_argument('--attack', type=int, default=2,
    help='Attack method of Byzantine agents (0, 1, 2)')
    parser.add_argument('--mc', type=int, default=1,
    help='How many different seeds the code will be run on')
    parser.add_argument('--epoch', type=int, default=200,
    help='max epoch')
    parser.add_argument('--horizon', type=int, default=32,
    help='tragectory length')
    parser.add_argument('--lr', type=float, default=5e-2,
    help='initial learning rate')
    parser.add_argument('--lam', type=float, default=0.,
    help='lambda')
    parser.add_argument('--diminish', action='store_true',
    help='whether to use diminishing step size')
    parser.add_argument('--plot', action='store_true',
    help='whether to plot result')
    parser.add_argument('--render', action='store_true',
    help='whether to render')
    parser.add_argument('--lnk', action='store_true')
    args = parser.parse_args()
    
    main(args)