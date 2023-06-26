import numpy as np
import pandas as pd
from optimparallel import minimize_parallel
from core.model import PrismRL, RL
from core.graph import Graph
from core.utils import Timer
from core.dataset import count_loops
import time
import argparse
np.random.seed(111)

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
  return v.lower() in ('true', '1')

def float_or_none(value):
    try:
        return float(value)
    except:
        return None

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--version', type=str, default='T15', help='version name')
model_arg.add_argument('--rl', type=str2bool, default=True, help='if estimate RL or not')
model_arg.add_argument('--prism', type=str2bool, default=True, help='if estimate prism RL or not')
model_arg.add_argument('--true_beta', nargs='+', type=float, default=[-2.5,2.], help='true parameter values')
model_arg.add_argument('--init_beta', nargs='+', type=float, default=[-1,-1], help='initial parameter values')
model_arg.add_argument('--uturn_penalty', type=float, default=-10., help='penalty for uturn')
model_arg.add_argument('--lb', nargs='+', type=float_or_none, default=[None,None], help='lower bounds')
model_arg.add_argument('--ub', nargs='+', type=float_or_none, default=[None,None], help='upper bounds')
model_arg.add_argument('--T', type=int, default=15, help='time constraint')
model_arg.add_argument('--uturn', type=str2bool, default=True, help='if add uturn dummy or not')
model_arg.add_argument('--state_key', type=str, default='d', help='od or d')
model_arg.add_argument('--n_obs', type=int, default=1000, help='number of samples for each od')
model_arg.add_argument('--n_samples', type=int, default=1, help='number of samples')
model_arg.add_argument('--parallel', type=str2bool, default=False, help='if implement parallel computation or not')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed

Niter = 1
def callbackF(x):
    global Niter
    txt = f'{Niter: d}'
    for i in range(len(x)): txt += f'\t{x[i]:.4f}'
    print(txt)
    Niter += 1

def generate_data(config):
    os = [1,2,3,4,5,6]
    ds = [8,12,16,20]
    observations_d = {}
    N = config.n_obs
    max_len = 0
    for d in ds:
        seq = rl.sample_path(os, d, N)
        if seq.shape[0] >= max_len:
            max_len = seq.shape[0]
        observations_d[d] = seq
    obs = {i:{} for i in range(config.n_samples)}

    # multiple samples
    if config.n_samples > 1 and config.state_key == 'd':
        # define for each sample
        sample_size = (N * len(os) * len(ds)) // config.n_samples
        for i in range(config.n_samples):
            d_counts = np.bincount(np.random.choice(ds, sample_size))
            for d in ds:
                idx_ = np.random.choice(np.arange(N * len(os)), d_counts[d], replace=False)
                obs[i][d] = observations_d[d][:, idx_]
    else:
        obs[0] = observations_d

    # loop counts
    n_loops = count_loops(observations_d)
    print(n_loops)
    return obs, observations_d, max_len

# %%
if __name__ == '__main__':
    config, _ = get_config()
    config.version += '_' + time.strftime("%Y%m%dT%H%M")
    timer = Timer()
    _ = timer.stop()

    # networks
    dir_ = 'data/network/SiouxFalls/'
    node_data = pd.read_csv(dir_+'node.csv')
    link_data = pd.read_csv(dir_+'link.csv')
    od_data = pd.read_csv(dir_+'od.csv')
    node_data['node_id'] = node_data['fid']
    link_data['link_id'] = link_data['fid']
    link_data['from_'] = link_data['O']
    link_data['to_'] = link_data['D']
    T = config.T

    # Graph
    g = Graph()
    g.read_data(node_data=node_data, link_data=link_data, od_data=od_data)

    # variables
    features = link_data.copy()
    features['capacity'] /= features['capacity'].max()
    xs = {
        'length': (features['length'], 'link'),
        'caplen': (features['length'] * features['capacity'], 'link'),
    }

    # true parameters & probtbility
    true_beta, lb, ub = np.array(config.true_beta), config.lb, config.ub
    betas = [
        ('b_len', true_beta[0], lb[0], ub[0], 'length', 0),
        ('b_cap', true_beta[1], lb[1], ub[1], 'caplen', 0),
    ]

    # add uturn dummy
    if config.uturn:
        U = (g.senders[:,np.newaxis] == g.receivers[np.newaxis,:]) * (g.receivers[:,np.newaxis] == g.senders[np.newaxis,:])
        U = np.where(U == True)[0]
        uturns = np.zeros_like(g.senders)
        uturns[U] = 1.
        xs['uturn'] = (uturns, 'edge')
        betas.append(('b_uturn', config.uturn_penalty, None, None, 'uturn', 1))

    # define models
    rl = RL(g, xs, betas)
    rl.eval_prob()

    ### Data Generation
    obs, observations_d, max_len = generate_data(config)

    # %%
    ### Model Estimation
    # output
    outputs = {}
    if config.rl: outputs['RL'] = {i:{} for i in range(config.n_samples)}
    if config.prism: outputs['PrismRL'] = {i:{} for i in range(config.n_samples)}
    # function for record results
    def record_res(i, model_type, res, stderr, t_val, L0, runtime):
        outputs[model_type][i] = {
            'beta_len': res.x[0], 'se_len': stderr[0], 't_len': t_val[0],
            'beta_cap': res.x[1], 'se_cap': stderr[1], 't_cap': t_val[1],
            'L0': L0,
            'LL': -res.fun,
            'runtime': runtime,
        }

    ## Estimation
    for i in range(config.n_samples):
        if config.rl:
            rl.beta = np.array(config.init_beta)
            LL0_rl = rl.calc_likelihood(observations=obs[i])
            print('RL model initial log likelihood:', LL0_rl)

            try:
                print(f"RL model estimation for sample {i}...")
                timer.start()
                results_rl = rl.estimate(observations=obs[i], method='L-BFGS-B', disp=False, hess='res')
                rl_time = timer.stop()
                print(f"estimation time is {rl_time}s.")
                t_rl = (true_beta - results_rl[0].x)/results_rl[2]
                rl.print_results(results_rl[0], results_rl[2], t_rl, LL0_rl)
                record_res(i, 'RL', results_rl[0], results_rl[2], t_rl, LL0_rl, rl_time)
            except:
                print(f'RL failed for sample {i}')

        if config.prism:
            print("Update network data for prism RL...")
            T = config.T if config.T >= max_len else max_len
            print(f'T is {T}.')
            if config.state_key == 'd':
                g.update(T={d: T for d in g.dests})
            else:
                g.update(T={od: T for od in g.ods})
            g.get_state_networks(method=config.state_key, parallel=False)

            # define model
            prism = PrismRL(g, xs, betas, method=config.state_key)
            # prepare observations
            observations = prism.translate_observations(obs[i]) #obs[config.state_key]
            # initial log likelihood
            prism.beta = np.array(config.init_beta)
            LL0 = prism.calc_likelihood(observations=observations)
            print('prism model initial log likelihood:', LL0)

            print(f"Prism RL model estimation for sample {i}...")
            if not config.parallel:
                timer.start()
                results = prism.estimate(observations=observations, method='L-BFGS-B', disp=False, hess='res')
                prism_time = timer.stop()
                print(f"estimation time is {prism_time}s.")
                t_prism = (true_beta - results[0].x)/results[2]
                prism.print_results(results[0], results[2], t_prism, LL0)
                record_res(i, 'PrismRL', results[0], results[2], t_prism, LL0, prism_time)
            else:
                def f(x):
                    # compute probability
                    prism.eval_prob(x)
                    # calculate log-likelihood
                    LL = 0.
                    for key_, paths in observations.items():
                        p = prism.p[key_]
                        max_len, N = paths.shape
                        Lk = np.zeros(N, dtype=np.float)
                        for j in range(max_len - 1):
                            L = np.array(p[paths[j], paths[j+1]])[0]
                            assert (L > 0 ).all(), f'L includes zeros: key_={key_}, j={j}, pathj={paths[j]}, pathj+1={paths[j+1]}'
                            Lk += np.log(L)
                        LL += np.sum(Lk)
                    return -LL
                Niter = 1
                timer.start()
                prism_res = minimize_parallel(f, x0=prism.beta, bounds=prism.bounds, options={'disp':False}, callback=callbackF)
                prism_time = timer.stop()
                print(f"estimation time is {prism_time}s.")
                # after estimation
                prism.beta = prism_res.x
                cov_matrix = prism_res.hess_inv if type(prism_res.hess_inv) == np.ndarray else prism_res.hess_inv.todense()
                stderr = np.sqrt(np.diag(cov_matrix))
                t_val = (true_beta - prism_res.x) / stderr
                prism.print_results(prism_res, stderr, t_val, LL0)
                # print(prism_res)
                record_res(i, 'PrismRL', prism_res, stderr, t_val, LL0, prism_time)

    if config.rl:
        df_rl = pd.DataFrame(outputs['RL']).T
        print(df_rl)
        df_rl.to_csv(f'results/reproducibility/RL_{config.version}.csv', index=True)
    if config.prism:
        df_prism = pd.DataFrame(outputs['PrismRL']).T
        print(df_prism)
        df_prism.to_csv(f'results/reproducibility/PrismRL_{config.version}.csv', index=True)


##### Generate synthetic data (since real data cannot be publicly shared) #####

# idx_to_id = {idx_:id_ for idx_, id_ in zip(link_data.index.values, link_data['link_id'])}
# idx_to_len = {idx_:len_ for idx_, len_ in zip(link_data.index.values, link_data['length'])}
#
# # %%
# synthetic = []
# trip_id = 1
# for d in observations_d.keys():
#     paths = observations_d[d].T
#     for k in range(paths.shape[0]):
#         if np.random.rand() > 0.2:
#             continue
#         path = paths[k]
#         for link in path:
#             if link not in idx_to_id.keys():
#                 break
#             else:
#                 link_id = idx_to_id[link]
#                 link_len = idx_to_len[link]
#                 synthetic.append({'trip_id': trip_id, 'link_id': link_id, 'link_len': link_len})
#         trip_id += 1
#
# # %%
# df = pd.DataFrame(synthetic)
# df.to_csv('synthetic_data.csv', index=False)
