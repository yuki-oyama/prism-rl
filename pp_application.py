import numpy as np
import pandas as pd
from optimparallel import minimize_parallel
from core.model import PrismRL, RL
from core.graph import Graph
from core.utils import Timer
from core.dataset import *
import time
import json
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
model_arg.add_argument('--rl', type=str2bool, default=True, help='if estimate RL or not')
model_arg.add_argument('--prism', type=str2bool, default=False, help='if estimate prism RL or not')
model_arg.add_argument('--parallel', type=str2bool, default=False, help='if implement parallel computation or not')
model_arg.add_argument('--version', type=str, default='neg', help='version name')

# Hyperparameters
model_arg.add_argument('--state_key', type=str, default='d', help='od or d')
model_arg.add_argument('--T', type=int, default=15, help='time constraint')
model_arg.add_argument('--T_range', type=float, default=1.34, help='range for T')
model_arg.add_argument('--uturn', type=str2bool, default=True, help='if add uturn dummy or not')
model_arg.add_argument('--uturn_penalty', type=float, default=-10., help='penalty for uturn')
model_arg.add_argument('--min_n', type=int, default=0, help='minimum number observed for d')

# parameters
model_arg.add_argument('--vars', nargs='+', type=str, default=['length', 'crosswalk', 'greenlen'], help='explanatory variables')
model_arg.add_argument('--init_beta', nargs='+', type=float, default=[-0.266, -0.791, 0.049], help='initial parameter values')
model_arg.add_argument('--lb', nargs='+', type=float_or_none, default=[None,None,None], help='lower bounds')
model_arg.add_argument('--ub', nargs='+', type=float_or_none, default=[0,0,None], help='upper bounds')

# Validation
model_arg.add_argument('--n_samples', type=int, default=1, help='number of samples')
model_arg.add_argument('--test_ratio', type=float, default=0., help='ratio of test samples')
model_arg.add_argument('--two_step', type=str2bool, default=False, help='if using prism results')

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

# %%
if __name__ == '__main__':
    config, _ = get_config()
    config.version += '_' + time.strftime("%Y%m%dT%H%M")
    timer = Timer()
    _ = timer.stop()

    # %%
    network_ = 'kannai'
    dir_ = f'data/{network_}/'
    link_data = pd.read_csv(dir_+'link_bidir_rev2302.csv')
    node_data = pd.read_csv(dir_+'node.csv')
    obs_data = pd.read_csv(dir_+'observations_link.csv')

    # add negative var of street without sidewalk
    link_data['length'] /= 10
    link_data['width'] /= 10
    link_data['walkwidth2'] /= 10
    link_data['pedst'] = (link_data['side'] == 2) * 1
    link_data['walkratio'] = link_data['walkwidth2']/link_data['width']
    link_data['carst'] = (link_data['walkwidth2'] == 0) * 1 * (link_data['crosswalk'] == 0) * 1
    # interactions with length
    link_data['greenlen'] = link_data['green'] * link_data['length']
    # link_data['greenlen'] = link_data['vegetation'] * link_data['length']
    link_data['skylen'] = link_data['sky'] * link_data['length']
    link_data['pedstlen'] = link_data['pedst'] * link_data['length']
    link_data['pedstgreenlen'] = link_data['pedst'] * link_data['green'] * link_data['length']
    link_data['walkgreenlen'] = link_data['walkratio'] * link_data['green'] * link_data['length']
    link_data['sidewalklen'] = link_data['walkwidth2'] * link_data['length']
    link_data['gradlen'] = link_data['gradient'] * link_data['length']
    features = link_data

    # %%
    obs_data = reset_index(link_data, node_data, obs_data)
    links = {link_id: (from_, to_) for link_id, from_, to_ in link_data[['link_id', 'from_', 'to_']].values}

    # %%
    dests, obs, obs_filled, n_paths, max_len, od_data, samples = read_mm_results(
        obs_data, links, min_n_paths=config.min_n, n_samples=config.n_samples, test_ratio=config.test_ratio, seed_=111)

    # %%
    # number of paths
    print(f"number of paths observed: {n_paths}")
    # loop counts
    n_loops = count_loops(obs_filled)
    for d, n_loop in n_loops.items():
        if n_loop > 0:
            print(f"number of paths including loops observed: {n_loop} for destination {d}")

    # %%
    # Graph
    g = Graph()
    g.read_data(node_data=node_data, link_data=link_data, od_data=od_data)
    # g.update(T=T)

    # %%
    if config.prism:
        # %%
        detour_df = analyze_detour_rate(g, obs)

        # %%
        g.define_T_from_obs(detour_df, range=config.T_range, default_T=0)
        print(f"T = {g.T}")

        # %%
        timer.start()
        g.get_state_networks(method=config.state_key, parallel=True)
        snet_time = timer.stop()
        print(f"time to get snets is {snet_time}s.")

    # %%
    # variables
    xs = {
        'length': (features['length'].values, 'link'),
        'crosswalk': (features['crosswalk'].values, 'link'),
        'greenlen': (features['greenlen'].values, 'link'),
        'sidewalklen': (features['sidewalklen'].values, 'link'),
        'skylen': (features['skylen'].values, 'link'),
        'gradlen': (features['gradlen'].values, 'link'),
    }

    # %%
    # add uturn dummy
    U = (g.senders[:,np.newaxis] == g.receivers[np.newaxis,:]) * (g.receivers[:,np.newaxis] == g.senders[np.newaxis,:])
    U = np.where(U == True)[0]
    uturns = np.zeros_like(g.senders)
    uturns[U] = 1.
    xs['uturn'] = (uturns, 'edge')

    # %%
    betas = []
    for var_name, init_val, lb, ub in zip(config.vars, config.init_beta, config.lb, config.ub):
        betas.append(
            (f'b_{var_name}', init_val, lb, ub, var_name, 0)
        )
    if config.uturn:
        betas.append(('b_uturn', config.uturn_penalty, None, None, 'uturn', 1))

    # %%
    rl = RL(g, xs, betas)
    prism = PrismRL(g, xs, betas, method=config.state_key, parallel=config.parallel, print_process=False)

    ### FOR TWO-STEP ESTIMATION
    if config.two_step:
        prism_res_data = pd.read_csv(f'results/pp_application/{network_}/estimation/PrismRL_greenlen.csv', index_col=0).T
        twostep_init_beta = prism_res_data[['beta_length', 'beta_crosswalk', 'beta_greenlen']].values[0]
        print(f"Two step estimation: starting with {twostep_init_beta}")

    ### Model Estimation
    # output
    outputs = {}
    if config.rl: outputs['RL'] = {i:{} for i in range(config.n_samples)}
    if config.prism: outputs['PrismRL'] = {i:{} for i in range(config.n_samples)}
    # function for record results
    def record_res(i, model_type, res, stderr, t_val, L0, L_val, runtime):
        outputs[model_type][i] = {
            'L0': L0,
            'LL': -res.fun,
            'Lv': L_val,
            'runtime': runtime,
        }
        for var_name, b, s, t in zip(config.vars, res.x, stderr, t_val):
            outputs[model_type][i].update({
                f'beta_{var_name}': b, f'se_{var_name}': s, f't_{var_name}': t,
            })

    for i, sample in enumerate(samples):
        train_obs = sample['train']
        test_obs = sample['test']

        # %%
        if config.rl:
            # %%
            # only observed destinations in samples
            rl.partitions = list(train_obs.keys())

            # %%
            # rl.beta = np.array(config.init_beta)
            rl.beta = np.array(config.init_beta)
            if config.two_step: rl.beta = twostep_init_beta
            LL0_rl = rl.calc_likelihood(observations=train_obs)
            print('RL model initial log likelihood:', LL0_rl)

            # %%
            try:
                # %%
                print(f"RL model estimation for sample {i}...")

                # %%
                def f(x):
                    # compute probability
                    rl.eval_prob(x)
                    # calculate log-likelihood
                    LL = 0.
                    for key_, paths in train_obs.items():
                        p = rl.p[key_]
                        max_len, N = paths.shape
                        Lk = np.zeros(N, dtype=np.float)
                        for j in range(max_len - 1):
                            L = np.array(p[paths[j], paths[j+1]])[0]
                            assert (L > 0 ).all(), f'L includes zeros: key_={key_}, j={j}, pathj={paths[j]}, pathj+1={paths[j+1]}'
                            Lk += np.log(L)
                        LL += np.sum(Lk)
                    return -LL

                # %%
                timer.start()
                # results_rl = rl.estimate(observations=train_obs, method='L-BFGS-B', disp=False, hess='res')
                results_rl = minimize_parallel(f, x0=rl.beta, bounds=rl.bounds, options={'disp':False, 'maxiter':100}, callback=callbackF) #, parallel={'max_workers':4, 'verbose': True}
                rl_time = timer.stop()
                print(f"estimation time is {rl_time}s.")
                rl.beta = results_rl.x
                cov_matrix = results_rl.hess_inv if type(results_rl.hess_inv) == np.ndarray else results_rl.hess_inv.todense()
                stderr = np.sqrt(np.diag(cov_matrix))
                t_val = results_rl.x / stderr
                rl.print_results(results_rl, stderr, t_val, LL0_rl)
                # %%
                if config.test_ratio > 0:
                    # validation
                    rl.partitions = list(test_obs.keys())
                    LL_val_rl = rl.calc_likelihood(observations=test_obs)
                    print('RL model validation log likelihood:', LL_val_rl)
                else:
                    LL_val_rl = 0.
                # %%
                # record results
                record_res(i, 'RL', results_rl, stderr, t_val, LL0_rl, LL_val_rl, rl_time)
            except:
                print(f"RL is not feasible for sample {i}")

        # %%
        if config.prism:
            # %%
            s_train_obs = prism.translate_observations(train_obs)
            s_test_obs = prism.translate_observations(test_obs)

            # %%
            # prism.beta = np.array(config.init_beta)
            prism.beta = np.array(config.init_beta)
            prism.partitions = list(s_train_obs.keys())
            LL0 = prism.calc_likelihood(observations=s_train_obs)
            print('prism model initial log likelihood:', LL0)

            def f(x):
                # compute probability
                prism.eval_prob(x)
                # calculate log-likelihood
                LL = 0.
                for key_, paths in s_train_obs.items():
                    p = prism.p[key_]
                    max_len, N = paths.shape
                    Lk = np.zeros(N, dtype=np.float)
                    for j in range(max_len - 1):
                        L = np.array(p[paths[j], paths[j+1]])[0]
                        assert (L > 0 ).all(), f'L includes zeros: key_={key_}, j={j}, pathj={paths[j]}, pathj+1={paths[j+1]}'
                        Lk += np.log(L)
                    LL += np.sum(Lk)
                return -LL
            # estimation
            print(f"Prism RL model estimation for sample {i}...")
            Niter = 1
            timer.start()
            prism_res = minimize_parallel(f, x0=prism.beta, bounds=prism.bounds, options={'disp':False}, callback=callbackF)
            prism_time = timer.stop()
            print(f"estimation time is {prism_time}s.")
            # after estimation
            prism.beta = prism_res.x
            cov_matrix = prism_res.hess_inv if type(prism_res.hess_inv) == np.ndarray else prism_res.hess_inv.todense()
            stderr = np.sqrt(np.diag(cov_matrix))
            t_val = prism_res.x / stderr
            prism.print_results(prism_res, stderr, t_val, LL0)
            print(prism_res)

            # %%
            # validation
            if config.test_ratio > 0:
                prism.partitions = list(s_test_obs.keys())
                LL_val = prism.calc_likelihood(observations=s_test_obs)
                print('Prism RL model validation log likelihood:', LL_val)
            else:
                LL_val = 0.

            # %%
            # record results
            record_res(i, 'PrismRL', prism_res, stderr, t_val, LL0, LL_val, prism_time)

    if config.rl:
        df_rl = pd.DataFrame(outputs['RL']).T
        print(df_rl)
        if config.test_ratio > 0:
            # for validation
            df_rl.to_csv(f'results/pp_application/{network_}/validation/RL_{config.version}.csv', index=True)
        else:
            # for estimation
            df_rl.T.to_csv(f'results/pp_application/{network_}/estimation/RL_{config.version}.csv', index=True)
    if config.prism:
        df_prism = pd.DataFrame(outputs['PrismRL']).T
        print(df_prism)
        if config.test_ratio > 0:
            # for validation
            df_prism.to_csv(f'results/pp_application/{network_}/validation/PrismRL_{config.version}.csv', index=True)
        else:
            # for estimation
            df_prism.T.to_csv(f'results/pp_application/{network_}/estimation/PrismRL_{config.version}.csv', index=True)

    # %%
    # write config file
    dir_ = f'results/pp_application/{network_}/'
    dir_ = dir_ + 'validation/' if config.test_ratio > 0 else dir_ + 'estimation/'
    with open(f"{dir_}{config.version}.json", mode="w") as f:
        json.dump(config.__dict__, f, indent=4)
