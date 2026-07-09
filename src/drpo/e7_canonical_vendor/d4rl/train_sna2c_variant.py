#!/usr/bin/env python3
"""
train_sna2c_variant.py — SNA2C + 3 negative-shaping variants for the α-follow-up
ablation (branch v7, 2026-05-01). Adapted from train_sna2c_only.py.

Variants:
  --variant base     : standard SNA2C (all negatives × α)
  --variant ptrunc   : only worst --p fraction of negatives × α, rest=0
  --variant negup    : negative weight ∝ |adv| (worse=bigger weight); --shape
  --variant negdown  : negative weight ∝ 1/|adv| (worse=smaller weight); --shape

Checkpoints are written to
  <ckpt_dir>/<algo_label>_seed<seed>/step_XXXXXXX.pt
so eval_d4rl_mp_greedy.py can pick them up directly.

Usage example:
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python train_sna2c_variant.py --dataset hopper-medium-expert-v2 \
      --variant ptrunc --p 0.5 --alpha 0.11 --seed 42 --steps 1000000 \
      --out_dir results_sna2c_v7 --ckpt_dir results_sna2c_v7/ckpts
"""
import numpy as np, os, time, json, argparse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import h5py
import gymnasium as gym
try:
    import gymnasium_robotics
except ImportError:
    pass

_nt = int(os.environ.get('OMP_NUM_THREADS', 2))
torch.set_num_threads(_nt)

# 2026-05-14 capacity sweep: 允许从 CLI 动态覆盖 NETWORK_PRESETS。
# 必须在从 agents 导入 Actor/QNet/Critic 类**之前**调用 set_network_preset，
# 否则类在实例化时读的是定义时的全局 _PRESET。
# 这里先 import _agents_module（仅读 _PRESET / set_network_preset），然后在 main()
# 里解析完 CLI 后才 import 具体 Agent 类。
import agents as _agents_module
from agents import set_network_preset, NETWORK_PRESETS

# ── D4RL normalization constants (match train_sna2c_only.py) ────────────
# 2026-05-11 P0-B refactor: NORM dict moved to d4rl_common.D4RL_REF
# (project-wide single source of truth; 15/15 entries audited byte-equal).
# normalize_score() preserved as thin shim — same `(dataset, raw)` signature,
# same "return raw if dataset not in NORM" fallback for byte-identical behavior.
from d4rl_common import D4RL_REF, normalize_score as _ns_common  # noqa: E402

# 2026-05-11 P1-A refactor: load_hdf5 / compute_mc_returns / evaluate /
# EVAL_ENV moved to d4rl_common.train_loop (project-wide single source of truth).
from d4rl_common.train_loop import (  # noqa: E402
    load_hdf5, compute_mc_returns, evaluate,
    EVAL_ENV_LOCOMOTION as EVAL_ENV,
)

def normalize_score(dataset, raw):
    if dataset not in D4RL_REF: return raw
    return _ns_common(dataset, raw)

# 2026-05-10 P0-A refactor: reward_norm_locomotion moved to d4rl_common.normalize
# (project-wide single source of truth, AST-verified byte-identical). Module-level
# alias preserved for backward compatibility.
from d4rl_common import reward_norm_locomotion  # noqa: E402


def get_obs_from_env(raw_obs, expected_dim=None):
    """Flatten Gymnasium observations, including dict observations.

    Goal-conditioned environments may return dict observations. AntMaze uses
    observation + desired_goal in some dataset formats, while Maze2D/Kitchen
    often match the flat HDF5 observation with the observation field only.
    """
    if isinstance(raw_obs, dict):
        obs_part = np.asarray(raw_obs['observation']).flatten().astype(np.float32)
        goal_part = raw_obs.get('desired_goal', None)
        if goal_part is not None and not isinstance(goal_part, dict):
            goal_arr = np.asarray(goal_part).flatten().astype(np.float32)
            if goal_arr.size > 0:
                combined = np.concatenate([obs_part, goal_arr])
                if expected_dim is None or combined.shape[0] == expected_dim:
                    return combined
        if expected_dim is not None:
            if obs_part.shape[0] < expected_dim:
                obs_part = np.concatenate([
                    obs_part,
                    np.zeros(expected_dim - obs_part.shape[0], dtype=np.float32),
                ])
            elif obs_part.shape[0] > expected_dim:
                obs_part = obs_part[:expected_dim]
        return obs_part.astype(np.float32)
    obs = np.asarray(raw_obs).flatten().astype(np.float32)
    if expected_dim is not None:
        if obs.shape[0] < expected_dim:
            obs = np.concatenate([obs, np.zeros(expected_dim - obs.shape[0], dtype=np.float32)])
        elif obs.shape[0] > expected_dim:
            obs = obs[:expected_dim]
    return obs.astype(np.float32)


def evaluate_custom(agent, dataset, env_name, n=10, seed=42,
                    goal_conditioned=False, expected_obs_dim=None,
                    max_steps=1000):
    if env_name is None:
        env_name = dataset
    env = gym.make(env_name)
    env.action_space.seed(seed + 10_000)
    returns = []
    for i in range(n):
        obs_out = env.reset(seed=seed + 20_000 + i)
        raw_obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        obs = get_obs_from_env(raw_obs, expected_dim=expected_obs_dim)
        ep_ret = 0.0
        success = False
        for _ in range(max_steps):
            action, _ = agent.get_action(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            result = env.step(action)
            if len(result) == 5:
                raw_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                raw_obs, reward, done, info = result
            obs = get_obs_from_env(raw_obs, expected_dim=expected_obs_dim)
            ep_ret += float(reward)
            if isinstance(info, dict) and float(info.get('success', 0.0)) > 0.0:
                success = True
            if reward > 0:
                success = True
            if done:
                break
        returns.append(float(success) if goal_conditioned else ep_ret)
    env.close()
    return float(np.mean(returns))


def build_agent(args, sdim, adim):
    if args.variant == 'base':
        tag = f'SNA2C(α={args.alpha:g})'
        return ScaledNegA2CAgent(sdim, adim, lr=args.lr, alpha=args.alpha), tag
    if args.variant == 'ptrunc':
        tag = f'SNA2C-Ptrunc(α={args.alpha:g},p={args.p:g})'
        return SNA2C_PTruncAgent(sdim, adim, lr=args.lr, alpha=args.alpha, p=args.p), tag
    if args.variant == 'negup':
        tag = f'SNA2C-NegUp(α={args.alpha:g},{args.shape})'
        return SNA2C_NegUpAgent(sdim, adim, lr=args.lr, alpha=args.alpha, shape=args.shape), tag
    if args.variant == 'negdown':
        if args.shape == 'exp2':
            tag = f'SNA2C-NegDown(α={args.alpha:g},exp2,T={args.temp:g})'
        elif args.shape == 'exp2_noNorm':
            tag = f'SNA2C-NegDown(α={args.alpha:g},exp2NN,T={args.temp:g})'
        else:
            tag = f'SNA2C-NegDown(α={args.alpha:g},{args.shape})'
        return SNA2C_NegDownAgent(sdim, adim, lr=args.lr, alpha=args.alpha,
                                   shape=args.shape, temp=args.temp), tag
    if args.variant == 'iqlv':
        tag = f'SNA2C-IQLV(α={args.alpha:g},τ={args.tau:g})'
        return SNA2C_IQLVAgent(sdim, adim, lr=args.lr, alpha=args.alpha, tau=args.tau), tag
    if args.variant == 'iqlv_exp':
        tag = f'SNA2C-IQLV-Exp(α={args.alpha:g},τ={args.tau:g},T={args.temp:g})'
        return SNA2C_IQLV_ExpAgent(sdim, adim, lr=args.lr, alpha=args.alpha,
                                    tau=args.tau, T=args.temp), tag
    if args.variant == 'iqlv_exp_std':
        tag = f'SNA2C-IQLV-ExpStd(α={args.alpha:g},τ={args.tau:g},T={args.temp:g})'
        return SNA2C_IQLV_ExpStdAgent(sdim, adim, lr=args.lr, alpha=args.alpha,
                                       tau=args.tau, T=args.temp), tag
    if args.variant == 'iqlv_exp_ema':
        tag = (f'SNA2C-IQLV-ExpEmaStd(α={args.alpha:g},τ={args.tau:g},'
               f'T={args.temp:g},β={args.ema_decay:g},wu={args.warmup_steps})')
        return SNA2C_IQLV_ExpEmaStdAgent(sdim, adim, lr=args.lr, alpha=args.alpha,
                                          tau=args.tau, T=args.temp,
                                          ema_decay=args.ema_decay,
                                          warmup_steps=args.warmup_steps), tag
    if args.variant == 'iqlv_exp_rank':
        tag = f'SNA2C-IQLV-ExpRank(α={args.alpha:g},τ={args.tau:g},T={args.temp:g})'
        return SNA2C_IQLV_ExpRankAgent(sdim, adim, lr=args.lr, alpha=args.alpha,
                                        tau=args.tau, T=args.temp), tag
    if args.variant == 'iqlv_exp_rank_noise_aug':
        # 2026-05-22: ExpRank + Q-free action-noise augmentation.
        tag = (f'SNA2C-IQLV-ExpRank-NoiseAug'
               f'(α={args.alpha:g},τ={args.tau:g},T={args.temp:g},'
               f'K={args.noise_aug_K},σ={args.noise_aug_sigma:g},'
               f'c={args.noise_aug_c:g},λ={args.noise_aug_lambda:g})')
        return SNA2C_IQLV_ExpRank_NoiseAugAgent(
            sdim, adim, lr=args.lr, alpha=args.alpha,
            tau=args.tau, T=args.temp,
            K=args.noise_aug_K, sigma=args.noise_aug_sigma,
            c=args.noise_aug_c, lam_aug=args.noise_aug_lambda), tag
    if args.variant == 'iqlv_exp_rank_baseline':
        # 2026-05-19 v7: ExpRank + global advantage baseline shift.
        # `baseline_value` is pre-computed in train() from the buffer reward
        # quantile (see args.baseline_p) and injected on args.
        b = float(getattr(args, 'baseline_value', 0.0))
        lam = float(getattr(args, 'neg_lambda', 1.0))
        tag = (f'SNA2C-IQLV-ExpRank-Baseline'
               f'(α={args.alpha:g},τ={args.tau:g},T={args.temp:g},'
               f'p={args.baseline_p:g},b={b:.4f},λ={lam:g})')
        return SNA2C_IQLV_ExpRankBaselineAgent(
            sdim, adim, lr=args.lr, alpha=args.alpha,
            tau=args.tau, T=args.temp, baseline=b,
            neg_lambda=lam), tag
    if args.variant == 'iqlpos_topk':
        tag = f'IQL-PosTopK(τ={args.tau:g},q={args.q:g})'
        return IQLPosTopKAgent(sdim, adim, lr=args.lr, tau=args.tau, q=args.q,
                               total_steps=args.steps), tag
    if args.variant == 'iql_vanilla':
        # IQL baseline (Kostrikov et al. 2021). Uses --tau (expectile) and
        # --temp (advantage temperature β). Defaults match the paper for
        # locomotion: tau=0.7, beta=3.0.
        tag = f'IQL(τ={args.tau:g},β={args.temp:g})'
        return IQLAgent(sdim, adim, lr=args.lr, tau=args.tau,
                        temperature=args.temp,
                        total_steps=args.steps), tag
    raise ValueError(args.variant)


def train(args):
    # 2026-05-14 capacity sweep: 在创建任何 Actor/QNet/Critic 之前覆盖 preset。
    # 仅当 任一覆盖参数 != -1 时才创建 custom preset，其他情况保持 default 不变（向后兼容）。
    if args.actor_h > 0 or args.critic_h > 0 or args.actor_depth > 0 or args.critic_depth > 0:
        custom = dict(NETWORK_PRESETS['default'])
        if args.actor_h > 0:      custom['actor_h']     = args.actor_h
        if args.critic_h > 0:     custom['v_h']         = args.critic_h
        if args.critic_h > 0:     custom['q_h']         = args.critic_h
        if args.actor_depth > 0:  custom['actor_depth'] = args.actor_depth
        if args.critic_depth > 0: custom['v_depth']     = args.critic_depth
        if args.critic_depth > 0: custom['q_depth']     = args.critic_depth
        # 注入 NETWORK_PRESETS（后续 eval 脚本也可读）
        NETWORK_PRESETS[args.preset_tag] = custom
        set_network_preset(args.preset_tag)
    elif args.preset_tag != 'default':
        # 只是重命名 preset_tag（跟踪用），不改网络参数
        pass
    # 在 set_preset 完成后才 import 所有 agent 类（这样它们实例化时读到的是新 _PRESET）。
    global ScaledNegA2CAgent, SNA2C_PTruncAgent, SNA2C_NegUpAgent, SNA2C_NegDownAgent
    global SNA2C_IQLVAgent, SNA2C_IQLV_ExpAgent, SNA2C_IQLV_ExpStdAgent
    global SNA2C_IQLV_ExpEmaStdAgent, SNA2C_IQLV_ExpRankAgent
    global SNA2C_IQLV_ExpRank_NoiseAugAgent
    global SNA2C_IQLV_ExpRankBaselineAgent
    global IQLPosTopKAgent, IQLAgent
    from agents import (ScaledNegA2CAgent, SNA2C_PTruncAgent,
                        SNA2C_NegUpAgent, SNA2C_NegDownAgent,
                        SNA2C_IQLVAgent, SNA2C_IQLV_ExpAgent,
                        SNA2C_IQLV_ExpStdAgent, SNA2C_IQLV_ExpEmaStdAgent,
                        SNA2C_IQLV_ExpRankAgent,
                        SNA2C_IQLV_ExpRank_NoiseAugAgent,
                        SNA2C_IQLV_ExpRankBaselineAgent,
                        IQLPosTopKAgent,
                        IQLAgent)

    dev = torch.device('cpu') if args.device is None else torch.device(f'cuda:{args.device}')
    _agents_module.DEVICE = dev
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    dataset = args.dataset
    if args.hdf5:
        hdf5_path = os.path.expanduser(args.hdf5)
    else:
        data_dir = os.path.expanduser(args.data_dir)
        hdf5_name = dataset.replace('-', '_') + '.hdf5'
        hdf5_path = os.path.join(data_dir, hdf5_name)
        if not os.path.exists(hdf5_path):
            hdf5_path = os.path.join(data_dir, dataset + '.hdf5')
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    ds = load_hdf5(hdf5_path, dataset_name=dataset)
    N = len(ds['obs']); sdim = ds['obs'].shape[1]; adim = ds['acts'].shape[1]
    ds['ep_ret'] = compute_mc_returns(ds['rews'], ds['terms'], ds['touts'])

    ds_dev = {
        'obs':      torch.from_numpy(ds['obs']).to(dev, dtype=torch.float32),
        'acts':     torch.from_numpy(ds['acts']).to(dev, dtype=torch.float32),
        'rews':     torch.from_numpy(ds['rews']).to(dev, dtype=torch.float32),
        'next_obs': torch.from_numpy(ds['next_obs']).to(dev, dtype=torch.float32),
        'terms':    torch.from_numpy(ds['terms']).to(dev, dtype=torch.bool),
        'ep_ret':   torch.from_numpy(ds['ep_ret']).to(dev, dtype=torch.float32),
    }
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)

    # 2026-05-21 return-weighted sampling: build per-transition probability vector.
    sampler_probs = None  # None means uniform fast path (torch.randint)
    if args.ret_weight_mode != 'none':
        ep_ret_np = ds['ep_ret']  # per-transition trajectory return, length N
        # Identify episode segments to compute trajectory-level rank.
        # ep_ret is constant within an episode; collect unique-by-segment values.
        ep_starts = np.zeros(N, dtype=bool)
        ep_starts[0] = True
        # episode boundary = where (term|tout) at step t-1
        ends_prev = (ds['terms'].astype(bool) | ds['touts'].astype(bool))
        ep_starts[1:] = ends_prev[:-1]
        # episode_id for each transition
        ep_id = np.cumsum(ep_starts) - 1
        n_ep = int(ep_id[-1]) + 1
        # per-episode return
        ep_ret_per_ep = np.zeros(n_ep, dtype=np.float64)
        ep_ret_per_ep[ep_id] = ep_ret_np  # last write wins; constant per ep so fine
        # Compute trajectory-level weight
        if args.ret_weight_mode == 'rank_pow':
            ranks_asc = np.empty(n_ep, dtype=np.int64)
            ranks_asc[np.argsort(ep_ret_per_ep)] = np.arange(n_ep)  # 0=worst
            w_ep = ((ranks_asc + 1) / n_ep) ** float(args.ret_weight_beta)
        elif args.ret_weight_mode == 'exp_return':
            r_min, r_max = ep_ret_per_ep.min(), ep_ret_per_ep.max()
            g = (ep_ret_per_ep - r_min) / max(r_max - r_min, 1e-9)
            w_ep = np.exp(float(args.ret_weight_alpha) * g)
        else:
            raise ValueError(args.ret_weight_mode)
        # Broadcast trajectory weight to per-transition; normalise.
        w_per_trans = w_ep[ep_id].astype(np.float64)
        w_per_trans = w_per_trans / w_per_trans.sum()
        sampler_probs = torch.from_numpy(w_per_trans).to(dev, dtype=torch.float32)
        # Diagnostic: top-K mass under this weighting.
        order_desc = np.argsort(-ep_ret_per_ep)
        ep_lens = np.bincount(ep_id, minlength=n_ep)
        trans_mass = w_ep * ep_lens
        trans_mass = trans_mass / trans_mass.sum()
        m_top1 = trans_mass[order_desc[:max(1, n_ep // 100)]].sum() * 100
        m_top5 = trans_mass[order_desc[:max(1, n_ep // 20)]].sum() * 100
        m_top10 = trans_mass[order_desc[:max(1, n_ep // 10)]].sum() * 100
        print(f'[ret_weight] mode={args.ret_weight_mode} '
              f'beta={args.ret_weight_beta} alpha={args.ret_weight_alpha} '
              f'n_ep={n_ep} top1%_mass={m_top1:.2f}% '
              f'top5%_mass={m_top5:.2f}% top10%_mass={m_top10:.2f}%',
              flush=True)

    # 2026-05-19 v7: compute global baseline for iqlv_exp_rank_baseline
    # variant from buffer reward quantile.  Frozen across training.
    if args.variant == 'iqlv_exp_rank_baseline':
        if args.baseline_abs is not None:
            args.baseline_value = float(args.baseline_abs)
            print(f'[baseline] mode=abs  b={args.baseline_value:.4f}',
                  flush=True)
        else:
            assert 0.0 <= args.baseline_p <= 1.0
            # quantile of buffer reward; computed on numpy to avoid GPU sync.
            args.baseline_value = float(
                np.quantile(ds['rews'], args.baseline_p))
            print(f'[baseline] mode=reward_quantile  p={args.baseline_p}  '
                  f'b={args.baseline_value:.4f}  '
                  f'(reward range: min={ds["rews"].min():.4f} '
                  f'max={ds["rews"].max():.4f} mean={ds["rews"].mean():.4f})',
                  flush=True)

    agent, algo_label = build_agent(args, sdim, adim)
    # 2026-05-14: 把 preset_tag 嵌入 algo_label，防止不同 capacity 输出互相覆盖。
    if args.preset_tag != 'default':
        algo_label = f'{algo_label}@{args.preset_tag}'
    tag = f"[{dataset}|{algo_label}|s{args.seed}]"
    print(f"\n{tag} sdim={sdim} adim={adim} N={N:,} steps={args.steps:,} "
          f"device={dev} OMP={_nt}", flush=True)

    hist = {'steps': [], algo_label: []}
    os.makedirs(args.out_dir, exist_ok=True)
    run_name = f'{dataset}_{algo_label}_seed{args.seed}'.replace('/', '_')
    log_path = os.path.join(args.out_dir, f'{run_name}.log')
    ckpt_dir = None
    if args.ckpt_dir:
        algo_key = f'{algo_label}_seed{args.seed}'
        ckpt_dir = os.path.join(args.ckpt_dir, algo_key)
        os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_start_step = int(args.steps * (1.0 - args.last_pct))
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if sampler_probs is None:
            idx = torch.randint(0, N, (args.batch,), generator=gen, device=dev)
        else:
            idx = torch.multinomial(sampler_probs, args.batch,
                                    replacement=True, generator=gen)
        s  = ds_dev['obs'].index_select(0, idx)
        a  = ds_dev['acts'].index_select(0, idx)
        r  = ds_dev['rews'].index_select(0, idx)
        ns = ds_dev['next_obs'].index_select(0, idx)
        d  = ds_dev['terms'].index_select(0, idx)
        ep = ds_dev['ep_ret'].index_select(0, idx)
        agent.update(s, a, r, ns, d, ep)

        if ckpt_dir is not None and step >= ckpt_start_step and step % args.ckpt_interval == 0:
            cp = os.path.join(ckpt_dir, f'step_{step:07d}.pt')
            torch.save({
                'actor': agent.actor.state_dict(),
                'sdim': sdim, 'adim': adim,
                'network_preset': args.preset_tag,
                'preset_actor_h':     getattr(args, 'actor_h', -1),
                'preset_critic_h':    getattr(args, 'critic_h', -1),
                'preset_actor_depth': getattr(args, 'actor_depth', -1),
                'preset_critic_depth':getattr(args, 'critic_depth', -1),
                'step': step, 'variant': args.variant,
                'alpha': args.alpha, 'p': args.p, 'shape': args.shape,
                'tau': args.tau, 'seed': args.seed,
            }, cp)

        if step % args.eval_interval == 0:
            if args.env:
                raw = evaluate_custom(
                    agent, dataset, args.env, n=args.eval_episodes,
                    seed=args.seed, goal_conditioned=args.goal_conditioned,
                    expected_obs_dim=sdim, max_steps=args.eval_max_steps)
            else:
                raw = evaluate(agent, dataset, n=args.eval_episodes, seed=args.seed)
            score = raw if args.goal_conditioned else normalize_score(dataset, raw)
            hist['steps'].append(step); hist[algo_label].append(score)
            elapsed = time.time() - t0; speed = step / elapsed if elapsed > 0 else 0
            eta = (args.steps - step) / speed if speed > 0 else 0
            metric_name = 'success' if args.goal_conditioned else 'raw'
            if args.goal_conditioned:
                metric_part = f"success={score:.3f}"
            else:
                metric_part = f"{metric_name}={raw:.3f} norm={score:.3f}"
            line = (f"{tag} step={step:>7d}/{args.steps} ({step/args.steps*100:.1f}%) "
                    f"| {speed:.0f}sps | ETA {eta/3600:.2f}h | "
                    f"{algo_label}: {metric_part}")
            print(line, flush=True)
            with open(log_path, 'a') as f: f.write(line + '\n')

    final_metric_name = 'success' if args.goal_conditioned else 'norm'
    print(f"\n{tag} ═══ Final ═══  {algo_label}: {final_metric_name}={hist[algo_label][-1]:.3f}")
    print(f"{tag} Done. {time.time()-t0:.0f}s", flush=True)

    summary = {
        'dataset': dataset, 'variant': args.variant, 'alpha': args.alpha,
        'p': args.p, 'shape': args.shape, 'tau': args.tau, 'seed': args.seed,
        'steps': args.steps, 'final_norm': hist[algo_label][-1],
        'final_score': hist[algo_label][-1],
        'score_type': final_metric_name,
        'goal_conditioned': args.goal_conditioned,
        'algo_label': algo_label, 'history': hist,
        'ret_weight_mode': args.ret_weight_mode,
        'ret_weight_beta': args.ret_weight_beta,
        'ret_weight_alpha': args.ret_weight_alpha,
    }
    json_path = os.path.join(args.out_dir, f'{run_name}_summary.json')
    with open(json_path, 'w') as f: json.dump(summary, f, indent=2)
    print(f"  Summary -> {json_path}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',        default='hopper-medium-expert-v2')
    ap.add_argument('--data_dir',       default='/root/d4rl/d4rl_datasets/locomotion')
    ap.add_argument('--hdf5',           default=None,
                    help='Explicit HDF5 path. Overrides --data_dir lookup when set.')
    ap.add_argument('--env',            default=None,
                    help='Explicit Gymnasium env id for non-locomotion evaluation.')
    ap.add_argument('--goal_conditioned', action='store_true',
                    help='Report eval success rate instead of episode return.')
    ap.add_argument('--eval_max_steps', type=int, default=1000,
                    help='Maximum environment steps per evaluation episode.')
    ap.add_argument('--variant',        required=True,
                    choices=['base', 'ptrunc', 'negup', 'negdown', 'iqlv',
                             'iqlv_exp', 'iqlv_exp_std',
                             'iqlv_exp_ema', 'iqlv_exp_rank',
                             'iqlv_exp_rank_noise_aug',
                             'iqlv_exp_rank_baseline',
                             'iqlpos_topk', 'iql_vanilla'])
    ap.add_argument('--alpha',          type=float, default=0.11)
    ap.add_argument('--p',              type=float, default=0.5,
                    help='ptrunc: fraction of worst negatives to penalise')
    ap.add_argument('--shape',          default='linear',
                    help="negup∈{linear,square,rank} | negdown∈{inv,neg_rank,exp,exp2}")
    ap.add_argument('--tau',            type=float, default=0.7,
                    help='iqlv: expectile tau (>0.5 raises V)')
    ap.add_argument('--q',              type=float, default=0.0,
                    help='iqlpos_topk: within-positive top-k quantile (0=orig, 0.5=top50%)')
    ap.add_argument('--temp',           type=float, default=1.0,
                    help='negdown exp2: temperature T in w∝exp(adv*T); T→0 ⇒ base SNA2C')
    ap.add_argument('--ema_decay',      type=float, default=0.99,
                    help='iqlv_exp_ema: EMA decay β for adv_neg std (0.99 ≙ half-life 69 steps)')
    ap.add_argument('--warmup_steps',   type=int,   default=5000,
                    help='iqlv_exp_ema: use batch_std for first N steps, then switch to EMA')
    # 2026-05-19 v7 baseline knobs
    ap.add_argument('--baseline_p',     type=float, default=0.5,
                    help='iqlv_exp_rank_baseline: reward-quantile p ∈ [0,1] '
                         'used as fixed advantage baseline; 0.5=median, '
                         '0.9=top-10% bar.')
    ap.add_argument('--baseline_abs',   type=float, default=None,
                    help='iqlv_exp_rank_baseline: optional absolute baseline '
                         '(overrides --baseline_p when set).')
    ap.add_argument('--neg_lambda',     type=float, default=1.0,
                    help='iqlv_exp_rank_baseline: explicit relative weight '
                         'for the negative-side gradient. λ = 1.0 reproduces '
                         'the prior implicit behavior; λ = 0 falls back to '
                         '"positives only".')
    # 2026-05-21 return-weighted sampling (boost top-1% trajectory mass)
    ap.add_argument('--ret_weight_mode', type=str, default='none',
                    choices=['none', 'rank_pow', 'exp_return'],
                    help='return-weighted batch sampler. none=uniform (default).')
    ap.add_argument('--ret_weight_beta', type=float, default=0.0,
                    help='rank_pow: w_traj = ((rank_asc+1)/N_ep)^beta. '
                         'beta=0 -> uniform; beta=8 -> top1% mass ~11%.')
    ap.add_argument('--ret_weight_alpha', type=float, default=0.0,
                    help='exp_return: w_traj = exp(alpha * g_norm), '
                         'g_norm in [0,1] from min-max normalised trajectory return.')
    # 2026-05-22 noise-augmentation knobs (iqlv_exp_rank_noise_aug variant)
    ap.add_argument('--noise_aug_K',      type=int,   default=4,
                    help='noise-aug: number of perturbed action copies per sample (0 disables aug)')
    ap.add_argument('--noise_aug_sigma',  type=float, default=0.1,
                    help='noise-aug: stdev of N(0,σ²I) action noise (action ∈ [-1,1])')
    ap.add_argument('--noise_aug_c',      type=float, default=5.0,
                    help='noise-aug: penalty coefficient on ‖ε‖ in adv_aug = adv - c·‖ε‖')
    ap.add_argument('--noise_aug_lambda', type=float, default=0.5,
                    help='noise-aug: weight of aux actor loss; 0 disables aug')
    ap.add_argument('--steps',          type=int,   default=1_000_000)
    ap.add_argument('--batch',          type=int,   default=256)
    ap.add_argument('--lr',             type=float, default=3e-4)
    ap.add_argument('--eval_interval',  type=int,   default=10_000)
    ap.add_argument('--eval_episodes',  type=int,   default=10)
    ap.add_argument('--seed',           type=int,   default=42)
    ap.add_argument('--device',         type=int,   default=None)
    ap.add_argument('--out_dir',        default='results_sna2c_v7')
    ap.add_argument('--ckpt_dir',       default=None)
    ap.add_argument('--ckpt_interval',  type=int, default=10_000)
    ap.add_argument('--last_pct',       type=float, default=0.10)
    # 2026-05-14 capacity sweep CLI flags. 默认都是 -1 = "不覆盖，用 default preset 原值".
    ap.add_argument('--actor_h',        type=int,   default=-1,
                    help='actor hidden width override (default -1 = use preset value 256)')
    ap.add_argument('--critic_h',       type=int,   default=-1,
                    help='critic V/Q hidden width override (default -1 = use preset value 256)')
    ap.add_argument('--actor_depth',    type=int,   default=-1,
                    help='actor hidden layer depth override (default -1 = use preset value 2)')
    ap.add_argument('--critic_depth',   type=int,   default=-1,
                    help='critic V/Q hidden layer depth override (default -1 = use preset value 2)')
    ap.add_argument('--preset_tag',     type=str,   default='default',
                    help='label written into ckpt + run_name for capacity ablation tracking')
    args = ap.parse_args()
    train(args)
