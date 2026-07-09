#!/usr/bin/env python3
"""
Shared agent & network definitions for offline RL experiments.
This is the SINGLE SOURCE OF TRUTH for all algorithms.
All training scripts should import from here instead of defining their own copies.

Usage:
    from agents import (
        DEVICE, tt, tb,
        set_network_preset,
        Actor, Critic, QNet,
        AsymREAgent, AWRAgent, IQLAgent, DRPOQAgent,
        IQLPosFiltAgent, IQLPosFiltLinearAgent, DRPOLinearAgent,
        ScaledNegA2CAgent, BPPOAgent, DRPOEXPAgent,
        SNA2C_IQLV_DistAgent,
    )

Network policy:
    All algorithms use separate Actor + Critic networks.
    Actor outputs tanh(mu) to keep actions in [-1, 1].
    There is NO shared-backbone ActorCritic class anymore
    (removed on 2026-04-24 to unify the action squashing policy
     across AWR / SNA2C and the rest of the algorithms).
"""

import numpy as np
import copy, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

_nt = int(os.environ.get('OMP_NUM_THREADS', 2))
torch.set_num_threads(_nt)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tt(x):
    if isinstance(x, torch.Tensor): return x.to(DEVICE)
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

def tb(x):
    if isinstance(x, torch.Tensor): return x.to(DEVICE)
    return torch.as_tensor(x, dtype=torch.bool, device=DEVICE)

# ── Network Presets ──────────────────────────────────────────────────────────
# 2026-05-14 (capacity sweep): 新增 'actor_depth' / 'q_depth' 字段。
#   * 'v_depth'      : Critic (V) 隐藏层数（已存在）
#   * 'q_depth'      : QNet 隐藏层数（新增，默认 2 = 旧硬编码值）
#   * 'actor_depth'  : Actor 隐藏层数（新增，默认 2 = 旧硬编码值）
# 默认值与改动前完全等价（actor=2 hidden, q=2 hidden）；
# 新 preset 可显式覆盖以做 capacity sweep（例如 deep4 / large）。
NETWORK_PRESETS = {
    'default': {
        'actor_h': 256, 'v_h': 256, 'q_h': 256, 'v_depth': 2,
        'actor_depth': 2, 'q_depth': 2,
        'log_std_mode': 'independent', 'output_relu': False,
        'init_gain': np.sqrt(2), 'log_std_min': -5.0, 'log_std_max': 2.0,
    },
    'bppo': {
        'actor_h': 1024, 'v_h': 512, 'q_h': 1024, 'v_depth': 3,
        'actor_depth': 2, 'q_depth': 2,
        'log_std_mode': 'state_dependent', 'output_relu': True,
        'init_gain': 1.0, 'log_std_min': -5.0, 'log_std_max': 0.0,
    },
}
_PRESET = NETWORK_PRESETS['default']

def set_network_preset(name):
    global _PRESET
    _PRESET = NETWORK_PRESETS[name]
    print(f"  Network preset: '{name}' -> {_PRESET}", flush=True)

def soft_clamp(x, low=-5.0, high=0.0):
    return low + 0.5 * (high - low) * (x + 1)

def _apply_ortho(module, gain=None):
    g = gain if gain is not None else _PRESET['init_gain']
    if g == 1.0:
        for p in module.parameters():
            if len(p.size()) >= 2: nn.init.orthogonal_(p)
    else:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=g)
                if m.bias is not None: nn.init.zeros_(m.bias)

# ── Networks ─────────────────────────────────────────────────────────────────
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

class Actor(nn.Module):
    def __init__(self, sdim, adim, h=None):
        super().__init__()
        p = _PRESET
        h = h if h is not None else p['actor_h']
        # 2026-05-14: actor_depth = 隐藏层数（连接 ReLU 的内部层数），默认 2 与历史一致
        depth = p.get('actor_depth', 2)
        self._log_std_mode = p['log_std_mode']
        self._log_std_min = p['log_std_min']
        self._log_std_max = p['log_std_max']
        if self._log_std_mode == 'state_dependent':
            # 输入层 + (depth-1) 个隐藏 + 输出层 (logit) + Tanh
            layers = [nn.Linear(sdim, h), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(h, h), nn.ReLU()]
            layers += [nn.Linear(h, 2 * adim), nn.Tanh()]
            self.net = nn.Sequential(*layers)
            self._adim = adim; _apply_ortho(self.net)
        else:
            # 输入层 + (depth-1) 个隐藏
            layers = [nn.Linear(sdim, h), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(h, h), nn.ReLU()]
            self.net = nn.Sequential(*layers)
            self.mu = nn.Linear(h, adim)
            self.log_std = nn.Parameter(torch.zeros(1, adim) * 1e-3)
            _apply_ortho(self.net); _apply_ortho(self.mu)
    def forward(self, s):
        if self._log_std_mode == 'state_dependent':
            out = self.net(s); mu, ls = out.chunk(2, dim=-1)
            return mu, soft_clamp(ls, self._log_std_min, self._log_std_max)
        f = self.net(s); mu = self.mu(f)
        ls = torch.clamp(self.log_std, self._log_std_min, self._log_std_max)
        return torch.tanh(mu), ls.expand_as(mu)

class Critic(nn.Module):
    def __init__(self, sdim, h=None):
        super().__init__()
        p = _PRESET; h = h if h is not None else p['v_h']
        layers = [nn.Linear(sdim,h), nn.ReLU()]
        for _ in range(p['v_depth']-1): layers += [nn.Linear(h,h), nn.ReLU()]
        layers.append(nn.Linear(h,1))
        if p['output_relu']: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers); _apply_ortho(self.net)
    def forward(self, s): return self.net(s)

class QNet(nn.Module):
    def __init__(self, sdim, adim, h=None):
        super().__init__()
        p = _PRESET; h = h if h is not None else p['q_h']
        # 2026-05-14: q_depth = 隐藏层数，默认 2 与历史一致
        depth = p.get('q_depth', 2)
        layers = [nn.Linear(sdim + adim, h), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.ReLU()]
        layers.append(nn.Linear(h, 1))
        if p['output_relu']: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers); _apply_ortho(self.net)
    def forward(self, s, a): return self.net(torch.cat([s,a],1))

# NOTE: The old shared-backbone `ActorCritic` class was removed
# on 2026-04-24. Every agent now composes a separate `Actor` and
# `Critic` (and QNet if needed). This keeps the tanh squashing on
# the policy output consistent across ALL algorithms.

# ── Agents ───────────────────────────────────────────────────────────────────────

class AsymREAgent:
    def __init__(self, sdim, adim, min_r, lr=3e-4):
        self.min_r = min_r
        self.net = Actor(sdim, adim).to(DEVICE)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.net(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.
    def update(self, s, a, r, ns, d, ep_ret):
        s, a, r = tt(s), tt(a), tt(r)
        adv = r - self.min_r
        mu, ls = self.net(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        loss = -(adv * dist.log_prob(a).sum(1)).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

class AWRAgent:
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, beta=0.05):
        self.gamma = gamma; self.beta = beta
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.
    def update(self, s, a, r, ns, d, ep_ret):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1); tgt = r + self.gamma * nv * (~d).float()
            cv = self.critic(s).squeeze(-1); adv = tgt - cv
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        w = torch.exp(torch.clamp(adv / self.beta, -10, 10)); w = w / (w.sum() + 1e-8)
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        pl = -(w * dist.log_prob(a).sum(1)).mean()
        v = self.critic(s).squeeze(-1)
        vl = nn.SmoothL1Loss()(v, tgt)
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())

class IQLAgent:
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, tau=0.7, temperature=3.0, total_steps=1_000_000):
        self.gamma=gamma; self.tau=tau; self.temperature=temperature
        self.actor=Actor(sdim,adim).to(DEVICE)
        self.q1=QNet(sdim,adim).to(DEVICE); self.q2=QNet(sdim,adim).to(DEVICE)
        self.tq1=copy.deepcopy(self.q1); self.tq2=copy.deepcopy(self.q2)
        self.v=Critic(sdim).to(DEVICE)
        self.a_opt=optim.Adam(self.actor.parameters(),lr=lr)
        self.a_sched=optim.lr_scheduler.CosineAnnealingLR(self.a_opt,T_max=total_steps)
        self.q_opt=optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.v_opt=optim.Adam(self.v.parameters(),lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu,_=self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.
    def _polyak(self):
        for p,tp in zip(self.q1.parameters(),self.tq1.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
        for p,tp in zip(self.q2.parameters(),self.tq2.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
    def update(self, s, a, r, ns, d, ep_ret):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        with torch.no_grad(): tq=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))
        ve=tq-self.v(s).squeeze(-1); wv=torch.where(ve>0,self.tau,1-self.tau)
        vl=(wv*ve**2).mean(); self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
        with torch.no_grad():
            adv=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))-self.v(s).squeeze(-1)
            w=torch.exp(torch.clamp(adv*self.temperature,max=10.)); w=torch.clamp(w,max=100.)
        mu,ls=self.actor(s); dist=torch.distributions.Normal(mu,ls.exp())
        al=-(w*dist.log_prob(a).sum(1)).mean()
        self.a_opt.zero_grad(); al.backward(); self.a_opt.step(); self.a_sched.step()
        with torch.no_grad(): qt=r+self.gamma*self.v(ns).squeeze(-1)*(~d).float()
        ql=((self.q1(s,a).squeeze(-1)-qt)**2+(self.q2(s,a).squeeze(-1)-qt)**2).mean()
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step(); self._polyak()
        return al.item()

class DRPOQAgent:
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, temperature=3.0, tau=0.5, init_p=0.5, total_steps=1_000_000, tgt_ratio=0.1, p_min=0.05, w_clip_max=100.0):
        self.gamma=gamma; self.temperature=temperature; self.tau=tau
        self.actor=Actor(sdim,adim).to(DEVICE)
        self.q1=QNet(sdim,adim).to(DEVICE); self.q2=QNet(sdim,adim).to(DEVICE)
        self.tq1=copy.deepcopy(self.q1); self.tq2=copy.deepcopy(self.q2)
        self.v=Critic(sdim).to(DEVICE)
        self.a_opt=optim.Adam(self.actor.parameters(),lr=lr)
        self.a_sched=optim.lr_scheduler.CosineAnnealingLR(self.a_opt,T_max=total_steps)
        self.q_opt=optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.v_opt=optim.Adam(self.v.parameters(),lr=lr)
        self.p=init_p; self.tgt_ratio=tgt_ratio
        self.p_min=p_min; self.w_clip_max=w_clip_max
    @torch.no_grad()
    def get_action(self, s):
        mu,_=self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.
    def _polyak(self):
        for p,tp in zip(self.q1.parameters(),self.tq1.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
        for p,tp in zip(self.q2.parameters(),self.tq2.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
    def update(self, s, a, r, ns, d, ep_ret):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        with torch.no_grad(): tq=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))
        ve=tq-self.v(s).squeeze(-1); wv=torch.where(ve>0,self.tau,1-self.tau)
        vl=(wv*ve**2).mean(); self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
        with torch.no_grad():
            adv_full=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))-self.v(s).squeeze(-1)
            k=max(2,int(s.shape[0]*self.p)); top_adv,idx=torch.topk(adv_full,k)
            s_t,a_t=s[idx],a[idx]
            tgt_std=(adv_full.std()+1e-8)*self.tgt_ratio
            if (top_adv.std()+1e-8)<tgt_std: self.p=min(1.0,self.p*1.02)
            else: self.p=max(self.p_min,self.p*0.98)
            w=torch.exp(torch.clamp(top_adv*self.temperature,max=10.)); w=torch.clamp(w,max=self.w_clip_max)
        mu,ls=self.actor(s_t); dist=torch.distributions.Normal(mu,ls.exp())
        al=-(w*dist.log_prob(a_t).sum(1)).mean()
        self.a_opt.zero_grad(); al.backward(); self.a_opt.step(); self.a_sched.step()
        with torch.no_grad(): qt=r+self.gamma*self.v(ns).squeeze(-1)*(~d).float()
        ql=((self.q1(s,a).squeeze(-1)-qt)**2+(self.q2(s,a).squeeze(-1)-qt)**2).mean()
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step(); self._polyak()
        return al.item()

class IQLPosFiltAgent:
    """IQL + positive-advantage filtering."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, tau=0.7, temperature=3.0, total_steps=1_000_000):
        self.gamma=gamma; self.tau=tau; self.temperature=temperature
        self.actor=Actor(sdim,adim).to(DEVICE)
        self.q1=QNet(sdim,adim).to(DEVICE); self.q2=QNet(sdim,adim).to(DEVICE)
        self.tq1=copy.deepcopy(self.q1); self.tq2=copy.deepcopy(self.q2)
        self.v=Critic(sdim).to(DEVICE)
        self.a_opt=optim.Adam(self.actor.parameters(),lr=lr)
        self.a_sched=optim.lr_scheduler.CosineAnnealingLR(self.a_opt,T_max=total_steps)
        self.q_opt=optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.v_opt=optim.Adam(self.v.parameters(),lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu,_=self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.
    def _polyak(self):
        for p,tp in zip(self.q1.parameters(),self.tq1.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
        for p,tp in zip(self.q2.parameters(),self.tq2.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
    def update(self, s, a, r, ns, d, ep_ret):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        with torch.no_grad(): tq=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))
        ve=tq-self.v(s).squeeze(-1); wv=torch.where(ve>0,self.tau,1-self.tau)
        vl=(wv*ve**2).mean(); self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
        with torch.no_grad():
            adv=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))-self.v(s).squeeze(-1)
            mask=adv>0
        if mask.sum()>0:
            s_pos,a_pos,adv_pos=s[mask],a[mask],adv[mask]
            with torch.no_grad(): w=torch.clamp(torch.exp(torch.clamp(adv_pos*self.temperature,max=10.)),max=100.)
            mu,ls=self.actor(s_pos); dist=torch.distributions.Normal(mu,ls.exp())
            al=-(w*dist.log_prob(a_pos).sum(1)).mean()
            self.a_opt.zero_grad(); al.backward(); self.a_opt.step()
        self.a_sched.step()
        with torch.no_grad(): qt=r+self.gamma*self.v(ns).squeeze(-1)*(~d).float()
        ql=((self.q1(s,a).squeeze(-1)-qt)**2+(self.q2(s,a).squeeze(-1)-qt)**2).mean()
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step(); self._polyak()
        return 0.

class IQLPosFiltLinearAgent:
    """IQL + positive-advantage filtering + linear weighting."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, tau=0.7, temperature=3.0, total_steps=1_000_000):
        self.gamma=gamma; self.tau=tau; self.temperature=temperature
        self.actor=Actor(sdim,adim).to(DEVICE)
        self.q1=QNet(sdim,adim).to(DEVICE); self.q2=QNet(sdim,adim).to(DEVICE)
        self.tq1=copy.deepcopy(self.q1); self.tq2=copy.deepcopy(self.q2)
        self.v=Critic(sdim).to(DEVICE)
        self.a_opt=optim.Adam(self.actor.parameters(),lr=lr)
        self.a_sched=optim.lr_scheduler.CosineAnnealingLR(self.a_opt,T_max=total_steps)
        self.q_opt=optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.v_opt=optim.Adam(self.v.parameters(),lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu,_=self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.
    def _polyak(self):
        for p,tp in zip(self.q1.parameters(),self.tq1.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
        for p,tp in zip(self.q2.parameters(),self.tq2.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
    def update(self, s, a, r, ns, d, ep_ret):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        with torch.no_grad(): tq=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))
        ve=tq-self.v(s).squeeze(-1); wv=torch.where(ve>0,self.tau,1-self.tau)
        vl=(wv*ve**2).mean(); self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
        with torch.no_grad():
            adv=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))-self.v(s).squeeze(-1)
            mask=adv>0
        if mask.sum()>0:
            s_pos,a_pos,adv_pos=s[mask],a[mask],adv[mask]
            mu,ls=self.actor(s_pos); dist=torch.distributions.Normal(mu,ls.exp())
            al=-(adv_pos.detach()*dist.log_prob(a_pos).sum(1)).mean()
            self.a_opt.zero_grad(); al.backward(); self.a_opt.step()
        self.a_sched.step()
        with torch.no_grad(): qt=r+self.gamma*self.v(ns).squeeze(-1)*(~d).float()
        ql=((self.q1(s,a).squeeze(-1)-qt)**2+(self.q2(s,a).squeeze(-1)-qt)**2).mean()
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step(); self._polyak()
        return 0.

class DRPOLinearAgent:
    """DRPO-Q with linear advantage weighting."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, temperature=3.0, tau=0.5, init_p=0.5, total_steps=1_000_000, tgt_ratio=0.1):
        self.gamma=gamma; self.temperature=temperature; self.tau=tau
        self.actor=Actor(sdim,adim).to(DEVICE)
        self.q1=QNet(sdim,adim).to(DEVICE); self.q2=QNet(sdim,adim).to(DEVICE)
        self.tq1=copy.deepcopy(self.q1); self.tq2=copy.deepcopy(self.q2)
        self.v=Critic(sdim).to(DEVICE)
        self.a_opt=optim.Adam(self.actor.parameters(),lr=lr)
        self.a_sched=optim.lr_scheduler.CosineAnnealingLR(self.a_opt,T_max=total_steps)
        self.q_opt=optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.v_opt=optim.Adam(self.v.parameters(),lr=lr)
        self.p=init_p; self.tgt_ratio=tgt_ratio
    @torch.no_grad()
    def get_action(self, s):
        mu,_=self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.
    def _polyak(self):
        for p,tp in zip(self.q1.parameters(),self.tq1.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
        for p,tp in zip(self.q2.parameters(),self.tq2.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
    def update(self, s, a, r, ns, d, ep_ret):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        with torch.no_grad(): tq=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))
        ve=tq-self.v(s).squeeze(-1); wv=torch.where(ve>0,self.tau,1-self.tau)
        vl=(wv*ve**2).mean(); self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
        with torch.no_grad():
            adv_full=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))-self.v(s).squeeze(-1)
            k=max(2,int(s.shape[0]*self.p)); top_adv,idx=torch.topk(adv_full,k)
            s_t,a_t=s[idx],a[idx]
            tgt_std=(adv_full.std()+1e-8)*self.tgt_ratio
            if (top_adv.std()+1e-8)<tgt_std: self.p=min(1.0,self.p*1.02)
            else: self.p=max(0.05,self.p*0.98)
        mu,ls=self.actor(s_t); dist=torch.distributions.Normal(mu,ls.exp())
        al=-(top_adv.detach()*dist.log_prob(a_t).sum(1)).mean()
        self.a_opt.zero_grad(); al.backward(); self.a_opt.step(); self.a_sched.step()
        with torch.no_grad(): qt=r+self.gamma*self.v(ns).squeeze(-1)*(~d).float()
        ql=((self.q1(s,a).squeeze(-1)-qt)**2+(self.q2(s,a).squeeze(-1)-qt)**2).mean()
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step(); self._polyak()
        return al.item()

# ── SNA2C Base + Four Variants (2026-06-05: merged into single skeleton) ─────
# All four agent classes share the same Actor+Critic architecture and
# training loop. They differ ONLY in how negative advantages are reweighted:
#   base   (ScaledNegA2CAgent)   — adv = where(adv<0, adv*alpha, adv)
#   ptrunc (SNA2C_PTruncAgent)  — keep only worst p% of negatives
#   negup  (SNA2C_NegUpAgent)   — weight ∝ |adv| (worse=heavier)
#   negdown(SNA2C_NegDownAgent) — weight ∝ 1/|adv| (worse=lighter)
#
# Common skeleton extracted into _SNA2CBase; each class is now a thin
# wrapper that only implements _transform_adv(). Public API unchanged.

class _SNA2CBase:
    """Shared skeleton for Actor+Critic-based SNA2C family."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11):
        self.gamma=gamma; self.alpha=alpha
        self.actor=Actor(sdim,adim).to(DEVICE)
        self.critic=Critic(sdim).to(DEVICE)
        self.a_opt=optim.Adam(self.actor.parameters(),lr=lr)
        self.c_opt=optim.Adam(self.critic.parameters(),lr=lr)

    @torch.no_grad()
    def get_action(self, s):
        mu,_=self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0

    def _transform_adv(self, adv):
        """Override in subclasses. Returns transformed adv tensor (same shape)."""
        raise NotImplementedError

    def update(self, s, a, r, ns, d, ep_ret=None):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        with torch.no_grad():
            nv=self.critic(ns).squeeze(-1); tgt=r+self.gamma*nv*(~d).float()
        v=self.critic(s).squeeze(-1)
        adv = tgt - v.detach()
        new_adv = self._transform_adv(adv)
        mu,ls=self.actor(s)
        dist=torch.distributions.Normal(mu,ls.exp())
        lp=dist.log_prob(a).sum(dim=-1)
        pl=-(lp*new_adv).mean()
        vl=nn.MSELoss()(v,tgt)
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())


class ScaledNegA2CAgent(_SNA2CBase):
    """SNA2C base: adv = where(adv<0, adv*alpha, adv)."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.5):
        super().__init__(sdim, adim, lr=lr, gamma=gamma, alpha=alpha)

    def _transform_adv(self, adv):
        return torch.where(adv < 0, adv * self.alpha, adv)


class SNA2C_PTruncAgent(_SNA2CBase):
    """SNA2C ptrunc: only keep worst p% of negatives, rest zeroed."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11, p=0.5):
        super().__init__(sdim, adim, lr=lr, gamma=gamma, alpha=alpha)
        self.p=p

    def _transform_adv(self, adv):
        neg_mask = adv < 0
        if neg_mask.any():
            neg_adv = adv[neg_mask]
            k = max(1, int(neg_adv.numel() * self.p))
            thresh = torch.kthvalue(neg_adv, k).values
            keep = neg_mask & (adv <= thresh)
            return torch.where(adv >= 0, adv,
                              torch.where(keep, adv * self.alpha, torch.zeros_like(adv)))
        return adv


class SNA2C_NegUpAgent(_SNA2CBase):
    """SNA2C negup: negative weight ∝ |adv| (worse samples punished more)."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11, shape='linear'):
        assert shape in ('linear','square','rank'), f"bad shape={shape}"
        super().__init__(sdim, adim, lr=lr, gamma=gamma, alpha=alpha)
        self.shape=shape

    def _transform_adv(self, adv):
        neg_mask = adv < 0
        new_adv = adv.clone()
        if neg_mask.any():
            with torch.no_grad():
                w = self._neg_weights(adv[neg_mask])
            new_adv[neg_mask] = adv[neg_mask] * w
        return new_adv

    def _neg_weights(self, neg_adv):
        mag = neg_adv.abs()
        if self.shape == 'linear':
            m = mag.mean() + 1e-8
            w = self.alpha * (mag / m)
        elif self.shape == 'square':
            m = mag.mean() + 1e-8
            w = self.alpha * (mag / m) ** 2
        else:  # rank
            order = torch.argsort(mag)
            ranks = torch.empty_like(mag); ranks[order] = torch.linspace(
                0.0, 1.0, steps=mag.numel(), device=mag.device, dtype=mag.dtype)
            w = self.alpha * 2.0 * ranks
        return w


class SNA2C_NegDownAgent(_SNA2CBase):
    """SNA2C negdown: negative weight ∝ 1/|adv| (worse samples punished less)."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11, shape='inv', temp=1.0):
        assert shape in ('inv','neg_rank','exp','exp2','exp2_noNorm'), f"bad shape={shape}"
        super().__init__(sdim, adim, lr=lr, gamma=gamma, alpha=alpha)
        self.shape=shape; self.temp=temp

    def _transform_adv(self, adv):
        neg_mask = adv < 0
        new_adv = adv.clone()
        if neg_mask.any():
            with torch.no_grad():
                w = self._neg_weights(adv[neg_mask])
            new_adv[neg_mask] = adv[neg_mask] * w
        return new_adv

    def _neg_weights(self, neg_adv):
        mag = neg_adv.abs()
        if self.shape == 'inv':
            inv = 1.0 / (mag + 1e-6)
            w = self.alpha * (inv / (inv.mean() + 1e-8))
        elif self.shape == 'neg_rank':
            order = torch.argsort(mag)
            ranks = torch.empty_like(mag); ranks[order] = torch.linspace(
                0.0, 1.0, steps=mag.numel(), device=mag.device, dtype=mag.dtype)
            w = self.alpha * 2.0 * (1.0 - ranks)
        elif self.shape == 'exp2':
            raw = torch.exp(torch.clamp(neg_adv * self.temp, min=-20.0))
            w = self.alpha * (raw / (raw.mean() + 1e-8))
        elif self.shape == 'exp2_noNorm':
            w = self.alpha * torch.exp(torch.clamp(neg_adv * self.temp, min=-20.0))
        else:  # exp
            scale = mag.mean() + 1e-8
            raw = torch.exp(-mag / scale)
            w = self.alpha * (raw / (raw.mean() + 1e-8))
        return w


class DRPOEXPAgent:
    """DRPO-EXP: DRPO-Q + exp(adv/beta) weighting on positive samples."""
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, temperature=3.0, tau=0.5, init_p=0.5, beta=30.0, total_steps=1_000_000):
        self.gamma=gamma; self.temperature=temperature; self.tau=tau; self.beta=beta
        self.actor=Actor(sdim,adim).to(DEVICE)
        self.q1=QNet(sdim,adim).to(DEVICE); self.q2=QNet(sdim,adim).to(DEVICE)
        self.tq1=copy.deepcopy(self.q1); self.tq2=copy.deepcopy(self.q2)
        self.v=Critic(sdim).to(DEVICE)
        self.a_opt=optim.Adam(self.actor.parameters(),lr=lr)
        self.a_sched=optim.lr_scheduler.CosineAnnealingLR(self.a_opt,T_max=total_steps)
        self.q_opt=optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.v_opt=optim.Adam(self.v.parameters(),lr=lr)
        self.p=init_p; self.tgt_ratio=0.1
    @torch.no_grad()
    def get_action(self, s):
        mu,_=self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.
    def _polyak(self):
        for p,tp in zip(self.q1.parameters(),self.tq1.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
        for p,tp in zip(self.q2.parameters(),self.tq2.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
    def update(self, s, a, r, ns, d, ep_ret):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        with torch.no_grad(): tq=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))
        ve=tq-self.v(s).squeeze(-1); wv=torch.where(ve>0,self.tau,1-self.tau)
        vl=(wv*ve**2).mean(); self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
        with torch.no_grad():
            adv_full=torch.min(self.tq1(s,a).squeeze(-1),self.tq2(s,a).squeeze(-1))-self.v(s).squeeze(-1)
            k=max(2,int(s.shape[0]*self.p)); top_adv,idx=torch.topk(adv_full,k)
            s_t,a_t=s[idx],a[idx]
            tgt_std=(adv_full.std()+1e-8)*self.tgt_ratio
            if (top_adv.std()+1e-8)<tgt_std: self.p=min(1.0,self.p*1.02)
            else: self.p=max(0.05,self.p*0.98)
            w=torch.exp(torch.clamp(top_adv/self.beta,-10,10)); w=w/(w.sum()+1e-8)
        mu,ls=self.actor(s_t); dist=torch.distributions.Normal(mu,ls.exp())
        al=-(w*dist.log_prob(a_t).sum(1)).mean()
        self.a_opt.zero_grad(); al.backward(); self.a_opt.step(); self.a_sched.step()
        with torch.no_grad(): qt=r+self.gamma*self.v(ns).squeeze(-1)*(~d).float()
        ql=((self.q1(s,a).squeeze(-1)-qt)**2+(self.q2(s,a).squeeze(-1)-qt)**2).mean()
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step(); self._polyak()
        return al.item()

class BPPOAgent:
    """BPPO: VQ -> BC -> PPO-clip -> Frozen.

    Two scheduling modes:
      (a) Legacy (default for backward compatibility): VQ = 40% of total_steps,
          BC = 10% of total_steps, PPO = BPPO_STEPS (=1000) steps, rest Frozen.
      (b) Explicit fixed-step schedule: pass vq_steps / bc_steps / bppo_steps
          directly.  total_steps is then only used for bookkeeping / lr schedule
          and the BPPO pretraining becomes INDEPENDENT of the arena horizon
          (recommended when comparing against other algorithms in an arena
          setting — pretraining finishes early and BPPO freezes, others keep
          training on their own schedule).
    """
    VQ_FRAC=0.4; BC_FRAC=0.5; BPPO_STEPS=1000
    def __init__(self, sdim, adim, lr=1e-4, gamma=0.99, total_steps=1_000_000,
                 clip_ratio=0.25, entropy_weight=0.0, decay=0.96, omega=0.9,
                 vq_steps=None, bc_steps=None, bppo_steps=None,
                 # ── Alignment-mode options (all default to legacy behavior) ──
                 old_policy_update_mode='loss',   # 'loss' (legacy) | 'external'
                 vq_mode='joint',                 # 'joint' (legacy) | 'separate'
                 grad_clip_actor=0.5,             # 0.5 (legacy) | None = disable
                 ):
        self.gamma=gamma; self.total_steps=total_steps; self.sdim=sdim; self.adim=adim
        # Store alignment-mode flags
        assert old_policy_update_mode in ('loss','external'), f"bad old_policy_update_mode={old_policy_update_mode}"
        assert vq_mode in ('joint','separate'), f"bad vq_mode={vq_mode}"
        self.old_policy_update_mode = old_policy_update_mode
        self.vq_mode = vq_mode
        self.grad_clip_actor = grad_clip_actor
        # Explicit schedule wins over fractional one when any *_steps given
        if vq_steps is not None or bc_steps is not None or bppo_steps is not None:
            # Fall back to sane defaults for whatever was not provided
            vq_steps   = int(vq_steps)   if vq_steps   is not None else int(total_steps*self.VQ_FRAC)
            bc_steps   = int(bc_steps)   if bc_steps   is not None else int(total_steps*(self.BC_FRAC-self.VQ_FRAC))
            bppo_steps = int(bppo_steps) if bppo_steps is not None else self.BPPO_STEPS
            self.vq_end   = vq_steps
            self.bc_end   = vq_steps + bc_steps
            self.bppo_end = self.bc_end + bppo_steps
        else:
            # Legacy behaviour (kept identical to previous versions)
            self.vq_end=int(total_steps*self.VQ_FRAC); self.bc_end=int(total_steps*self.BC_FRAC)
            self.bppo_end=self.bc_end+self.BPPO_STEPS
        self.v=Critic(sdim).to(DEVICE); self.v_opt=optim.Adam(self.v.parameters(),lr=lr)
        self.q=QNet(sdim,adim).to(DEVICE); self.tq=copy.deepcopy(self.q)
        self.q_opt=optim.Adam(self.q.parameters(),lr=lr)
        self._q_update_count=0; self._target_update_freq=2; self._tau=0.005
        self.actor=Actor(sdim,adim).to(DEVICE)
        self.a_opt=optim.Adam(self.actor.parameters(),lr=lr)
        self.a_sched=optim.lr_scheduler.StepLR(self.a_opt,step_size=2,gamma=0.98)
        self.old_actor=copy.deepcopy(self.actor)
        self.clip_ratio=clip_ratio; self.entropy_weight=entropy_weight
        self.decay=decay; self.omega=omega
        self._current_clip=clip_ratio; self._step=0; self._phase='vq'
        self._best_bppo_loss=float('inf')
    @torch.no_grad()
    def get_action(self, s):
        mu,ls=self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).clamp(-1,1).cpu().numpy(), 0.
    def _polyak_q(self):
        for p,tp in zip(self.q.parameters(),self.tq.parameters()):
            tp.data.copy_(self._tau*p.data+(1-self._tau)*tp.data)
    def _weighted_advantage(self, adv):
        if self.omega==0.5: return adv
        w=torch.where(adv>0,torch.full_like(adv,self.omega),torch.full_like(adv,1-self.omega))
        return w*adv
    def update(self, s, a, r, ns, d, ep_ret):
        self._step+=1
        s_t,a_t,r_t,ns_t,d_t=tt(s),tt(a),tt(r),tt(ns),tb(d); ep_ret_t=tt(ep_ret)
        if self._step<=self.vq_end:
            if self.vq_mode == 'joint':
                # Legacy: V and Q trained simultaneously per step
                self._phase='vq'
                vl=F.mse_loss(self.v(s_t).squeeze(-1),ep_ret_t)
                self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
                with torch.no_grad(): qt=r_t+self.gamma*self.v(ns_t).squeeze(-1)*(~d_t).float()
                ql=F.mse_loss(self.q(s_t,a_t).squeeze(-1),qt)
                self.q_opt.zero_grad(); ql.backward(); self.q_opt.step()
                self._q_update_count+=1
                if self._q_update_count%self._target_update_freq==0: self._polyak_q()
                return ql.item()
            else:
                # Separate: V-only for first half of vq_end, then Q-only
                # Mirrors BPPO official: V 2M steps -> Q 2M steps
                half = self.vq_end // 2
                if self._step <= half:
                    if self._phase != 'v':
                        self._phase='v'; print(f"  [BPPO] V phase (step {self._step})",flush=True)
                    vl=F.mse_loss(self.v(s_t).squeeze(-1),ep_ret_t)
                    self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
                    return vl.item()
                else:
                    if self._phase != 'q':
                        self._phase='q'; print(f"  [BPPO] Q phase (step {self._step})",flush=True)
                    with torch.no_grad(): qt=r_t+self.gamma*self.v(ns_t).squeeze(-1)*(~d_t).float()
                    ql=F.mse_loss(self.q(s_t,a_t).squeeze(-1),qt)
                    self.q_opt.zero_grad(); ql.backward(); self.q_opt.step()
                    self._q_update_count+=1
                    if self._q_update_count%self._target_update_freq==0: self._polyak_q()
                    return ql.item()
        elif self._step<=self.bc_end:
            if self._phase!='bc':
                self._phase='bc'; print(f"  [BPPO] BC phase (step {self._step})",flush=True)
            mu,ls=self.actor(s_t); dist=torch.distributions.Normal(mu,ls.exp())
            bl=-dist.log_prob(a_t).sum(-1).mean()
            self.a_opt.zero_grad(); bl.backward(); self.a_opt.step()
            return bl.item()
        elif self._step<=self.bppo_end:
            if self._phase!='bppo':
                self._phase='bppo'; self.old_actor.load_state_dict(self.actor.state_dict())
                self._current_clip=self.clip_ratio
                self.a_sched=optim.lr_scheduler.StepLR(self.a_opt,step_size=2,gamma=0.98)
                print(f"  [BPPO] PPO phase (step {self._step})",flush=True)
            bppo_step=self._step-self.bc_end
            with torch.no_grad():
                old_mu,old_ls=self.old_actor(s_t); old_dist=torch.distributions.Normal(old_mu,old_ls.exp())
                a_new=old_dist.rsample(); old_lp=old_dist.log_prob(a_new).sum(-1,keepdim=True)
                adv=self.q(s_t,a_new).squeeze(-1)-self.v(s_t).squeeze(-1)
                adv=(adv-adv.mean())/(adv.std()+1e-10); adv=self._weighted_advantage(adv)
            new_mu,new_ls=self.actor(s_t); new_dist=torch.distributions.Normal(new_mu,new_ls.exp())
            new_lp=new_dist.log_prob(a_new).sum(-1,keepdim=True)
            ratio=(new_lp-old_lp).exp()
            if bppo_step<=200: self._current_clip*=self.decay
            l1=ratio*adv.unsqueeze(-1); l2=torch.clamp(ratio,1-self._current_clip,1+self._current_clip)*adv.unsqueeze(-1)
            ent=new_dist.entropy().sum(-1,keepdim=True)*self.entropy_weight
            bppo_loss=-(torch.min(l1,l2)+ent).mean()
            self.a_opt.zero_grad(); bppo_loss.backward()
            if self.grad_clip_actor is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_actor)
            self.a_opt.step()
            # Official BPPO: lr_scheduler only steps while clip is decaying (step<=200)
            if bppo_step<=200:
                self.a_sched.step()
            # Old-policy update: 'loss' (legacy) updates when bppo_loss hits new low;
            # 'external' defers to trainer calling agent.set_old_policy() after env-eval.
            if self.old_policy_update_mode == 'loss':
                if bppo_loss.item()<self._best_bppo_loss:
                    self._best_bppo_loss=bppo_loss.item(); self.old_actor.load_state_dict(self.actor.state_dict())
            return bppo_loss.item()
        else:
            if self._phase!='frozen':
                self._phase='frozen'; print(f"  [BPPO] Frozen (step {self._step})",flush=True)
            return 0.0

    # ── External-control APIs (used by train_bppo_align.py when old_policy_update_mode='external') ──
    def set_old_policy(self):
        """Copy current actor weights into old_actor. Called by trainer after env-eval
        succeeds (score improved), mirroring BPPO official bppo.set_old_policy()."""
        self.old_actor.load_state_dict(self.actor.state_dict())

    def load_actor_state_dict(self, sd):
        """Load external actor weights into self.actor (e.g., best BC checkpoint
        before BPPO phase starts)."""
        self.actor.load_state_dict(sd)


class SNA2C_IQLVAgent:
    """SNA2C ②-B: IQL-style expectile V (V-raising via expectile regression).

    Only change vs. base SNA2C: V loss is expectile (τ > 0.5) instead of MSE.
    - target: tgt = r + gamma * V(ns)                   (same as base SNA2C, no Q-net)
    - v_residual  = tgt - V(s)
    - w_v          = tau  where v_residual > 0 else 1 - tau
    - V loss       = (w_v * v_residual**2).mean()        # asymmetric MSE
    When tau > 0.5, V is pulled toward the right-tail of the Bellman target
    distribution, effectively raising V -> more samples have adv<0 with cleaner
    quality separation among positive-adv samples.

    Actor loss: identical to base SNA2C.
        adv' = adv                if adv >= 0
             = adv * alpha        if adv <  0
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11, tau=0.7):
        assert 0.5 < tau < 1.0, f"expectile tau should be >0.5, got {tau}"
        self.gamma = gamma; self.alpha = alpha; self.tau = tau
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()
        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()
        # Actor loss: base SNA2C (no trunc, no re-weight)
        new_adv = torch.where(adv >= 0, adv, adv * self.alpha)
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        lp = dist.log_prob(a).sum(dim=-1)
        pl = -(lp * new_adv).mean()
        # V loss: IQL expectile
        ve = tgt - v
        wv = torch.where(ve > 0, self.tau, 1.0 - self.tau)
        vl = (wv * ve ** 2).mean()
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())


class SNA2C_IQLV_ExpAgent:
    """SNA2C ②-B × ①-D-exp2_noNorm combo:
      - Critic: IQL-expectile V (tau>0.5 pushes V up so more samples have adv<0)
      - Actor:  positive samples untouched; negative samples down-weighted by
                exp2_noNorm schedule: w_neg = alpha * exp(adv * T)  (adv<0 => w∈(0,α])

    Hypothesis (user 2026-05-04): raising V (IQLv tau>0.5) produces more negative
    samples; if we keep alpha fixed the effective negative pressure goes up and
    policy can blow up. Coupling with exp2_noNorm automatically down-weights the
    newly-created (barely-negative) samples (their adv is close to 0, so
    exp(adv*T)≈1 ⇒ weight ≈ α; but as adv gets more negative the weight decays
    exponentially, giving a continuous "soft PosFilt" effect).

    Knobs:
        tau ∈ (0.5, 1.0)  -- IQL expectile (0.5=MSE base; 0.7=IQL standard)
        alpha             -- negative scaling as in base SNA2C
        T                 -- exp2_noNorm temperature (T=0 ⇒ flat α; T→∞ ⇒ PosFilt)
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11,
                 tau=0.7, T=1.0):
        assert 0.5 <= tau < 1.0, f"expectile tau should be in [0.5, 1.0), got {tau}"
        self.gamma = gamma; self.alpha = alpha; self.tau = tau; self.T = T
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()
        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()
        # Actor: positives untouched; negatives get exp2_noNorm down-weight
        neg_mask = adv < 0
        new_adv = adv.clone()
        if neg_mask.any():
            with torch.no_grad():
                # exp2_noNorm: w = α · exp(adv·T), adv<0 ⇒ w∈(0,α]
                w = self.alpha * torch.exp(torch.clamp(adv[neg_mask] * self.T, min=-20.0))
            new_adv[neg_mask] = adv[neg_mask] * w
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        lp = dist.log_prob(a).sum(dim=-1)
        pl = -(lp * new_adv).mean()
        # V: IQL expectile (tau>0.5 raises V; tau=0.5 ⇒ MSE base)
        ve = tgt - v
        wv = torch.where(ve > 0, self.tau, 1.0 - self.tau)
        vl = (wv * ve ** 2).mean()
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())


class SNA2C_IQLV_ExpStdAgent:
    """Scale-invariant variant of SNA2C_IQLV_ExpAgent.

    Motivation (user 2026-05-04 16:00): The fixed-T version of exp2_noNorm
    suffers from adv-magnitude drift across datasets: hopper-MR has |adv|~0.5,
    while hopper-expert has |adv|~5-10, so the same T gives radically different
    decay rates. IQL / AWR normalize β by mean advantage magnitude or variance.
    Here we do the same: divide the exponent by the running std of negative
    advantages (computed per batch, detached). Then T becomes "how many σ of
    negative advantage produces an e-fold decay", which is physically
    consistent across datasets.

    Formula (V1 std-norm):
        s = std(adv[adv<0]).detach()       # batch-level, scalar, >0
        w = α · exp(adv · T / max(s, 1e-6))   (adv<0)

    Properties:
      * T is unit-free (in units of "std of negative adv").
      * When adv is perfectly Gaussian-centered, T=2 means a sample at -2σ
        gets weight α/e^2 ≈ 0.135α; at 5σ, α/e^5 ≈ 0.007α ≈ PosFilt limit.
      * Small-batch std noise is mild (batch=256, stable CLT).
      * std(.) falls back to 1.0 if no negative samples in batch.

    Knobs:
        tau ∈ [0.5, 1.0)  -- IQL expectile (same as ExpAgent)
        alpha             -- same meaning as SNA2C base alpha
        T                 -- scale-invariant temperature (0 ⇒ flat α; ∞ ⇒ PosFilt)

    Expected win vs ExpAgent: if the hypothesis ("T should be normalized") is
    correct, the optimal T across datasets should *collapse to one value*
    (e.g. T≈2-3 everywhere), and the peak of each dataset's sweep should rise
    over the fixed-T optimum. If not, batch-noise overhead hurts more than it
    helps and we'd see flat or worse numbers -- that itself falsifies the
    scale-invariance hypothesis cleanly.
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11,
                 tau=0.7, T=1.0):
        assert 0.5 <= tau < 1.0, f"expectile tau should be in [0.5, 1.0), got {tau}"
        self.gamma = gamma; self.alpha = alpha; self.tau = tau; self.T = T
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()
        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()
        # Actor: positives untouched; negatives get scale-invariant exp down-weight
        neg_mask = adv < 0
        new_adv = adv.clone()
        if neg_mask.any():
            with torch.no_grad():
                neg_adv = adv[neg_mask]
                # Use unbiased=False for numerical stability when |neg_adv|<2
                if neg_adv.numel() >= 2:
                    scale = neg_adv.std(unbiased=False).clamp_min(1e-6)
                else:
                    scale = torch.tensor(1.0, device=neg_adv.device)
                # exponent = adv*T/scale; adv<0 ⇒ negative exponent ⇒ w∈(0,α]
                w = self.alpha * torch.exp(torch.clamp(neg_adv * self.T / scale,
                                                       min=-20.0))
            new_adv[neg_mask] = adv[neg_mask] * w
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        lp = dist.log_prob(a).sum(dim=-1)
        pl = -(lp * new_adv).mean()
        # V: IQL expectile (same as ExpAgent)
        ve = tgt - v
        wv = torch.where(ve > 0, self.tau, 1.0 - self.tau)
        vl = (wv * ve ** 2).mean()
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())


class SNA2C_IQLV_ExpEmaStdAgent:
    """v5: EMA-smoothed scale-invariant exp down-weight.

    Motivation (user 2026-05-05 11:40): v3 std-norm (SNA2C_IQLV_ExpStdAgent) uses
    per-batch std(adv<0), which successfully collapsed the cross-dataset optimal
    T (20× → 1.7×) but introduced batch-level noise: batch_neg~128 → CV ~8%,
    which depressed each dataset's peak score by 1-6 pts vs v2 fixed-T. v5 keeps
    the σ-unit T semantics while replacing per-batch std with an EMA over recent
    steps, dramatically reducing scale noise with negligible bias.

    Algorithm:
        batch_std = std(adv[adv<0]).clamp_min(1e-6)
        ema_scale ← β · ema_scale + (1-β) · batch_std        (β=ema_decay, default 0.99)
        scale     = batch_std           if step < warmup_steps (default 5000)
                  = ema_scale           otherwise
        w = α · exp(adv · T / scale)    (adv<0)

    Rationale:
      * warmup avoids using a cold EMA (seeded from batch 0) in early training
        when adv distribution is still drifting wildly.
      * β=0.99 ≙ half-life ≈ 69 steps ≈ ½ of an eval interval → smooth enough
        but still adapts to slow drift of adv distribution through training.

    Expected: (a) cross-ds T collapse preserved (v3 ✓); (b) peak scores at least
    match v2 fixed-T (closing the 1-6 pt v3 gap); best case strictly dominates.

    Knobs:
        tau, alpha, T        -- same meaning as ExpStdAgent
        ema_decay ∈ (0,1)   -- 0.99 default; 0 ⇒ back to v3, 1 ⇒ frozen after warmup
        warmup_steps         -- 5000 default (batches); use batch_std during warmup
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11,
                 tau=0.7, T=1.0, ema_decay=0.99, warmup_steps=5000):
        assert 0.5 <= tau < 1.0, f"expectile tau should be in [0.5, 1.0), got {tau}"
        assert 0.0 < ema_decay < 1.0
        self.gamma = gamma; self.alpha = alpha; self.tau = tau; self.T = T
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.ema_scale = None
        self._step = 0
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()
        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()
        self._step += 1
        neg_mask = adv < 0
        new_adv = adv.clone()
        if neg_mask.any():
            with torch.no_grad():
                neg_adv = adv[neg_mask]
                if neg_adv.numel() >= 2:
                    batch_std = neg_adv.std(unbiased=False).clamp_min(1e-6)
                else:
                    batch_std = torch.tensor(1.0, device=neg_adv.device)
                # EMA update (always, so post-warmup switch is seamless)
                if self.ema_scale is None:
                    self.ema_scale = batch_std.detach()
                else:
                    self.ema_scale = (self.ema_decay * self.ema_scale +
                                      (1.0 - self.ema_decay) * batch_std.detach())
                scale = batch_std if self._step < self.warmup_steps else self.ema_scale
                w = self.alpha * torch.exp(torch.clamp(neg_adv * self.T / scale,
                                                       min=-20.0))
            new_adv[neg_mask] = adv[neg_mask] * w
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        lp = dist.log_prob(a).sum(dim=-1)
        pl = -(lp * new_adv).mean()
        ve = tgt - v
        wv = torch.where(ve > 0, self.tau, 1.0 - self.tau)
        vl = (wv * ve ** 2).mean()
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())


class SNA2C_IQLV_ExpRankAgent:
    """v6: Rank-based scale/shift-invariant exp down-weight.

    Motivation (user 2026-05-05 10:00): std-norm (v3/v5) only normalizes the
    scale of adv_neg distribution, not its mean shift. Two datasets with the
    same std but different mean of adv_neg get wildly different weights under
    v3 because the exponent `adv*T/σ` still depends on μ_neg.

    v6 solves BOTH at once: use only the within-batch rank of neg samples,
    not their values. Fully scale-AND-shift-invariant; natural continuous
    generalization of Ptrunc (hard percentile truncation).

    Algorithm:
        S = {i : adv_i < 0};  N = |S|
        order = argsort(adv_S, ASC)    # order[0] = worst (most negative)
        rank  = inverse permutation    # rank[i] ∈ [0, N-1]
        p     = rank / (N-1)           # [0, 1];  0=worst,  1=best-in-neg
        score = 1 - p                  # 0=best-in-neg, 1=worst
        w     = α · exp(-T · score)
            worst: w = α·exp(-T)    (e.g. T=5 → α·0.007, close to Ptrunc limit)
            best : w = α·exp(0) = α (unchanged)

    Properties:
        + Fully invariant to reward/adv rescaling and global shift.
        + T semantics universal across datasets and normalization conventions.
        + Continuous Ptrunc: as T→∞, only best-ranked neg samples keep weight.
        - Discretization noise from rank (~0.8% per step, batch_neg~128).
        - Discards magnitude info; may lose if adv has heavy-tailed signal.

    Expected: most stable optimal T across datasets; may strictly beat
    Ptrunc on MR (soft vs hard percentile) and match v5 on ME/expert.
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11,
                 tau=0.7, T=1.0):
        assert 0.5 <= tau < 1.0, f"expectile tau should be in [0.5, 1.0), got {tau}"
        self.gamma = gamma; self.alpha = alpha; self.tau = tau; self.T = T
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()
        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()
        neg_mask = adv < 0
        new_adv = adv.clone()
        if neg_mask.any():
            with torch.no_grad():
                neg_adv = adv[neg_mask]
                N = neg_adv.numel()
                if N >= 2:
                    order = neg_adv.argsort()
                    ranks = torch.empty_like(order)
                    ranks[order] = torch.arange(N, device=neg_adv.device)
                    p = ranks.float() / float(N - 1)
                    score = 1.0 - p
                    w = self.alpha * torch.exp(torch.clamp(-self.T * score,
                                                           min=-20.0))
                else:
                    w = torch.tensor([self.alpha], device=neg_adv.device)
            new_adv[neg_mask] = adv[neg_mask] * w
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        lp = dist.log_prob(a).sum(dim=-1)
        pl = -(lp * new_adv).mean()
        ve = tgt - v
        wv = torch.where(ve > 0, self.tau, 1.0 - self.tau)
        vl = (wv * ve ** 2).mean()
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())


class SNA2C_IQLV_ExpRank_NoiseAugAgent:
    """2026-05-22: ExpRank + Q-free anti-extrapolation via action-noise augmentation.

    Motivation (user 2026-05-22):
        SNA2C is unstable on narrow-mode ds (h-medium, hc-medium) — best-protocol
        h-ME can hit 90+ but mean only 60-70. Symptom = unstable, not "can't learn".
        Idea: for every transition (s, a), generate K perturbed actions
              a' = a + ε,  ε ~ N(0, σ²I)
        and give them an *artificial* penalty proportional to ‖ε‖ so they
        appear systematically *worse* than the data action.  This pulls the
        policy toward a (it acts as a local attractor), reducing the chance
        the policy is dragged toward neighbouring noisy actions.

        Compared to SQOG, this is fully Q-free: the penalty is hand-crafted
        from ε magnitude, no critic-side enforcement needed.  Generalises to
        LLM RLHF (s=prefix, a=completion, ‖ε‖=edit distance) where no Q exists.

    Algorithm:
        1. Compute ExpRank-reweighted `new_adv` exactly as parent class.
        2. Sample K noise copies per batch sample:
               eps ∼ N(0, σ²I),     a_aug = clip(a + eps, -1, 1)
        3. Penalised advantage:
               adv_aug = new_adv - c · ‖ε‖₂        (broadcast across K copies)
        4. Auxiliary actor loss:
               L_aug = -(adv_aug · log_prob(a_aug | s)).mean()
        5. Total actor loss:
               L_actor_total = L_actor + λ_aug · L_aug

    Hyper-params:
        K        : noise copies per sample          (default 4)
        sigma    : noise stdev (action ∈ [-1,1])    (default 0.1)
        c        : penalty coefficient on ‖ε‖       (default 5.0; sweep {0,1,3,10,30})
        lam_aug  : weight of aux actor loss         (default 0.5)

    Reproduces parent ExpRank exactly when K=0 OR lam_aug=0.
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11,
                 tau=0.7, T=1.0, K=4, sigma=0.1, c=5.0, lam_aug=0.5):
        assert 0.5 <= tau < 1.0, f"expectile tau should be in [0.5, 1.0), got {tau}"
        self.gamma = gamma; self.alpha = alpha; self.tau = tau; self.T = T
        self.K = int(K); self.sigma = float(sigma)
        self.c = float(c); self.lam_aug = float(lam_aug)
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()
        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()
        # Re-rank negatives identically to ExpRank parent.
        neg_mask = adv < 0
        new_adv = adv.clone()
        if neg_mask.any():
            with torch.no_grad():
                neg_adv = adv[neg_mask]
                N = neg_adv.numel()
                if N >= 2:
                    order = neg_adv.argsort()
                    ranks = torch.empty_like(order)
                    ranks[order] = torch.arange(N, device=neg_adv.device)
                    p = ranks.float() / float(N - 1)
                    score = 1.0 - p
                    w = self.alpha * torch.exp(torch.clamp(-self.T * score,
                                                           min=-20.0))
                else:
                    w = torch.tensor([self.alpha], device=neg_adv.device)
            new_adv[neg_mask] = adv[neg_mask] * w
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        lp = dist.log_prob(a).sum(dim=-1)
        pl_main = -(lp * new_adv).mean()

        # Noise-aug auxiliary loss.
        if self.K > 0 and self.lam_aug > 0.0:
            B, A = a.shape
            # eps shape: (B, K, A)
            eps = torch.randn(B, self.K, A, device=a.device,
                              dtype=a.dtype) * self.sigma
            a_aug = (a.unsqueeze(1) + eps).clamp(-1.0, 1.0)        # (B, K, A)
            eps_norm = eps.norm(dim=-1)                            # (B, K)
            # adv_aug penalised: broadcast new_adv over K copies.
            adv_aug = new_adv.detach().unsqueeze(1) - self.c * eps_norm  # (B, K)
            # log_prob of a_aug under current policy (per-sample, per-K, per-A → sum over A).
            mu_b   = mu.unsqueeze(1).expand(-1, self.K, -1)        # (B, K, A)
            ls_b   = ls.unsqueeze(1).expand(-1, self.K, -1)
            dist_b = torch.distributions.Normal(mu_b, ls_b.exp())
            lp_aug = dist_b.log_prob(a_aug).sum(dim=-1)            # (B, K)
            pl_aug = -(adv_aug * lp_aug).mean()
            pl = pl_main + self.lam_aug * pl_aug
        else:
            pl = pl_main

        ve = tgt - v
        wv = torch.where(ve > 0, self.tau, 1.0 - self.tau)
        vl = (wv * ve ** 2).mean()
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())


class SNA2C_IQLV_ExpRankBaselineAgent:
    """v7 (2026-05-19): ExpRank + global advantage baseline shift.

    Motivation (user 2026-05-19 19:30): negative-advantage gradient is what drives
    *generalisation*; positive-advantage gradient is the *constraint* that bounds
    the OOD push. Raising the bar at which a sample is judged "negative" (i.e.
    subtracting a constant baseline `b` from advantage) increases the fraction of
    samples that contribute to neg-grad, which deepens the signal for "push away
    from mediocre" without losing "anchor on best-in-batch" via positives.

    Algorithm (sole change vs ExpRank):
        adv_raw      = r + γ·V(s') - V(s)            # original advantage
        b            = quantile(rewards, p) of dataset (frozen, computed once
                       outside this Agent and passed to constructor)
        adv_shifted  = adv_raw - b
        pos_mask     = adv_raw > 0                   # ⟵ positive judged on raw,
                                                       not shifted; preserves the
                                                       "best-in-batch as anchor"
                                                       semantics of SNA2C.
        neg_mask     = adv_shifted < 0               # ⟵ baseline lifts the
                                                       judge-line, so MORE samples
                                                       enter neg.
        new_adv[pos_mask] = adv_raw[pos_mask]                             (anchor)
        new_adv[neg_mask] = adv_shifted[neg_mask] · w_rank(neg_subset)    (push)
        new_adv[else]     = 0                                             (skip)

    Properties vs ExpRank:
        + Smoothly raises baseline → more neg samples → stronger push.
        + Keeps positive-anchor logic untouched.
        + p ∈ {0.5, 0.7, 0.9}: 50%/30%/10% of dataset has reward ≥ b.
          Higher p ⇒ smaller b ⇒ closer to ExpRank.   p→1 ⇒ b→max(r) ⇒ all neg.
          (Note: this is reward-quantile, not adv-quantile, but they share the
          same coarse ordering and reward is constant across V iterations.)
        + Preserves rank-decay weights w (α·exp(-T·score)) on the neg subset.

    Args:
        baseline (float)  -- constant b subtracted from advantage when judging
                             and weighting negative samples.  Computed externally
                             from the buffer's reward distribution: e.g.
                             baseline = quantile(buffer_rewards, baseline_p).
        neg_lambda (float) -- 2026-05-19 user request: explicit relative weight
                             for the negative-side gradient.  λ = 1.0 reproduces
                             the original behavior (neg term enters at scale w);
                             λ < 1 down-weights the negative push (e.g. λ = 0.1
                             keeps positives as the dominant signal and treats
                             baseline-shifted negatives as a soft regulariser).
                             Set λ = 0 to fall back to "positives only".
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11,
                 tau=0.7, T=1.0, baseline=0.0, neg_lambda=1.0):
        assert 0.5 <= tau < 1.0, f"expectile tau should be in [0.5, 1.0), got {tau}"
        self.gamma = gamma; self.alpha = alpha; self.tau = tau; self.T = T
        self.baseline = float(baseline)
        self.neg_lambda = float(neg_lambda)
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()
        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()                    # raw advantage (used for pos)
        adv_shift = adv - self.baseline           # shifted advantage (used for neg judge+value)
        pos_mask = adv > 0                        # positive judged on RAW advantage (anchor)
        neg_mask = adv_shift < 0                  # negative judged on SHIFTED advantage
        new_adv = torch.zeros_like(adv)
        # ── pos branch: copy raw advantage straight through ──
        new_adv[pos_mask] = adv[pos_mask]
        # ── neg branch: rank-weighted shifted advantage ──
        if neg_mask.any():
            with torch.no_grad():
                neg_adv_shift = adv_shift[neg_mask]
                N = neg_adv_shift.numel()
                if N >= 2:
                    order = neg_adv_shift.argsort()      # asc: most-neg first
                    ranks = torch.empty_like(order)
                    ranks[order] = torch.arange(N, device=neg_adv_shift.device)
                    p = ranks.float() / float(N - 1)
                    score = 1.0 - p                       # 0=best-in-neg, 1=worst
                    w = self.alpha * torch.exp(torch.clamp(-self.T * score,
                                                           min=-20.0))
                else:
                    w = torch.tensor([self.alpha], device=neg_adv_shift.device)
            new_adv[neg_mask] = adv_shift[neg_mask] * w * self.neg_lambda
        # ── actor + critic update ──
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        lp = dist.log_prob(a).sum(dim=-1)
        pl = -(lp * new_adv).mean()
        ve = tgt - v
        wv = torch.where(ve > 0, self.tau, 1.0 - self.tau)
        vl = (wv * ve ** 2).mean()
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())


class IQLPosTopKAgent:
    """IQL + positive-advantage filtering tightened via top-k within the adv>0 subset.

    Difference from IQLPosFiltAgent:
        original: mask = adv > 0                               (all positive-adv samples)
        this one: mask = (adv > 0) AND (adv >= q-quantile(adv_pos))
                   where q ∈ [0, 1].  q=0 ⇒ same as original.
                                      q=0.5 ⇒ top 50% within positive subset.
                                      q=0.85 ⇒ top 15% within positive subset.

    This implements the "tighten actor filter above adv>0" direction discussed on
    2026-04-30. τ is fixed at the IQL-optimal 0.7; only the actor-side quantile q
    is swept. Rest of the critic / V / Q pipeline is IDENTICAL to IQLPosFiltAgent,
    so any delta vs baseline isolates the effect of the tightened actor filter.
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, tau=0.7, temperature=3.0,
                 q=0.5, total_steps=1_000_000):
        assert 0.0 <= q < 1.0, f"q must be in [0,1), got {q}"
        self.gamma = gamma; self.tau = tau; self.temperature = temperature; self.q = q
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.q1 = QNet(sdim, adim).to(DEVICE); self.q2 = QNet(sdim, adim).to(DEVICE)
        self.tq1 = copy.deepcopy(self.q1);       self.tq2 = copy.deepcopy(self.q2)
        self.v = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.a_sched = optim.lr_scheduler.CosineAnnealingLR(self.a_opt, T_max=total_steps)
        self.q_opt = optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def _polyak(self):
        for p,tp in zip(self.q1.parameters(), self.tq1.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
        for p,tp in zip(self.q2.parameters(), self.tq2.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
    def update(self, s, a, r, ns, d, ep_ret=None):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        # 1. V update — IQL standard expectile
        with torch.no_grad():
            tq = torch.min(self.tq1(s,a).squeeze(-1), self.tq2(s,a).squeeze(-1))
        ve = tq - self.v(s).squeeze(-1)
        wv = torch.where(ve > 0, self.tau, 1 - self.tau)
        vl = (wv * ve**2).mean()
        self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
        # 2. Actor update — tighten positive-adv filter via top-k quantile
        with torch.no_grad():
            adv = torch.min(self.tq1(s,a).squeeze(-1), self.tq2(s,a).squeeze(-1)) \
                  - self.v(s).squeeze(-1)
            pos_mask = adv > 0
            if pos_mask.sum() > 1 and self.q > 0:
                adv_pos = adv[pos_mask]
                thresh = torch.quantile(adv_pos, self.q)
                mask = pos_mask & (adv >= thresh)
            else:
                mask = pos_mask
        if mask.sum() > 0:
            s_p, a_p, adv_p = s[mask], a[mask], adv[mask]
            with torch.no_grad():
                w = torch.clamp(torch.exp(torch.clamp(adv_p * self.temperature, max=10.)), max=100.)
            mu, ls = self.actor(s_p)
            dist = torch.distributions.Normal(mu, ls.exp())
            al = -(w * dist.log_prob(a_p).sum(1)).mean()
            self.a_opt.zero_grad(); al.backward(); self.a_opt.step()
        self.a_sched.step()
        # 3. Q update — uses updated V as bootstrap target
        with torch.no_grad():
            qt = r + self.gamma * self.v(ns).squeeze(-1) * (~d).float()
        ql = ((self.q1(s,a).squeeze(-1) - qt)**2 + (self.q2(s,a).squeeze(-1) - qt)**2).mean()
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step(); self._polyak()
        return 0.
    """SNA2C ②-B×①-D-exp2_noNorm combo:
      - Critic: IQL-expectile V (tau>0.5 pushes V up so more samples have adv<0)
      - Actor:  positive samples untouched; negative samples down-weighted by
                exp2_noNorm schedule: w_neg = alpha * exp(adv * T)   (adv<0 => w∈(0,α])

    Hypothesis (user 2026-05-04): IQLv alone sharpens positive filtering (raising
    V -> fewer, higher-quality positive samples) but would blow up by inflating
    the effective alpha (more neg samples, same alpha => stronger neg pull).
    Coupling with exp2_noNorm solves this: the extra neg samples created by
    raising V get exponentially-smaller weights since their adv is less negative
    (they are the "new" neg samples just pushed below zero).

    Knobs:
        tau ∈ (0.5, 1.0)      -- expectile (0.5=MSE/base; 0.7=IQL standard)
        alpha               -- negative scaling, as in base SNA2C
        T                   -- exp2_noNorm temperature:
                                T=0 ⇒ w_neg=α (no decay) -- same as IQLv base
                                T→∞ ⇒ w_neg→0 for all neg -- IQL-PosFilt
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11,
                 tau=0.7, T=1.0):
        assert 0.5 < tau < 1.0, f"expectile tau should be >0.5, got {tau}"
        self.gamma = gamma; self.alpha = alpha; self.tau = tau; self.T = T
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()
        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()
        # Actor: positive samples keep adv; negatives get exp2_noNorm down-weight
        neg_mask = adv < 0
        new_adv = adv.clone()
        if neg_mask.any():
            with torch.no_grad():
                # exp2_noNorm: w = α · exp(adv·T), no renormalisation
                w = self.alpha * torch.exp(torch.clamp(adv[neg_mask] * self.T, min=-20.0))
            new_adv[neg_mask] = adv[neg_mask] * w
        mu, ls = self.actor(s)
        dist = torch.distributions.Normal(mu, ls.exp())
        lp = dist.log_prob(a).sum(dim=-1)
        pl = -(lp * new_adv).mean()
        # V: IQL expectile (asymmetric MSE, tau>0.5 raises V)
        ve = tgt - v
        wv = torch.where(ve > 0, self.tau, 1.0 - self.tau)
        vl = (wv * ve ** 2).mean()
        self.a_opt.zero_grad(); pl.backward(); self.a_opt.step()
        self.c_opt.zero_grad(); vl.backward(); self.c_opt.step()
        return (pl.item() + 0.5 * vl.item())
    """IQL + positive-advantage filtering tightened via top-k within the adv>0 subset.

    Difference from IQLPosFiltAgent:
        original: mask = adv > 0                               (all positive-adv samples)
        this one: mask = (adv > 0) AND (adv >= q-quantile(adv_pos))
                   where q ∈ [0, 1].  q=0 ⇒ same as original.
                                      q=0.5 ⇒ top 50% within positive subset.
                                      q=0.85 ⇒ top 15% within positive subset.

    This implements the "tighten actor filter above adv>0" direction discussed on
    2026-04-30. τ is fixed at the IQL-optimal 0.7; only the actor-side quantile q
    is swept. Rest of the critic / V / Q pipeline is IDENTICAL to IQLPosFiltAgent,
    so any delta vs baseline isolates the effect of the tightened actor filter.
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, tau=0.7, temperature=3.0,
                 q=0.5, total_steps=1_000_000):
        assert 0.0 <= q < 1.0, f"q must be in [0,1), got {q}"
        self.gamma = gamma; self.tau = tau; self.temperature = temperature; self.q = q
        self.actor = Actor(sdim, adim).to(DEVICE)
        self.q1 = QNet(sdim, adim).to(DEVICE); self.q2 = QNet(sdim, adim).to(DEVICE)
        self.tq1 = copy.deepcopy(self.q1);       self.tq2 = copy.deepcopy(self.q2)
        self.v = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.a_sched = optim.lr_scheduler.CosineAnnealingLR(self.a_opt, T_max=total_steps)
        self.q_opt = optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=lr)
        self.v_opt = optim.Adam(self.v.parameters(), lr=lr)
    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0
    def _polyak(self):
        for p,tp in zip(self.q1.parameters(), self.tq1.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
        for p,tp in zip(self.q2.parameters(), self.tq2.parameters()): tp.data.mul_(0.995).add_(0.005*p.data)
    def update(self, s, a, r, ns, d, ep_ret=None):
        s,a,r,ns,d = tt(s),tt(a),tt(r),tt(ns),tb(d)
        # 1. V update — IQL standard expectile
        with torch.no_grad():
            tq = torch.min(self.tq1(s,a).squeeze(-1), self.tq2(s,a).squeeze(-1))
        ve = tq - self.v(s).squeeze(-1)
        wv = torch.where(ve > 0, self.tau, 1 - self.tau)
        vl = (wv * ve**2).mean()
        self.v_opt.zero_grad(); vl.backward(); self.v_opt.step()
        # 2. Actor update — tighten positive-adv filter via top-k quantile
        with torch.no_grad():
            adv = torch.min(self.tq1(s,a).squeeze(-1), self.tq2(s,a).squeeze(-1)) \
                  - self.v(s).squeeze(-1)
            pos_mask = adv > 0
            if pos_mask.sum() > 1 and self.q > 0:
                adv_pos = adv[pos_mask]
                thresh = torch.quantile(adv_pos, self.q)
                mask = pos_mask & (adv >= thresh)
            else:
                mask = pos_mask
        if mask.sum() > 0:
            s_p, a_p, adv_p = s[mask], a[mask], adv[mask]
            with torch.no_grad():
                w = torch.clamp(torch.exp(torch.clamp(adv_p * self.temperature, max=10.)), max=100.)
            mu, ls = self.actor(s_p)
            dist = torch.distributions.Normal(mu, ls.exp())
            al = -(w * dist.log_prob(a_p).sum(1)).mean()
            self.a_opt.zero_grad(); al.backward(); self.a_opt.step()
        self.a_sched.step()
        # 3. Q update — uses updated V as bootstrap target
        with torch.no_grad():
            qt = r + self.gamma * self.v(ns).squeeze(-1) * (~d).float()
        ql = ((self.q1(s,a).squeeze(-1) - qt)**2 + (self.q2(s,a).squeeze(-1) - qt)**2).mean()
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step(); self._polyak()
        return 0.


class SNA2C_IQLV_DistAgent:
    """Distance-aware SNA2C / Soft-DRPO (distance-only V1).

    Critic:
        IQL-style expectile V, identical to SNA2C_IQLVAgent.

    Actor:
        positives:  new_adv = adv
        negatives:  new_adv = adv * alpha * distance_gate

        dist2 = mean_j [((a_j - mu_j) / std_j)^2]
        gate  = exp(-lambda_t * relu(dist2 - rho))

    Key properties:
        * Advantage is NOT replaced or exponentially reweighted.
        * mu/log_std used by the gate are detached.
        * update() keeps returning a scalar; diagnostics are in last_stats.
        * dist_decay=0 + warmup=0 exactly matches SNA2C_IQLVAgent (equivalence baseline).
    """
    def __init__(self, sdim, adim, lr=3e-4, gamma=0.99, alpha=0.11,
                 tau=0.7, dist_radius=2.0, dist_decay=0.5,
                 dist_warmup_steps=5000, gate_floor=0.0):
        assert 0.5 <= tau < 1.0, f"tau must be in [0.5, 1), got {tau}"
        assert alpha >= 0.0, f"alpha must be non-negative, got {alpha}"
        assert dist_radius >= 0.0
        assert dist_decay >= 0.0
        assert dist_warmup_steps >= 0
        assert 0.0 <= gate_floor < 1.0

        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.dist_radius = dist_radius
        self.dist_decay = dist_decay
        self.dist_warmup_steps = dist_warmup_steps
        self.gate_floor = gate_floor
        self._step = 0
        self.last_stats = {}

        self.actor = Actor(sdim, adim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def get_action(self, s):
        mu, _ = self.actor(tt(s).unsqueeze(0))
        return mu.squeeze(0).cpu().numpy(), 0.0

    @staticmethod
    def _normalized_mahalanobis(a, mu, log_std):
        """Per-action-dimension Mahalanobis distance.

        Under a correctly specified diagonal Gaussian, E[dist2] ~= 1.
        Gate computation must not create gradients through mu or std.
        """
        std = log_std.exp().clamp_min(1e-4)
        z = (a - mu) / std
        return z.square().mean(dim=-1)

    def _current_decay(self):
        if self.dist_warmup_steps == 0:
            return self.dist_decay
        progress = min(1.0, self._step / float(self.dist_warmup_steps))
        return self.dist_decay * progress

    def _distance_gate(self, dist2):
        decay = self._current_decay()
        far_excess = torch.relu(dist2 - self.dist_radius)
        exponent = torch.clamp(-decay * far_excess, min=-20.0, max=0.0)
        gate = torch.exp(exponent)
        if self.gate_floor > 0.0:
            gate = self.gate_floor + (1.0 - self.gate_floor) * gate
        return gate

    def update(self, s, a, r, ns, d, ep_ret=None):
        s, a, r, ns, d = tt(s), tt(a), tt(r), tt(ns), tb(d)
        self._step += 1

        # 1) One-step TD target (IQL-style)
        with torch.no_grad():
            nv = self.critic(ns).squeeze(-1)
            tgt = r + self.gamma * nv * (~d).float()

        v = self.critic(s).squeeze(-1)
        adv = tgt - v.detach()

        # 2) Actor forward: needed for log-prob AND policy distance
        mu, log_std = self.actor(s)

        # 3) Distance-only negative reliability gate
        neg_mask = adv < 0
        new_adv = adv.clone()

        with torch.no_grad():
            dist2_all = self._normalized_mahalanobis(
                a, mu.detach(), log_std.detach(),
            )

            if neg_mask.any():
                neg_dist2 = dist2_all[neg_mask]
                neg_gate = self._distance_gate(neg_dist2)
                neg_weight = self.alpha * neg_gate
                new_adv[neg_mask] = adv[neg_mask] * neg_weight
            else:
                neg_dist2 = torch.empty(0, device=adv.device)
                neg_gate = torch.empty(0, device=adv.device)

        # 4) Policy loss (standard policy gradient with weighted adv)
        dist = torch.distributions.Normal(mu, log_std.exp())
        log_prob = dist.log_prob(a).sum(dim=-1)
        policy_loss = -(log_prob * new_adv).mean()

        # 5) IQL-style expectile V
        value_error = tgt - v
        value_weight = torch.where(value_error > 0, self.tau, 1.0 - self.tau)
        value_loss = (value_weight * value_error.square()).mean()

        self.a_opt.zero_grad()
        policy_loss.backward()
        self.a_opt.step()

        self.c_opt.zero_grad()
        value_loss.backward()
        self.c_opt.step()

        # 6) Diagnostics (non-breaking, stored in last_stats)
        with torch.no_grad():
            stats = {
                "step": float(self._step),
                "neg_ratio": float(neg_mask.float().mean().item()),
                "dist_decay_now": float(self._current_decay()),
                "log_std_mean": float(log_std.detach().mean().item()),
            }

            if neg_mask.any():
                raw_neg_mass = adv[neg_mask].abs().sum().clamp_min(1e-8)
                eff_neg_mass = new_adv[neg_mask].abs().sum()
                stats.update({
                    "neg_dist2_mean": float(neg_dist2.mean().item()),
                    "neg_dist2_median": float(neg_dist2.median().item()),
                    "neg_dist2_p90": float(torch.quantile(neg_dist2, 0.90).item()),
                    "near_neg_ratio": float(
                        (neg_dist2 <= self.dist_radius).float().mean().item()
                    ),
                    "neg_gate_mean": float(neg_gate.mean().item()),
                    "neg_gate_p10": float(torch.quantile(neg_gate, 0.10).item()),
                    "effective_neg_mass_ratio": float(
                        (eff_neg_mass / raw_neg_mass).item()
                    ),
                    "raw_neg_abs_adv_mean": float(
                        adv[neg_mask].abs().mean().item()
                    ),
                    "weighted_neg_abs_adv_mean": float(
                        new_adv[neg_mask].abs().mean().item()
                    ),
                })
            else:
                stats.update({
                    "neg_dist2_mean": 0.0,
                    "neg_dist2_median": 0.0,
                    "neg_dist2_p90": 0.0,
                    "near_neg_ratio": 0.0,
                    "neg_gate_mean": 0.0,
                    "neg_gate_p10": 0.0,
                    "effective_neg_mass_ratio": 0.0,
                    "raw_neg_abs_adv_mean": 0.0,
                    "weighted_neg_abs_adv_mean": 0.0,
                })

            self.last_stats = stats

        return policy_loss.item() + 0.5 * value_loss.item()
