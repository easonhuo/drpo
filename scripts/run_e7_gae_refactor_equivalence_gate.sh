#!/usr/bin/env bash
set -euo pipefail
O=2fe97cdcff0e8361b33193dd2a7be8cf63c44a3b; R=$(git rev-parse --show-toplevel); N=$(git rev-parse HEAD); D=${1:-$R/artifacts/e7_gae_refactor_equivalence}; T=$(mktemp -d); trap 'git -C "$R" worktree remove --force "$T/o" >/dev/null 2>&1||:; rm -rf "$T"' EXIT
mkdir -p "$D"; git cat-file -e "$O^{commit}" 2>/dev/null||git fetch --no-tags origin "$O"; git worktree add --detach "$T/o" "$O" >/dev/null
cat >"$T/g.py" <<'PY'
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('commit');p.add_argument('out');a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
def H(x):
 h=hashlib.sha256()
 def F(v):
  if torch.is_tensor(v):
   z=v.detach().cpu().contiguous();h.update(b'T'+str(z.dtype).encode()+repr(tuple(z.shape)).encode()+z.numpy().tobytes())
  elif isinstance(v,dict):
   h.update(b'D');[(F(k),F(v[k])) for k in sorted(v,key=repr)]
  elif isinstance(v,(list,tuple)):h.update(type(v).__name__.encode());[F(i) for i in v]
  else:h.update((type(v).__name__+':'+repr(v)).encode())
 F(x);return h.hexdigest()
class A(torch.nn.Module):
 def __init__(s):super().__init__();s.m=torch.nn.Linear(2,1,bias=False);s.l=torch.nn.Parameter(torch.zeros(1,1))
 def forward(s,x):y=s.m(x);return y,s.l.expand_as(y)
class C(torch.nn.Module):
 def __init__(s):super().__init__();s.v=torch.nn.Linear(2,1,bias=False)
 def forward(s,x):return s.v(x)
class B:
 def __init__(s):s.gamma=.9;s.tau=.7;s.alpha=.11;s.actor=A();s.critic=C();s.a_opt=torch.optim.SGD(s.actor.parameters(),lr=.01);s.c_opt=torch.optim.SGD(s.critic.parameters(),lr=.01)
r={'observations':np.array([[1,0],[0,1],[1,1],[2,1],[-1,.5],[.5,-1],[-.5,-.5],[1.5,-.5]],np.float32),'actions':np.array([[.8],[-.7],[.2],[1.1],[-.4],[.6],[-.2],[.9]],np.float32),'rewards':np.array([.2,-.4,.3,-.8,.5,-.2,.1,-.6],np.float32),'terminals':np.array([0,1,0,0,0,0,0,0],bool),'timeouts':np.array([0,0,1,0,0,1,0,0],bool)};r['next_observations']=r['observations']*.6+np.array([.1,-.05],np.float32)
seq=[[0,1],[2,3],[4,5],[6,7],[1,6],[0,7],[2,5],[3,4],[0,4]];cases=[]
from drpo.e7_canonical_injection import NegativeControl
for est in ('td','gae'):
 for cn in ('positive_only','sqexp_c128'):
  c=NegativeControl(method='positive_only' if cn[0]=='p' else 'exponential',negative_scale=0 if cn[0]=='p' else 1/.11,canonical_alpha=.11,reference_distance=2,exponential_coefficient=0 if cn[0]=='p' else 128)
  if a.kind=='old':
   from drpo.e7_canonical_gae_injection import OrderedReplay,SnapshotEstimatorConfig,build_joint_snapshot_agent_class
   q=OrderedReplay(r['observations'],r['actions'],r['rewards'],r['next_observations'],r['terminals'],r['timeouts']);K=build_joint_snapshot_agent_class(B,replay=q,negative_control=c,estimator=SnapshotEstimatorConfig(est,.95,2),return_mode='metrics_dict');x=K();tab=lambda:x._drpo_advantage_table;sn=lambda:list(x._drpo_snapshot_hashes)
  else:
   from drpo.e7_canonical_injection import build_injected_agent_class
   from drpo.e7_squared_exp_night_bootstrap import TrajectorySnapshotAdvantage
   q=TrajectorySnapshotAdvantage(r,est,batch_size=2);K=build_injected_agent_class(B,control=c,return_mode='metrics_dict',advantage_provider=q);x=K();tab=lambda:q.table;sn=lambda:list(q.snapshot_hashes)
  with torch.no_grad():x.actor.m.weight[:]=torch.tensor([[.15,-.2]]);x.actor.l[:]=-.1;x.critic.v.weight[:]=torch.tensor([[.25,-.35]])
  z={'case':est+'__'+cn,'initial':[H(x.actor.state_dict()),H(x.critic.state_dict()),H(x.a_opt.state_dict()),H(x.c_opt.state_dict())],'updates':[]}
  for u,ii in enumerate(seq,1):
   j=np.array(ii);m=x.update(torch.from_numpy(r['observations'][j]),torch.from_numpy(r['actions'][j]),torch.from_numpy(r['rewards'][j]),torch.from_numpy(r['next_observations'][j]),torch.from_numpy(r['terminals'][j]),torch.from_numpy(j.astype(np.float32)))
   z['updates'].append({'u':u,'ids':ii,'h':[H(x.actor.state_dict()),H(x.critic.state_dict()),H(x.a_opt.state_dict()),H(x.c_opt.state_dict()),H(tab())],'snap':sn(),'m':[m[k] for k in ('actor_loss','critic_loss','positive_fraction','negative_fraction','negative_factor_mean')]})
  cases.append(z)
Path(a.out).write_text(json.dumps({'commit':a.commit,'cases':cases},sort_keys=True,indent=2)+'\n')
PY
PYTHONHASHSEED=0 PYTHONPATH="$T/o/src" python "$T/g.py" old "$O" "$D/OLD_TRACE.json"; PYTHONHASHSEED=0 PYTHONPATH="$R/src" python "$T/g.py" new "$N" "$D/NEW_TRACE.json"
O="$O" N="$N" D="$D" python - <<'PY'
import json,math,os
from pathlib import Path
p=Path(os.environ['D']);o=json.loads((p/'OLD_TRACE.json').read_text());n=json.loads((p/'NEW_TRACE.json').read_text());f=[]
for a,b in zip(o['cases'],n['cases'],strict=True):
 if a['case']!=b['case'] or a['initial']!=b['initial']:f.append({'case':a['case'],'u':0})
 for x,y in zip(a['updates'],b['updates'],strict=True):
  if x['ids']!=y['ids'] or x['h']!=y['h'] or x['snap']!=y['snap'] or any(not math.isclose(float(i),float(j),rel_tol=0,abs_tol=1e-12) for i,j in zip(x['m'],y['m'],strict=True)):f.append({'case':a['case'],'u':x['u']});break
A={'gate_id':'EXT-H-E7-SQEXP-GAE-REFACTOR-EQUIVALENCE-01','status':'PASS' if not f else 'FAIL','old_commit':o['commit'],'new_commit':n['commit'],'case_count':4,'updates_per_case':9,'refresh_positions':[1,5,9],'failure_count':len(f),'first_failure':f[0] if f else None,'formal_evidence_allowed':False,'held_out_seeds_touched':False,'real_data_layer_executed':False};(p/'SOURCE_IDENTITY.json').write_text(json.dumps({'old_commit':o['commit'],'new_commit':n['commit']},indent=2)+'\n');(p/'EQUIVALENCE_AUDIT.json').write_text(json.dumps(A,indent=2)+'\n');print(json.dumps(A,indent=2));raise SystemExit(bool(f))
PY
