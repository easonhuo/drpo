from __future__ import annotations

import json
import sys
from pathlib import Path

from drpo.e7_canonical_bootstrap import main as bootstrap_main
from drpo.e7_canonical_injection import write_fingerprint_contract


def test_bootstrap_runs_original_trainer_with_injected_class(tmp_path: Path) -> None:
    source = tmp_path / "canonical"
    source.mkdir()
    (source / "agents.py").write_text(
        "import torch\n"
        "class Actor(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__(); self.w=torch.nn.Linear(1,1); "
        "self.ls=torch.nn.Parameter(torch.zeros(1,1))\n"
        "    def forward(self,s):\n"
        "        m=self.w(s); return m,self.ls.expand_as(m)\n"
        "class Critic(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__(); self.w=torch.nn.Linear(1,1)\n"
        "    def forward(self,s): return self.w(s)\n"
        "class SNA2C_IQLV_DistAgent:\n"
        "    def __init__(self):\n"
        "        self.gamma=.99; self.tau=.7; self.alpha=.11; "
        "self.actor=Actor(); self.critic=Critic(); "
        "self.a_opt=torch.optim.SGD(self.actor.parameters(),lr=.001); "
        "self.c_opt=torch.optim.SGD(self.critic.parameters(),lr=.001)\n"
        "    def update(self,*args,**kwargs): "
        "raise RuntimeError('original update should be patched')\n"
    )
    (source / "trainer.py").write_text(
        "import json,sys,torch,agents\n"
        "out=sys.argv[sys.argv.index('--output')+1]\n"
        "agent=agents.SNA2C_IQLV_DistAgent()\n"
        "s=torch.tensor([[0.],[1.]])\n"
        "agent.update(s,torch.tensor([[0.],[1.]]),torch.tensor([1.,-1.]),"
        "s,torch.tensor([False,True]))\n"
        "json.dump(agent._drpo_last_negative_control_metrics,open(out,'w'))\n"
    )
    contract_path = tmp_path / "contract.json"
    write_fingerprint_contract(
        canonical_root=source,
        agents_relpath="agents.py",
        trainer_relpath="trainer.py",
        module_name="agents",
        target_class="SNA2C_IQLV_DistAgent",
        expected_canonical_alpha=0.11,
        output=contract_path,
    )
    branch_path = tmp_path / "branch.json"
    branch_path.write_text(
        json.dumps(
            {
                "branch_id": "fixture",
                "branch_kind": "injected",
                "dataset_id": "fixture",
                "dataset_sha256": "0" * 64,
                "seed": 1,
                "template_values": {},
                "negative_control": {
                    "method": "global",
                    "negative_scale": 0.01,
                    "canonical_alpha": 0.11,
                    "reference_distance": 2.0,
                    "reciprocal_linear_coefficient": 0.4362580032734791,
                    "reciprocal_quadratic_coefficient": 0.5520268617673281,
                    "exponential_coefficient": 0.374162511054291,
                },
            }
        )
    )
    output = tmp_path / "metrics.json"
    manifest = tmp_path / "manifest.json"
    try:
        code = bootstrap_main(
            [
                "--contract",
                str(contract_path),
                "--branch-config",
                str(branch_path),
                "--branch-manifest",
                str(manifest),
                "--",
                "--output",
                str(output),
            ]
        )
    finally:
        sys.modules.pop("agents", None)
    assert code == 0
    metrics = json.loads(output.read_text())
    assert metrics["method"] == "global"
    assert metrics["negative_scale"] == 0.01
    assert json.loads(manifest.read_text())["status"] == "completed"
