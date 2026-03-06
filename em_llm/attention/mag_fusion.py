import torch
import torch.nn as nn


class MAGFusion(nn.Module):
    """Fuse episodic and recurrent branches with MAG-style gating.

    Contract:
        fused = out_proj(norm_ep + sigmoid(gate_proj(norm_ep)) * norm_ttt)
    """

    def __init__(self, hidden_size: int, norm_eps: float = 1e-5, baseline_safe_init: bool = True):
        super().__init__()
        self.ep_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ttt_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if baseline_safe_init:
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)

    def forward(self, episodic_out: torch.Tensor, ttt_out: torch.Tensor) -> torch.Tensor:
        norm_ep = self.ep_norm(episodic_out)
        norm_ttt = self.ttt_norm(ttt_out)
        gate = torch.sigmoid(self.gate_proj(norm_ep))
        fused = self.out_proj(norm_ep + gate * norm_ttt)
        return fused
