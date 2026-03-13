import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


def install_fla_stubs():
    # stub the minimum `fla` surface so the vendored modules can be imported in
    # this repo without installing the full donor runtime stack.
    if "fla" in sys.modules:
        return

    fla_module = types.ModuleType("fla")
    models_module = types.ModuleType("fla.models")
    models_utils_module = types.ModuleType("fla.models.utils")
    modules_module = types.ModuleType("fla.modules")
    l2warp_module = types.ModuleType("fla.modules.l2warp")
    utils_module = types.ModuleType("fla.utils")
    layers_module = types.ModuleType("fla.layers")
    attn_module = types.ModuleType("fla.layers.attn")

    class Cache(list):
        @classmethod
        def from_legacy_cache(cls, cache):
            return cls(cache or [])

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x * norm * self.weight

    class FusedRMSNormGated(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super().__init__()
            self.norm = RMSNorm(hidden_size, eps=eps)

        def forward(self, x, gate):
            return self.norm(x * torch.sigmoid(gate))

    class ShortConvolution(nn.Module):
        def __init__(self, hidden_size, kernel_size, bias=False, activation="silu"):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, x):
            return torch.zeros_like(x), None

    class Attention(nn.Module):
        def __init__(self, hidden_size, num_heads, num_kv_heads, qk_norm, max_position_embeddings, window_size):
            super().__init__()
            self.kv_dim = hidden_size
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x, attn_mask=None):
            return (x,)

    class FusedCrossEntropyLoss(nn.CrossEntropyLoss):
        def __init__(self, inplace_backward=True):
            super().__init__()

    class FusedLinearCrossEntropyLoss(nn.CrossEntropyLoss):
        def __init__(self, use_l2warp=False):
            super().__init__()

    class GatedMLP(nn.Module):
        def __init__(self, hidden_size, hidden_ratio, intermediate_size, hidden_act, fuse_swiglu):
            super().__init__()
            inner_size = intermediate_size or hidden_size * hidden_ratio
            self.up_proj = nn.Linear(hidden_size, inner_size, bias=False)
            self.down_proj = nn.Linear(inner_size, hidden_size, bias=False)

        def forward(self, x, **kwargs):
            return self.down_proj(torch.nn.functional.silu(self.up_proj(x)))

    def l2_warp(loss, logits):
        return loss

    def identity_decorator(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    models_utils_module.Cache = Cache
    modules_module.RMSNorm = RMSNorm
    modules_module.FusedRMSNormGated = FusedRMSNormGated
    modules_module.ShortConvolution = ShortConvolution
    modules_module.FusedCrossEntropyLoss = FusedCrossEntropyLoss
    modules_module.FusedLinearCrossEntropyLoss = FusedLinearCrossEntropyLoss
    modules_module.GatedMLP = GatedMLP
    attn_module.Attention = Attention
    l2warp_module.l2_warp = l2_warp
    utils_module.input_guard = identity_decorator
    utils_module.autocast_custom_bwd = identity_decorator
    utils_module.autocast_custom_fwd = identity_decorator

    sys.modules["fla"] = fla_module
    sys.modules["fla.models"] = models_module
    sys.modules["fla.models.utils"] = models_utils_module
    sys.modules["fla.modules"] = modules_module
    sys.modules["fla.modules.l2warp"] = l2warp_module
    sys.modules["fla.utils"] = utils_module
    sys.modules["fla.layers"] = layers_module
    sys.modules["fla.layers.attn"] = attn_module


def load_vendor_submodule(name: str):
    package_name = "_vendor_ttt_mag_testpkg"
    root = Path(__file__).resolve().parents[1] / "em_llm" / "ttt_mag"

    if package_name not in sys.modules:
        # load the vendored package by file path so the test can target the new
        # subtree directly without importing the whole EM-LLM package.
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        package_module = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package_module
        package_spec.loader.exec_module(package_module)

    full_name = f"{package_name}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    module_spec = importlib.util.spec_from_file_location(full_name, root / f"{name}.py")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[full_name] = module
    module_spec.loader.exec_module(module)
    return module


def disable_torch_compile():
    # disable compilation in the test harness so the vendored kernels run in a
    # plain eager mode that is stable across local environments.
    def identity_compile(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    torch.compile = identity_compile


class VendoredLinearSGDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disable_torch_compile()
        install_fla_stubs()
        cls.linear_sgd_module = load_vendor_submodule("linear_sgd")
        cls.LinearSGD = cls.linear_sgd_module.LinearSGD

    def create_model(self, mode: str):
        hidden_size = 16
        num_heads = 4
        head_dim = hidden_size // num_heads

        model = self.LinearSGD(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand_v=1,
            chunk_size=4,
            use_gate=False,
            mode=mode,
            layer_idx=0,
        )
        # bypass the projection front-end so this test isolates recurrent-state behavior.
        model.set_qkv_callable(lambda x: (x, x, x))
        return model

    def test_state_continuation_matches_single_pass(self):
        torch.manual_seed(0)
        model = self.create_model(mode="chunk")
        x = torch.randn(2, 12, 16)
        # use a chunk-aligned split because the chunked linear rule only commits
        # state at chunk boundaries.
        split = 8

        full_out, (full_state, _) = model(x)

        prefix_out, prefix_cache = model(x[:, :split, :])
        suffix_out, (suffix_state, _) = model(x[:, split:, :], past_key_values=prefix_cache)

        stitched = torch.cat((prefix_out, suffix_out), dim=1)

        self.assertTrue(torch.allclose(stitched, full_out, atol=1e-5))
        self.assertTrue(torch.allclose(suffix_state, full_state, atol=1e-5))

    def test_chunk_and_naive_match_with_initial_state(self):
        torch.manual_seed(1)
        chunk_model = self.create_model(mode="chunk")
        naive_model = self.create_model(mode="naive")
        naive_model.load_state_dict(chunk_model.state_dict())

        initial_state = torch.randn(2, 4, 4, 4)
        x = torch.randn(2, 8, 16)

        chunk_out, (chunk_state, _) = chunk_model(x, past_key_values=(initial_state, None))
        naive_out, (naive_state, _) = naive_model(x, past_key_values=(initial_state, None))

        self.assertTrue(torch.allclose(chunk_out, naive_out, atol=1e-5))
        self.assertTrue(torch.allclose(chunk_state, naive_state, atol=1e-5))


class VendoredTitansMAGModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disable_torch_compile()
        install_fla_stubs()
        cls.config_module = load_vendor_submodule("configuration_titans_mag")
        cls.modeling_module = load_vendor_submodule("modeling_titans_mag")

    def test_vendored_titans_mag_model_imports_and_instantiates(self):
        config = self.config_module.TitansMAGConfig(
            hidden_size=16,
            num_heads=4,
            head_dim=4,
            expand_v=1,
            chunk_size=4,
            conv_size=2,
            diff_projections=False,
            use_attention=False,
            num_hidden_layers=1,
            vocab_size=32,
            fuse_norm=False,
            fuse_swiglu=False,
            fuse_cross_entropy=False,
            memory_model="linear",
            memory_model_config={"norm_eps": 1e-5, "base_lr": 1.0, "use_gate": False},
        )

        model = self.modeling_module.TitansMAGForCausalLM(config)
        out = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True, return_dict=True)

        self.assertEqual(out.logits.shape, (1, 3, 32))
        self.assertEqual(len(out.past_key_values), 1)


if __name__ == "__main__":
    unittest.main()
