import warnings
from copy import copy

import pytest
from fla.models.utils import Cache

from sparse_linear_attention.layers.ttt import TTT
from sparse_linear_attention.modules.linear_sgd import LinearSGD
from sparse_linear_attention.modules.mlp_sgd import SwiGLUSGD
from sparse_linear_attention.ops.sgd.chunk import *
from sparse_linear_attention.ops.sgd.naive import mlp_chunk

warnings.simplefilter(action="ignore", category=FutureWarning)


def create_memory_model(memory_type: str, hidden_size: int, num_heads: int, head_dim: int, chunk_size: int,
                        memory_params: Optional[dict] = None):
    """Helper function to create memory models for testing."""
    if memory_params is None:
        memory_params = {}

    def dummy_get_qkv(x):
        return x, x, x

    if memory_type == "linear":
        model = LinearSGD(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand_v=2,
            chunk_size=chunk_size,
            get_qkv_fn=None,
            **memory_params
        )
    elif memory_type == "mlp":
        model = SwiGLUSGD(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand_v=2,
            base_lr=1.0,
            chunk_size=chunk_size,
            **memory_params
        )
    else:
        raise ValueError(f"Unknown memory_type: {memory_type}")

    model.set_qkv_callable(dummy_get_qkv)
    return model


def create_ttt(hidden_size, device, dtype, num_heads, memory_model) -> TTT:
    return TTT(
        hidden_size=hidden_size,
        memory_model=memory_model,
        num_heads=num_heads,
        expand_v=2,
        chunk_size=16,
    ).to(device).to(dtype)


def test_ttt_linear_causality():
    dtype = torch.bfloat16
    B, L = 2, 512
    L_new = 96

    hidden_size = 128
    num_heads = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    memory_model = create_memory_model(
        memory_type="linear",
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
        chunk_size=64,
    )

    x_a = torch.randn(B, L, hidden_size).to(device).to(dtype)
    model = TTT(
        hidden_size=hidden_size,
        memory_model=memory_model,
        num_heads=num_heads,
        expand_v=2,
        chunk_size=64,
    ).to(device).to(dtype)

    x_b = x_a.clone()
    x_b[:, L_new:] = torch.randn(B, L - L_new, hidden_size)  # different future

    with torch.no_grad():
        y_a, _, _ = model(x_a)
        y_b, _, _ = model(x_b)

    assert torch.allclose(y_a[:, :L_new], y_b[:, :L_new], atol=1e-3), "Layer causality is violated"


def test_ttt_linear_equivalence():
    dtype = torch.float32
    B, L = 2, 512

    d_model = 128
    heads = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    memory_model_chunk = create_memory_model(
        memory_type="linear",
        hidden_size=d_model,
        num_heads=heads,
        head_dim=d_model // heads,
        chunk_size=64,
        memory_params={"mode": "chunk"},
    )

    memory_model_naive = create_memory_model(
        memory_type="linear",
        hidden_size=d_model,
        num_heads=heads,
        head_dim=d_model // heads,
        chunk_size=64,
        memory_params={"mode": "naive"},
    )

    model_chunk = create_ttt(d_model, device, dtype, heads, memory_model_chunk)
    model_naive = create_ttt(d_model, device, dtype, heads, memory_model_naive)
    model_naive.load_state_dict(model_chunk.state_dict())

    x = torch.randn(B, L, d_model).to(device).to(dtype)
    with torch.no_grad():
        y_a, _, (final_state_a, _) = model_chunk(x, use_cache=True)
        y_b, _, (final_state_b, _) = model_naive(x, use_cache=True)

    assert torch.allclose(y_a, y_b, atol=1e-5)
    assert torch.allclose(final_state_a, final_state_b, atol=1e-5)


@pytest.mark.parametrize("mode", ["naive", "chunk"])
def test_ttt_mlp(mode):
    dtype = torch.float32
    B, L = 2, 512

    d_model = 1024
    heads = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    memory_model = create_memory_model(
        memory_type="mlp",
        hidden_size=d_model,
        num_heads=heads,
        head_dim=d_model // heads,
        chunk_size=64,
        memory_params={"mode": mode},
    )

    model_chunk = create_ttt(d_model, device, dtype, heads, memory_model)

    x = torch.randn(B, L, d_model).to(device).to(dtype)
    with torch.no_grad():
        y_a, _, (final_state_a, _) = model_chunk(x, use_cache=True)

    assert torch.isfinite(y_a).all(), "Output contains non-finite values"


def test_ttt_mlp_equivalence():
    dtype = torch.float32
    B, L = 2, 512

    hidden_size = 1024
    num_heads = 8
    head_dim = hidden_size // num_heads
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    memory_model_chunk = create_memory_model(
        memory_type="mlp",
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        chunk_size=64,
        memory_params={"mode": "chunk", "layer_idx": 0},
    )

    memory_model_naive = create_memory_model(
        memory_type="mlp",
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        chunk_size=64,
        memory_params={"mode": "naive", "layer_idx": 0},
    )

    model_chunk = create_ttt(hidden_size, device, dtype, num_heads, memory_model_chunk)
    model_naive = create_ttt(hidden_size, device, dtype, num_heads, memory_model_naive)
    model_naive.load_state_dict(model_chunk.state_dict())

    x = torch.randn(B, L, hidden_size).to(device).to(dtype)
    w1 = 0.02 * torch.randn(B, num_heads, 4 * head_dim, head_dim, device=device, dtype=dtype)
    w2 = 0.02 * torch.randn(B, num_heads, 2 * head_dim, 4 * head_dim, device=device, dtype=dtype)
    past_key_values = Cache()
    past_key_values.update(recurrent_state=(w1, w2), layer_idx=0)
    with torch.no_grad():
        y_a, _, (final_state_a, _) = model_chunk(x, past_key_values=copy(past_key_values), use_cache=True)
        y_b, _, (final_state_b, _) = model_naive(x, past_key_values=copy(past_key_values), use_cache=True)

    assert torch.allclose(y_a, y_b, atol=1e-5)
    for state1, state2 in zip(final_state_a, final_state_b):
        assert torch.allclose(state1, state2, atol=1e-5)


def test_ttt_mlp_backward():
    dtype = torch.float32
    B, L = 2, 512

    hidden_size = 1024
    num_heads = 8
    head_dim = hidden_size // num_heads
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    memory_model_chunk = create_memory_model(
        memory_type="mlp",
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        chunk_size=64,
        memory_params={"mode": "chunk", "layer_idx": 0},
    )

    memory_model_naive = create_memory_model(
        memory_type="mlp",
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        chunk_size=64,
        memory_params={"mode": "naive", "layer_idx": 0},
    )

    model_chunk = create_ttt(hidden_size, device, dtype, num_heads, memory_model_chunk)
    model_naive = create_ttt(hidden_size, device, dtype, num_heads, memory_model_naive)
    model_naive.load_state_dict(model_chunk.state_dict())

    x = torch.randn(B, L, hidden_size).to(device).to(dtype).requires_grad_(True)
    w1 = 0.02 * torch.randn(B, num_heads, 4 * head_dim, head_dim, device=device, dtype=dtype)
    w2 = 0.02 * torch.randn(B, num_heads, 2 * head_dim, 4 * head_dim, device=device, dtype=dtype)
    past_key_values = Cache()
    past_key_values.update(recurrent_state=(w1, w2), layer_idx=0)
    y_a, _, _ = model_chunk(x, past_key_values=copy(past_key_values), use_cache=True)
    y_a.sum().backward()
    dx1 = x.grad
    x.grad = None
    y_b, _, _ = model_naive(x, past_key_values=copy(past_key_values), use_cache=True)
    y_b.sum().backward()
    dx2 = x.grad
    assert torch.allclose(dx1, dx2, atol=1e-5)


def test_mlp_rule_backward():
    dtype = torch.float32
    B, T = 2, 512
    BT = 64
    hidden_size = 1024
    num_heads = 8
    head_dim = hidden_size // num_heads
    hidden_ratio = 4
    init_std = 0.02
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    w1 = init_std * torch.randn(B, num_heads, hidden_ratio * head_dim, head_dim, device=device, dtype=dtype)
    w2 = init_std * torch.randn(B, num_heads, 2 * head_dim, hidden_ratio * head_dim, device=device, dtype=dtype)
    initial_weights = (w1, w2)
    q = torch.randn(B, num_heads, T, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, num_heads, T, head_dim, device=device, dtype=dtype)
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    v = torch.randn(B, num_heads, T, 2 * head_dim, device=device, dtype=dtype)
    eta = torch.rand(B, num_heads, T, device=device, dtype=dtype)
    theta = torch.rand(B, num_heads, T, device=device, dtype=dtype)
    q, k, v, eta, theta = map(lambda t: t.requires_grad_(True), (q, k, v, eta, theta))
    _, outputs1 = ChunkMLPUpdate.apply(q, k, v, eta, theta, initial_weights, BT)
    outputs1.sum().backward(retain_graph=True)
    dq1, dk1, dv1, deta1, dtheta1 = q.grad, k.grad, v.grad, eta.grad, theta.grad
    q.grad, k.grad, v.grad, eta.grad, theta.grad = None, None, None, None, None
    _, outputs2 = mlp_chunk(q, k, v, eta, theta, initial_weights, init_std, hidden_ratio, B, num_heads, head_dim, T,
                            2 * head_dim, BT)
    outputs2.sum().backward()
    dq2, dk2, dv2, deta2, dtheta2 = q.grad, k.grad, v.grad, eta.grad, theta.grad
    assert torch.allclose(dq1, dq2, atol=1e-5)
    assert torch.allclose(dk1, dk2, atol=1e-5)
    assert torch.allclose(dv1, dv2, atol=1e-5)
    assert torch.allclose(deta1, deta2, atol=1e-5)
    assert torch.allclose(dtheta1, dtheta2, atol=1e-5)


def test_mlp_rule_backward_step():
    dtype = torch.float32
    B, T = 2, 512
    BT = 64
    hidden_size = 1024
    num_heads = 8
    head_dim = hidden_size // num_heads
    hidden_ratio = 4
    init_std = 0.02
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    w1 = init_std * torch.randn(B, num_heads, hidden_ratio * head_dim, head_dim, device=device, dtype=dtype)
    w2 = init_std * torch.randn(B, num_heads, 2 * head_dim, hidden_ratio * head_dim, device=device, dtype=dtype)
    q = torch.randn(B, num_heads, BT, head_dim, device=device, dtype=dtype)
    k = torch.randn(B, num_heads, BT, head_dim, device=device, dtype=dtype)
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    v = torch.randn(B, num_heads, BT, 2 * head_dim, device=device, dtype=dtype)
    eta = torch.rand(B, num_heads, BT, device=device, dtype=dtype)
    eta.requires_grad_(True)
    theta = torch.rand(B, num_heads, BT, device=device, dtype=dtype)
    beta, eta_cumsum = compute_temporal_gradient_weights(eta)
    q, k, v, theta, w1, w2 = map(lambda t: t.requires_grad_(True), (q, k, v, theta, w1, w2))
    w1_new, w2_new, z2_bar, cache = update_weights_forward(w1, w2, q, k, v, beta, eta_cumsum, theta, return_cache=True)
    (w1_new.sum() + w2_new.sum() + z2_bar.sum()).backward(retain_graph=True)
    dq1, dk1, dv1, deta1, dtheta1, dw1_1, dw2_1 = (q.grad, k.grad, v.grad, eta.grad, theta.grad, w1.grad, w2.grad)
    q.grad, k.grad, v.grad, eta.grad, theta.grad = None, None, None, None, None
    dz2_bar, dw1_new, dw2_new = torch.ones_like(z2_bar), torch.ones_like(w1_new), torch.ones_like(w2_new)
    dq2, dk2, dv2, dbeta2, deta_cumsum2, dtheta2, dw1_2, dw2_2 = update_weights_backward(dz2_bar, dw1_new, dw2_new, q,
                                                                                         k, beta, eta_cumsum, theta,
                                                                                         cache)
    deta2 = compute_temporal_gradient_weights_backward(dbeta2, deta_cumsum2, eta, eta_cumsum)
    assert torch.allclose(dq1, dq2, atol=1e-5)
    assert torch.allclose(dk1, dk2, atol=1e-5)
    assert torch.allclose(dv1, dv2, atol=1e-5)
    assert torch.allclose(deta1, deta2, atol=1e-5)
    assert torch.allclose(dtheta1, dtheta2, atol=1e-5)
    assert torch.allclose(dw1_1, dw1_2, atol=1e-5)
    assert torch.allclose(dw2_1, dw2_2, atol=1e-5)


def test_ttt_mlp_causality():
    dtype = torch.bfloat16
    B, L = 2, 512
    L_new = 96

    hidden_size = 1024
    num_heads = 8
    head_dim = hidden_size // num_heads
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    memory_model = create_memory_model(
        memory_type="mlp",
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        chunk_size=64,
        memory_params={"mode": "chunk", "layer_idx": 0},
    )

    x_a = torch.randn(B, L, hidden_size).to(device).to(dtype)
    model = TTT(
        hidden_size=hidden_size,
        memory_model=memory_model,
        num_heads=num_heads,
        chunk_size=64,
        expand_v=2,
    ).to(device).to(dtype)

    x_b = x_a.clone()
    x_b[:, L_new:] = torch.randn(B, L - L_new, hidden_size)  # different future
    w1 = 0.02 * torch.randn(B, num_heads, 4 * head_dim, head_dim, device=device, dtype=dtype)
    w2 = 0.02 * torch.randn(B, num_heads, 2 * head_dim, 4 * head_dim, device=device, dtype=dtype)
    past_key_values = Cache()
    past_key_values.update(recurrent_state=(w1, w2), layer_idx=0)

    with torch.no_grad():
        y_a, _, _ = model(x_a, past_key_values=copy(past_key_values), use_cache=True)
        y_b, _, _ = model(x_b, past_key_values=copy(past_key_values), use_cache=True)

    assert torch.allclose(y_a[:, :L_new], y_b[:, :L_new], atol=1e-3), "Layer causality is violated"
