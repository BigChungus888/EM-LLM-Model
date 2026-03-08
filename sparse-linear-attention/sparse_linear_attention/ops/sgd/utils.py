import torch


# Modified from https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/fusions/fused_bias_gelu.py#L26
def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


def gelu_bwd_bwd(x):
    a = 0.79788456
    b = 0.044715
    c = 0.1070322243

    arg = a * x * (1 + b * x * x)
    tanh_out = torch.tanh(arg)
    tanh_out_sq = tanh_out * tanh_out
    sech_sq = 1 - tanh_out_sq

    # d(tanh(arg))/dx = sech²(arg) * d(arg)/dx
    # d(arg)/dx = a * (1 + b*x²) + a*x*(2*b*x) = a * (1 + 3*b*x²)
    d_arg_dx = a * (1 + 3 * b * x * x)
    d_tanh_dx = sech_sq * d_arg_dx

    # gelu_bwd = 0.5 * x * sech² * (a + c*x²) + 0.5 * (1 + tanh)
    # d(gelu_bwd)/dx = 0.5 * [sech² * (a + c*x²) + x * d(sech²)/dx * (a + c*x²) + x * sech² * 2*c*x] + 0.5 * d(tanh)/dx

    # d(sech²)/dx = d(1 - tanh²)/dx = -2*tanh * d(tanh)/dx
    d_sech_sq_dx = -2 * tanh_out * d_tanh_dx

    term1 = 0.5 * sech_sq * (a + c * x * x)
    term2 = 0.5 * x * d_sech_sq_dx * (a + c * x * x)
    term3 = 0.5 * x * sech_sq * 2 * c * x
    term4 = 0.5 * d_tanh_dx

    return term1 + term2 + term3 + term4
