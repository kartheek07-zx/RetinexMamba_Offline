import torch
import torch.nn.functional as F

def selective_scan_fn(
    u, delta, A, B, C, D=None, z=None,
    delta_bias=None, delta_softplus=False,
    return_last_state=False
):
    original_dtype = u.dtype

    u = u.float()
    delta = delta.float()
    A = A.float()

    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)

    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    state = torch.zeros(
        batch, dim, dstate,
        device=u.device,
        dtype=torch.float32
    )

    output = torch.empty(
        batch, dim, seqlen,
        device=u.device,
        dtype=torch.float32
    )

    grouped_B = B.dim() == 4
    grouped_C = C.dim() == 4

    if grouped_B:
        groups = B.shape[1]
        channels_per_group = dim // groups

    for i in range(seqlen):
        u_i = u[:, :, i]
        delta_i = delta[:, :, i]

        deltaA = torch.exp(delta_i.unsqueeze(-1) * A)

        if grouped_B:
            B_i = B[:, :, :, i].float()
            B_i = B_i.unsqueeze(2).expand(
                batch, groups, channels_per_group, dstate
            ).reshape(batch, dim, dstate)
        else:
            B_i = B[:, :, i].float()

        deltaB_u = delta_i.unsqueeze(-1) * B_i * u_i.unsqueeze(-1)

        state = deltaA * state + deltaB_u

        if grouped_C:
            C_i = C[:, :, :, i].float()
            C_i = C_i.unsqueeze(2).expand(
                batch, groups, channels_per_group, dstate
            ).reshape(batch, dim, dstate)
        else:
            C_i = C[:, :, i].float()

        y_i = torch.sum(state * C_i, dim=-1)

        if D is not None:
            y_i = y_i + u_i * D.float().view(1, -1)

        output[:, :, i] = y_i

    if z is not None:
        output = output * F.silu(z.float())

    output = output.to(original_dtype)

    if return_last_state:
        return output, state

    return output
