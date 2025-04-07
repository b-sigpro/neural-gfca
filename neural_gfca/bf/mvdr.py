import numpy as np

import torch


def mvdr_ban(
    Rs: torch.Tensor,
    Rn: torch.Tensor,
    out_ch: int = -1,
    eps: float = 1e-8,
    use_ban: bool = True,
):
    *_, M = Rs.shape

    # mvdr
    eI = eps * torch.eye(M, device=Rs.device)
    V: torch.Tensor = torch.linalg.solve(Rn + eI, Rs)
    V /= torch.einsum("fmm->f", V).real.clip(eps)[..., None, None]

    if out_ch < 0:
        max_snr = -np.inf
        for m in range(M):
            h_ = V[..., m]

            tgt_pwr = torch.einsum("fm,fmn,fn->", h_.conj(), Rs, h_).real
            noi_pwr = torch.einsum("fm,fmn,fn->", h_.conj(), Rn, h_).real

            snr = tgt_pwr / noi_pwr.clip(eps)
            if snr > max_snr:
                max_snr, h = snr, h_
    else:
        h = V[..., out_ch]

    # ban
    if use_ban:
        Rnh = torch.einsum("fmn,fn->fm", Rn, h)

        num = torch.einsum("fm,fm->f", Rnh.conj(), Rnh).real.clip(0).sqrt()
        den = torch.einsum("fm,fm->f", h.conj(), Rnh).real.clip(eps)

        ban = num / den

        h = torch.einsum("f, fm->fm", ban, h)

    return h
