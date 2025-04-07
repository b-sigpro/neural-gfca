import torch


def calc_exponential_window(T: int, decay: float, device: str | torch.device = "cuda"):
    """_summary_

    Args:
        T (int): window length
        decay (float): decay parameter

    Returns:
        Window (torch [T, T])
    """

    win = torch.arange(T)
    win = torch.abs(win[None, :] - win[:, None])
    win = (1 - decay) ** win

    return win.to(device)
