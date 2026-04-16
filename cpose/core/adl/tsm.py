import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM) for ADL research.
    Shifts a portion of channels along the temporal dimension.
    Reference: Lin et al., ICCV 2019
    """
    def __init__(self, n_segment=8, n_div=8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        # x order: (N*T, C, H, W) -> (N, T, C, H, W)
        nt, c, h, w = x.size()
        n = nt // self.n_segment
        x = x.view(n, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        
        # Shift forward
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # Shift backward
        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
        # Keep rest
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)

class TSM(nn.Module):
    def __init__(self, num_classes, base_model='resnet50'):
        super(TSM, self).__init__()
        # In research, we usually wrap a backbone with TSM
        self.tsm = TemporalShift(n_segment=8)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Placeholder for TSM forward pass
        x = self.tsm(x)
        # ... logic to pass through backbone ...
        return x
