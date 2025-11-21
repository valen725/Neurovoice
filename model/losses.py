import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and focusing on hard examples.
    Args:
        gamma (float): Focusing parameter > 0. Default: 2.0
        alpha (list or tensor or float): Class weighting factor. If list, length must = num_classes.
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):  # noqa: D401
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha])  # broadcast later
            else:
                self.alpha = alpha.type(torch.float32)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        Args:
            logits: (B, C) raw outputs.
            targets: (B,) class indices.
        """
        if logits.ndim != 2:
            raise ValueError("FocalLoss expects logits shape (B, C)")
        if targets.ndim != 1:
            targets = targets.view(-1)

        probs = F.softmax(logits, dim=1)
        # Gather probabilities of the true classes
        true_probs = probs[torch.arange(probs.size(0)), targets]
        log_true = torch.log(true_probs + 1e-12)
        focal_factor = (1 - true_probs) ** self.gamma

        if self.alpha is not None:
            if self.alpha.numel() == 1:
                alpha_t = self.alpha.to(logits.device)
            else:
                alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = 1.0

        loss = -alpha_t * focal_factor * log_true

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

__all__ = ['FocalLoss']
