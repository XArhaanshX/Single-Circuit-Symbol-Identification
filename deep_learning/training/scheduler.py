"""
Training scheduler: Phase 1 (frozen backbone) -> Phase 2 (fine-tune).
"""

import torch.optim as optim


def create_optimizer_phase1(model, lr: float = 1e-3, weight_decay: float = 1e-4):
    """Phase 1: Freeze backbone, train embedding head only."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adam(trainable, lr=lr, weight_decay=weight_decay)


def create_optimizer_phase2(model, lr: float = 1e-4, weight_decay: float = 1e-4):
    """Phase 2: Unfreeze backbone with reduced LR."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, T_max: int = 20):
    """Cosine annealing LR scheduler."""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
