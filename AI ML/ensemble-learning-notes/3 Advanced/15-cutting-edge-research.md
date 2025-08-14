# 15 — Cutting-Edge Research

Goal: Explore research-driven ensemble strategies used in modern systems.

## Snapshot Ensembles (SNE)
- Train one model with cyclical/cosine learning rates; save multiple snapshots; average predictions.
- Cheap ensembling without training multiple models from scratch.

```python
# Pseudo-PyTorch sketch
for cycle in range(K):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=0.9)
    for epoch in range(T_cycle):
        lr = cosine_anneal(epoch, T_cycle, lr_max, lr_min)
        set_lr(optimizer, lr)
        train_one_epoch()
    save_snapshot(model.state_dict())
# At test time: average logits from snapshots
```

## Stochastic Weight Averaging (SWA)
- Maintain a running average of weights along the tail of training; improves generalization.

```python
# PyTorch SWA utils
from torch.optim.swa_utils import AveragedModel, SWALR
base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
swa_model = AveragedModel(model)
swa_start = 150
for epoch in range(200):
    train_one_epoch()
    if epoch >= swa_start:
        swa_model.update_parameters(model)
# Update BN stats
torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

## Deep Ensembles
- Train M independently initialized networks; average predictive probabilities.
- Strong uncertainty estimates and OOD detection; higher compute cost.

## Mixture of Experts (MoE)
- A gating network routes inputs to specialized expert models (can be sparse for efficiency).
- Scales capacity; used in large-scale language and vision models.

## Notes
- For uncertainty, prefer deep ensembles or probabilistic approaches (MC Dropout, SWAG).
- Combine SWA/Snapshot with classical ensembles for robust winners.
