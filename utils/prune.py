import torch.nn.utils.prune as prune
import torch.nn as nn

def apply_pruning(model, amount=0.5):
    """Apply L1 unstructured pruning to all Conv2d and Linear layers."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.l1_unstructured(module, name="bias", amount=amount)
    print(f"Applied pruning: {amount*100}% of weights set to zero.")

def remove_pruning(model):
    """Remove pruning masks and make pruning permanent."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
            prune.remove(module, "bias")
    print("Pruning masks removed. Weights are now permanently sparse.")
