import torch.nn as nn

def forward_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN detected in {module.__class__.__name__} during forward pass")
        raise ValueError("NaN detected in forward pass")

class ActivationMonitor:
    def __init__(self):
        self.activations = {}

    def hook_fn(self, module, input, output):
        if module not in self.activations:
            self.activations[module] = []
        self.activations[module].append(output.detach().cpu().numpy())

    def clear(self):
        self.activations = {}

    def get_statistics(self):
        stats = {}
        for module, activations in self.activations.items():
            activations = np.concatenate(activations, axis=0)
            stats[module] = {
                'mean': np.mean(activations),
                'min': np.min(activations),
                'max': np.max(activations),
                'std': np.std(activations)
            }
        return stats

def register_hooks(model, monitor):
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                sub_layer.register_forward_hook(monitor.hook_fn)
        else:
            layer.register_forward_hook(monitor.hook_fn)
