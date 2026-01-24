import torch
from typing import Dict, Optional, Any

class SafetensorMmapWrapper:
    """Wraps mmap'd safetensor state dict for streaming GPU transfer.
    
    Enables per-tensor GPU loading to avoid RAM spikes from full clone.
    """
    
    def __init__(self, mmap_sd: Dict[str, torch.Tensor]):
        self._mmap = mmap_sd
    
    def stream_to_model(self, model, device: torch.device) -> int:
        """Transfer weights per-parameter to avoid RAM spike.
        
        Returns number of parameters transferred.
        """
        transferred = 0
        param_map = {name: param for name, param in model.named_parameters()}
        
        for name, mmap_tensor in self._mmap.items():
            target_name = self._resolve_name(name, param_map)
            if target_name and target_name in param_map:
                # Use as_subclass to stream directly from mmap to GPU (Zero-Copy)
                param_map[target_name].data = mmap_tensor.as_subclass(torch.Tensor).to(device)
                transferred += 1
        
        return transferred
    
    def _resolve_name(self, name: str, param_map: Dict) -> Optional[str]:
        """Resolve mmap key to model parameter name."""
        # Direct match
        if name in param_map:
            return name
        # Common prefixes
        prefixes = ["diffusion_model.", "model.diffusion_model.", ""]
        for prefix in prefixes:
            candidate = f"{prefix}{name}" if prefix else name
            if candidate in param_map:
                return candidate
            # Strip prefix if present
            if name.startswith(prefix) and name[len(prefix):] in param_map:
                return name[len(prefix):]
        return None
