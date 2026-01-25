import torch
from typing import Dict, Any, Optional
import comfy.sd
import gc

class LazySafetensorsModelPatcher:
    """Lazy wrapper for safetensor models - defers instantiation to sampling time.
    
    Mimics GGUFModelPatcher pattern: holds mmap refs without creating real model.
    Model is only instantiated when .load() is called (at sampling time).
    """
    
    def __init__(self, mmap_sd: Dict[str, torch.Tensor], model_options: Dict, unet_path: str):
        self.mmap_cache = mmap_sd  # Shared mmap refs
        self.model_options = model_options
        self.unet_path = unet_path
        self.current_device = torch.device('cpu')
        self._real_model = None  # Actual ModelPatcher, created lazily
        self._is_instantiated = False
    
    def load(self, device: torch.device, **kwargs):
        """Called at sampling time - instantiate model and load to GPU."""
        
        if not self._is_instantiated:
            print(f"[LazySafetensorsModelPatcher] Instantiating model on first load...")
            # Clone tensors for model instantiation (unavoidable)
            # But this happens at sampling time, not loader time
            # So only one worker instantiates at a time (sequential by nature of sampling)
            isolated = {k: v.clone() for k, v in self.mmap_cache.items()}
            
            load_options = self.model_options.copy()
            cast_dtype = load_options.pop("dtype", None)
            
            model = comfy.sd.load_diffusion_model_state_dict(isolated, model_options=load_options)
            
            if model is None:
                raise RuntimeError(f"Could not load model: {self.unet_path}")
            
            if cast_dtype and hasattr(model, "model"):
                model.model.manual_cast_dtype = cast_dtype
            
            self._real_model = model
            self._is_instantiated = True
            
            # Free the cloned dict
            del isolated
            gc.collect()
            print(f"[LazySafetensorsModelPatcher] Model instantiated.")
        
        # Load to device (this moves weights to GPU)
        if hasattr(self._real_model, 'load'):
            self._real_model.load(device, **kwargs)
        elif hasattr(self._real_model, 'model'):
            self._real_model.model.to(device)
        
        self.current_device = device
    
    def __getattr__(self, name):
        """Proxy all other attributes to the real model.
        
        If model not instantiated, force instantiation on CPU first.
        This handles patching scenarios where model is accessed before sampling.
        """
        if name in ('mmap_cache', 'model_options', 'unet_path', 'current_device', 
                    '_real_model', '_is_instantiated', 'load'):
            return object.__getattribute__(self, name)
        
        real_model = object.__getattribute__(self, '_real_model')
        if real_model is None:
            # Force instantiation on CPU for attribute access
            print(f"[LazySafetensorsModelPatcher] Force instantiating for attribute '{name}'...")
            self.load(torch.device('cpu'))
            real_model = object.__getattribute__(self, '_real_model')
        
        return getattr(real_model, name)
