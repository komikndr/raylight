import os
import uuid
import comfy.utils
import comfy.lora
from comfy.lora_convert import convert_lora
from raylight.comfy_dist.lora import load_lora as dist_load_lora

class LoraManager:
    def __init__(self, worker):
        self.worker = worker
        self._applied_loras = set()
        self._lora_configs = {} # config_hash -> [(lora_path, strength), ...]
        self._current_lora_config_hash = None

    def load_lora(self, lora_path, strength_model, lora_config_hash=None):
        """Loads a LoRA into the model on this worker via mmap."""
        filename = os.path.basename(lora_path)
        
        if filename.startswith("._") or filename == ".DS_Store":
            print(f"[RayWorker {self.worker.local_rank}] Skipping hidden/junk file: {filename}")
            return True

        # Store this LoRA in the config registry for re-application after offload
        if lora_config_hash is not None:
            if lora_config_hash not in self._lora_configs:
                self._lora_configs[lora_config_hash] = []
            lora_entry = (lora_path, strength_model)
            # Avoid duplicate entries in the same config
            if lora_entry not in self._lora_configs[lora_config_hash]:
                self._lora_configs[lora_config_hash].append(lora_entry)
                print(f"[RayWorker {self.worker.local_rank}] Registered LoRA for config {lora_config_hash}: {filename}")
        
        # BRANCH ISOLATION: If lora_config_hash changed, reset all LoRAs first
        if lora_config_hash is not None and lora_config_hash != self._current_lora_config_hash:
            if self._current_lora_config_hash is not None:
                print(f"[RayWorker {self.worker.local_rank}] LoRA config changed ('{self._current_lora_config_hash}' -> '{lora_config_hash}'). Resetting patches...")
                self.reset_loras()
            self._current_lora_config_hash = lora_config_hash
        
        # Create a unique signature for this LoRA
        lora_sig = (lora_path, strength_model)
        
        # IDEMPOTENCY: Skip if this exact LoRA was already applied in this config
        if lora_sig in self._applied_loras:
            print(f"[RayWorker {self.worker.local_rank}] LoRA already applied: {filename} (strength={strength_model}). Skipping duplicate.")
            return True

        print(f"[RayWorker {self.worker.local_rank}] Loading LoRA: {filename} (strength={strength_model})")
        
        # 1. Ensure model is loaded/re-hydrated
        self.worker.reload_model_if_needed()
        
        # 2. Apply using core logic (shared with re-apply)
        self._apply_lora_core(lora_path, strength_model)
        
        print(f"[RayWorker {self.worker.local_rank}] LoRA applied successfully.")
        return True

    def reset_loras(self):
        """Clears all applied LoRAs and resets the model patches."""
        print(f"[RayWorker {self.worker.local_rank}] Resetting LoRAs...")
        
        # Clear tracking
        self._applied_loras.clear()
        
        # FIX Bug #1: Clear the config hash so next LoRA application is seen as new
        self._current_lora_config_hash = None
        
        # Clear model patches if model exists
        if self.worker.model is not None:
            # Clear the patches dict
            if hasattr(self.worker.model, "patches"):
                self.worker.model.patches.clear()
            
            # FIX Bug #3: Update patches_uuid to signal ComfyUI that patches changed
            if hasattr(self.worker.model, "patches_uuid"):
                self.worker.model.patches_uuid = uuid.uuid4()
            
            # Restore original weights from backup if any
            if hasattr(self.worker.model, "backup") and self.worker.model.backup:
                print(f"[RayWorker {self.worker.local_rank}] Restoring {len(self.worker.model.backup)} backed up weights...")
                for k, bk in self.worker.model.backup.items():
                    if bk.inplace_update:
                        comfy.utils.copy_to_param(self.worker.model.model, k, bk.weight)
                    else:
                        comfy.utils.set_attr_param(self.worker.model.model, k, bk.weight)
                self.worker.model.backup.clear()
        
        print(f"[RayWorker {self.worker.local_rank}] LoRAs reset complete.")

    def reapply_loras_for_config(self, config_hash):
        """Re-apply all LoRAs for a specific config_hash after model reload."""
        # Nothing to do if no config stored
        if config_hash not in self._lora_configs:
            print(f"[RayWorker {self.worker.local_rank}] No LoRAs registered for config {config_hash}. Skipping re-apply.")
            return True
        
        # Check if we already have the right config applied
        if self._current_lora_config_hash == config_hash and self._applied_loras:
            print(f"[RayWorker {self.worker.local_rank}] Config {config_hash} already applied. Skipping re-apply.")
            return True
        
        print(f"[RayWorker {self.worker.local_rank}] Re-applying LoRAs for config {config_hash}...")
        
        # Reset current patches first
        self.reset_loras()
        
        # Ensure model is loaded
        self.worker.reload_model_if_needed()
        
        # Re-apply each LoRA in this config
        lora_list = self._lora_configs[config_hash]
        for lora_path, strength in lora_list:
            filename = os.path.basename(lora_path)
            print(f"[RayWorker {self.worker.local_rank}] Re-applying LoRA: {filename} (strength={strength})")
            self._apply_lora_core(lora_path, strength)
        
        # Update tracking
        self._current_lora_config_hash = config_hash
        
        print(f"[RayWorker {self.worker.local_rank}] Re-applied {len(lora_list)} LoRAs for config {config_hash}.")
        return True
    
    def _apply_lora_core(self, lora_path, strength_model):
        """Core LoRA application logic without registration/tracking."""
        filename = os.path.basename(lora_path)
        
        # Load LoRA State Dict (Mmap)
        lora_sd = comfy.utils.load_torch_file(lora_path)
        
        # Resolve Keys & Convert
        key_map = {}
        if self.worker.model is not None:
            key_map = comfy.lora.model_lora_keys_unet(self.worker.model.model, key_map)
        
        lora_patches = convert_lora(lora_sd)
        
        # Load Patches using Raylight's distributed lora helper
        loaded_patches = dist_load_lora(lora_patches, key_map)
        
        # Apply to Model
        self.worker.model.add_patches(loaded_patches, strength_model)
        
        # Track as applied
        self._applied_loras.add((lora_path, strength_model))
        
        print(f"[RayWorker {self.worker.local_rank}] LoRA core apply complete: {filename}")
    
    def clear_gpu_refs(self):
        """Clean up references to allow VRAM release."""
        # 1. Clear weight_function/bias_function closures on ALL modules
        if self.worker.model is None: return

        diffusion_model = getattr(self.worker.model, "model", None)
        if diffusion_model is not None:
            cleared_funcs = 0
            for m in diffusion_model.modules():
                if hasattr(m, "weight_function") and len(m.weight_function) > 0:
                    m.weight_function = []
                    cleared_funcs += 1
                if hasattr(m, "bias_function") and len(m.bias_function) > 0:
                    m.bias_function = []
                    cleared_funcs += 1
            if cleared_funcs > 0:
                print(f"[RayWorker {self.worker.local_rank}] Cleared {cleared_funcs} weight/bias functions.")
        
        # 2. Clear patches dict on model patcher
        if hasattr(self.worker.model, "patches") and self.worker.model.patches:
            patch_count = len(self.worker.model.patches)
            self.worker.model.patches.clear()
            print(f"[RayWorker {self.worker.local_rank}] Cleared {patch_count} patches from patcher.")
        
        # 3. Clear .patches attribute on individual parameters (GGMLTensor carries these)
        if diffusion_model is not None:
            cleared_param_patches = 0
            for name, param in diffusion_model.named_parameters():
                if hasattr(param, "patches") and param.patches:
                    param.patches = []
                    cleared_param_patches += 1
            if cleared_param_patches > 0:
                print(f"[RayWorker {self.worker.local_rank}] Cleared .patches on {cleared_param_patches} parameters.")

    def clear_tracking(self):
        """Clears tracking state."""
        self._applied_loras.clear()
        self._current_lora_config_hash = None
