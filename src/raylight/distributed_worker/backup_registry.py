
import ray

@ray.remote
class BackupRegistry:
    """
    Acts as a central registry for deduplicating GGUF model backups.
    Workers upgrade their weights to RAM (Object Store) and register them here.
    Subsequent workers reuse the ObjectRef, preventing N-times RAM usage.
    
    This class is isolated in its own file to avoid importing ComfyUI libraries,
    which trigger CUDA initialization and crash if the actor is scheduled on a CPU node.
    """
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put_if_missing(self, key, ref):
        """
        Stores the ref if key is missing.
        Returns the stored ref (whether it was just added or existed).
        This ensures 'first writer wins' and all workers agree on the same Ref.
        """
        if key not in self.store:
            self.store[key] = ref
        return self.store[key]

    def clear(self):
        self.store.clear()
