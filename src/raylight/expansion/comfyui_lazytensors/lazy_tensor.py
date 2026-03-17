import torch


class MaterializedTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func.__name__ == "_has_compatible_shallow_copy_type":
            return True

        if not all(issubclass(t, (torch.Tensor, MaterializedTensor, LazySafetensor)) for t in types):
            return NotImplemented

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def __new__(cls, data: torch.Tensor, mmap_ref: torch.Tensor):
        return torch.Tensor._make_subclass(cls, data)

    def __init__(self, data: torch.Tensor, mmap_ref: torch.Tensor):
        super().__init__()
        object.__setattr__(self, "_mmap_ref", mmap_ref)

    def to(self, *args, **kwargs):
        device = None
        if args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        if str(device) == "meta":
            meta = torch.empty(self.shape, dtype=self.dtype, device="meta")
            return MaterializedTensor(meta, self.mmap_ref)

        return MaterializedTensor(super().to(*args, **kwargs), self.mmap_ref)

    @property
    def mmap_ref(self) -> torch.Tensor:
        return object.__getattribute__(self, "_mmap_ref")

    def detach(self):
        return MaterializedTensor(super().detach(), self.mmap_ref)


class LazySafetensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func.__name__ == "_has_compatible_shallow_copy_type":
            return True

        if not all(issubclass(t, (torch.Tensor, MaterializedTensor, LazySafetensor)) for t in types):
            return NotImplemented

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def __new__(cls, mmap_tensor: torch.Tensor, patches=None):
        return torch.Tensor._make_subclass(cls, mmap_tensor)

    def __init__(self, mmap_tensor: torch.Tensor, patches=None):
        super().__init__()
        object.__setattr__(self, "_is_mmap", True)
        object.__setattr__(self, "patches", patches if patches is not None else [])

    def __copy__(self):
        return LazySafetensor(self, patches=self.patches.copy())

    def __deepcopy__(self, memo):
        return LazySafetensor(self, patches=[p for p in self.patches])

    def to(self, *args, **kwargs):
        device = None
        if args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        if str(device) == "meta":
            meta = torch.empty(self.shape, dtype=self.dtype, device="meta")
            return LazySafetensor(meta, patches=getattr(self, "patches", []))

        if device is not None:
            target_device = torch.device(device)
            current_device = torch.device(self.device)
            if target_device.type == current_device.type:
                if target_device.index == current_device.index or target_device.type == "cpu":
                    return self

        materialized = self.as_subclass(torch.Tensor).to(*args, **kwargs)
        return MaterializedTensor(materialized, self)

    def clone(self, *args, **kwargs):
        return torch.Tensor.clone(self, *args, **kwargs)

    def detach(self):
        new = torch.Tensor._make_subclass(LazySafetensor, torch.Tensor.detach(self))
        object.__setattr__(new, "_is_mmap", True)
        object.__setattr__(new, "patches", getattr(self, "patches", []))
        return new


def wrap_state_dict_lazy(sd: dict) -> dict:
    wrapped = {}
    for key, tensor in sd.items():
        if isinstance(tensor, torch.Tensor) and not isinstance(tensor, LazySafetensor):
            wrapped[key] = LazySafetensor(tensor)
        else:
            wrapped[key] = tensor
    return wrapped
