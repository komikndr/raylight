import gc
import logging

from torch.distributed.fsdp import FSDPModule
import comfy.model_management as mm


def cleanup_models_gc():
    models = mm.current_loaded_models

    do_gc = any(m.is_dead() for m in models)
    if not do_gc:
        return
    logging.info(
        "Potential memory leak detected, doing a full garbage collect. "
        "For maximum performance avoid circular references in model code."
    )

    gc.collect()
    mm.soft_empty_cache()

    for i in range(len(models) - 1, -1, -1):
        cur = models[i]

        if not cur.is_dead():
            continue

        model_name = cur.real_model().__class__.__name__

        if isinstance(cur.real_model().diffusion_model, FSDPModule):
            logging.info(f"FSDP wrapped module detected with model {model_name}")
            models.pop(i)
        else:
            logging.warning(
                f"WARNING, memory leak with model {model_name}. "
                f"Please make sure it is not being referenced somewhere."
            )
