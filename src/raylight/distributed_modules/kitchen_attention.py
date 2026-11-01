import torch
import comfy_kitchen


def kitchen_attention(q, k, v, *args, softmax_scale=None, causal=False, window_size=(-1, -1),
                      alibi_slopes=None, dropout_p=0.0, return_attn_probs=False,
                      joint_tensor_key=None, joint_tensor_value=None, joint_strategy="none", **kwargs):
    if causal or tuple(window_size) != (-1, -1) or alibi_slopes is not None or dropout_p or return_attn_probs:
        raise ValueError("Comfy Kitchen INT8 requires non-causal, full attention without attention-probability output.")
    if joint_tensor_key is not None:
        if joint_strategy == "rear":
            k = torch.cat((k, joint_tensor_key), dim=1)
            v = torch.cat((v, joint_tensor_value), dim=1)
        elif joint_strategy == "front":
            k = torch.cat((joint_tensor_key, k), dim=1)
            v = torch.cat((joint_tensor_value, v), dim=1)
        else:
            raise ValueError(f"Unsupported joint attention strategy: {joint_strategy}")
    # xFuser exchanges BLHD tensors; Kitchen consumes BHLD.
    out = comfy_kitchen.int8_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous()
