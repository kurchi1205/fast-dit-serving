import torch
import torch.nn as nn

# def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
#     """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
#     for key in ckpt.keys():
#         model_key = key
#         if remap is not None and key in remap:
#             model_key = remap[key]
#         if model_key.startswith(prefix) and not model_key.startswith("loss."):
#             path = model_key[len(prefix) :].split(".")
#             obj = model
#             for p in path:
#                 if obj is list:
#                     obj = obj[int(p)]
#                 else:
#                     obj = getattr(obj, p, None)
#                     if obj is None:
#                         print(
#                             f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
#                         )
#                         break
#             if obj is None:
#                 continue
#             try:
#                 tensor = ckpt.get_tensor(key).to(device=device)
#                 if dtype is not None and tensor.dtype != torch.int32:
#                     tensor = tensor.to(dtype=dtype)
#                 obj.requires_grad_(False)
#                 # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
#                 if obj.shape != tensor.shape:
#                     print(
#                         f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
#                     )
#                 obj.set_(tensor)
#             except Exception as e:
#                 print(f"Failed to load key '{key}' in safetensors file: {e}")
#                 raise e


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """
    Apply weights from a safetensors handle into a PyTorch module safely.

    Changes vs original:
      - Traversal supports lists/tuples/ModuleList/Sequential with numeric indices.
      - Moves each tensor to the *target param's device* (falls back to `device` arg).
      - Uses .copy_/.data.copy_ under no_grad (no cross-device set_).
      - Casts only floating dtypes when dtype is provided.
      - Skips missing paths and shape-mismatch safely.
    """
    SEQ_TYPES = (list, tuple, nn.ModuleList, nn.Sequential)
    norm_prefix = prefix if not prefix or prefix.endswith(".") else prefix + "."
    loaded = mismatched = skipped = 0

    @torch.no_grad()
    def _copy_into(obj, t, name):
        nonlocal loaded, mismatched, skipped
        # determine target device (prefer the param's device)
        target_dev = getattr(obj, "device", None)
        if target_dev is None:
            target_dev = torch.device(device)
        t = t.to(target_dev)

        # optional dtype cast (floats only)
        if dtype is not None and t.dtype.is_floating_point:
            t = t.to(dtype)

        # shape check
        try:
            obj_shape = obj.shape
        except AttributeError:
            print(f"Skipping key '{name}' (target is not a Tensor/Parameter).")
            skipped += 1
            return

        if obj_shape != t.shape:
            print(f"W: shape mismatch for key {name}, {obj_shape} != {t.shape}")
            mismatched += 1
            return

        # copy (no storage swap)
        if isinstance(obj, nn.Parameter):
            obj.requires_grad_(False)
            obj.data.copy_(t)
            loaded += 1
        elif isinstance(obj, torch.Tensor):
            obj.copy_(t)
            loaded += 1
        else:
            print(f"Skipping key '{name}' (target not a Tensor/Parameter).")
            skipped += 1

    for key in ckpt.keys():
        model_key = remap.get(key, key) if remap and key in remap else key
        if norm_prefix and not model_key.startswith(norm_prefix):
            continue
        if model_key.startswith("loss."):
            continue

        subkey = model_key[len(norm_prefix):] if norm_prefix else model_key
        if not subkey:
            continue

        # traverse the path safely
        path_elems = subkey.split(".")
        obj = model
        valid = True
        for p in path_elems:
            if p.isdigit() and isinstance(obj, SEQ_TYPES):
                idx = int(p)
                try:
                    obj = obj[idx]
                except Exception:
                    valid = False
                    break
            else:
                obj = getattr(obj, p, None)
                if obj is None:
                    valid = False
                    break

        if not valid or obj is None:
            print(f"Skipping key '{model_key}' in safetensors file: path '{subkey}' not found in model")
            skipped += 1
            continue

        try:
            t = ckpt.get_tensor(key)  # typically CPU tensor
            _copy_into(obj, t, model_key)
        except Exception as e:
            print(f"Failed to load key '{key}' in safetensors file: {e}")
            raise

    print(f"load_into summary: loaded={loaded}, mismatched={mismatched}, skipped={skipped}")

def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text


def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text


def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)