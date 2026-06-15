import json
import struct

import torch


def pack_int4(
    quantized: dict,
    smoothing_folded: bool = True,
    layout_version: int = 1,
    pack_order: str = "low_high",
):
    qmin = -8
    qmax = 7

    packed = {}

    for module_name, layer_data in quantized.items():
        w_q = layer_data["qweight"]
        scale = layer_data["scale"]
        rows, cols = w_q.shape

        w_q_flat = w_q.flatten().int()
        w_q_clamped = torch.clamp(w_q_flat, qmin, qmax)

        if cols % 2 != 0:
            w_q_clamped = torch.cat([w_q_clamped, torch.zeros(1, dtype=torch.int)])
            cols_padded = cols + 1
        else:
            cols_padded = cols

        w_q_padded = w_q_clamped.reshape(rows, cols_padded)

        if pack_order == "low_high":
            low = (w_q_padded[:, ::2] & 0xF).to(torch.uint8)
            high = ((w_q_padded[:, 1::2] & 0xF) << 4).to(torch.uint8)
        else:
            low = ((w_q_padded[:, ::2] & 0xF) << 4).to(torch.uint8)
            high = (w_q_padded[:, 1::2] & 0xF).to(torch.uint8)

        packed_bytes = (low | high).contiguous()

        alignment = 16
        current_size = packed_bytes.numel()
        remainder = current_size % alignment
        if remainder != 0:
            pad = alignment - remainder
            packed_bytes = torch.cat([
                packed_bytes.flatten(),
                torch.zeros(pad, dtype=torch.uint8),
            ])

        scale_bytes = scale.detach().cpu().numpy().tobytes()

        meta = {
            "layout_version": layout_version,
            "rows": rows,
            "cols": cols,
            "cols_padded": cols_padded,
            "pack_order": pack_order,
            "signed_range": [qmin, qmax],
            "weight_scale_dtype": "float16",
            "smoothing_folded": smoothing_folded,
            "transpose_required": False,
            "alignment_bytes": alignment,
        }

        packed[module_name] = {
            "weight_int4_bytes": packed_bytes.cpu().numpy().tobytes(),
            "weight_scale_bytes": scale_bytes,
            "weight_scale": scale.cpu(),
            "shape": (rows, cols),
            "pack_meta": meta,
        }

    return packed
