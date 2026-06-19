import torch


def fold_act_scale_into_weights(packed: dict, clip: dict) -> dict:
    qmin = -8
    qmax = 7

    packed_folded = {}

    for module_name, layer_data in packed.items():
        rows = layer_data["shape"][0]
        cols = layer_data["shape"][1]

        w_q = unpack_int4(layer_data["weight_int4_bytes"], rows, cols)

        w_scale = layer_data["weight_scale"]
        s_x = clip[module_name] / 7.0
        s_x = s_x.to(device=w_scale.device, dtype=w_scale.dtype)

        w_dq = w_q.float() * w_scale.unsqueeze(1).float()
        w_new = w_dq * s_x.unsqueeze(0).float()

        row_max = w_new.abs().amax(dim=1)
        new_scale = (row_max / qmax).clamp(min=1e-8)
        new_scale_unsq = new_scale.unsqueeze(1)
        w_new_q = torch.clamp(torch.round(w_new / new_scale_unsq), qmin, qmax).to(torch.int8)

        rows, cols = w_new_q.shape
        if cols % 2 != 0:
            w_new_q = torch.cat([w_new_q, torch.zeros(rows, 1, dtype=torch.int8)], dim=1)

        low = (w_new_q[:, ::2] & 0xF).to(torch.uint8)
        high = ((w_new_q[:, 1::2] & 0xF) << 4).to(torch.uint8)
        packed_bytes = (low | high).contiguous()

        alignment = 16
        current = packed_bytes.numel()
        rem = current % alignment
        if rem != 0:
            pad = alignment - rem
            packed_bytes = torch.cat([packed_bytes.flatten(), torch.zeros(pad, dtype=torch.uint8)])

        packed_folded[module_name] = {
            "weight_int4_bytes": packed_bytes.cpu().numpy().tobytes(),
            "weight_scale": new_scale.half().cpu(),
            "weight_scale_bytes": new_scale.half().detach().cpu().numpy().tobytes(),
            "shape": (rows, cols),
        }

    return packed_folded


def unpack_int4(packed_bytes: bytes, rows: int, cols: int) -> torch.Tensor:
    packed = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8).clone()
    packed = packed[:(rows * cols + 1) // 2]
    packed = packed.reshape(rows, -1)

    lo = (packed & 0xF).to(torch.int8)
    hi = ((packed >> 4) & 0xF).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)

    w_q = torch.empty(rows, cols if cols % 2 == 0 else cols + 1, dtype=torch.int8)
    w_q[:, ::2] = lo
    w_q[:, 1::2] = hi

    if cols % 2 != 0:
        w_q = w_q[:, :cols]

    return w_q
