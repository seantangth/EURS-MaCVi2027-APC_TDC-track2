"""Export helpers copied verbatim from our competition code (3_src/eurs/export_onnx.py), minus the steps that only checked
the result (official FLOPs script, onnxruntime comparison) and do not change the file."""
import hashlib

import numpy as np
import onnx
import torch


def _dedup_initializers(m):
    """exp087：同一模組在圖中呼叫兩次（雙尺度共享編碼器）時，匯出器在每個呼叫點各自摺疊 Conv＋BN，產生內容相同的兩份權重；
    官方腳本以 initializer 總和計參數，會重複計算 → 內容（dtype、形狀、位元組）相同的 initializer 合併成一份。單次呼叫的模型不受影響。"""
    import hashlib
    from onnx import numpy_helper
    canon, alias, keep = {}, {}, []
    for t in m.graph.initializer:
        a = numpy_helper.to_array(t)
        key = (a.dtype.str, a.shape, hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest())
        if key in canon:
            alias[t.name] = canon[key]
        else:
            canon[key] = t.name; keep.append(t)
    if alias:
        for n in m.graph.node:
            for i, name in enumerate(n.input):
                if name in alias:
                    n.input[i] = alias[name]
        del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    return len(alias)


class _HSig(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x * (1.0 / 6.0) + 0.5, 0.0, 1.0)            # ＝relu6(x+3)/6，匯出成 Mul/Add/Clip


class _HSwish(torch.nn.Module):
    def forward(self, x):
        return x * torch.clamp(x * (1.0 / 6.0) + 0.5, 0.0, 1.0)


def basic_acts(model):
    """把 HardSwish／HardSigmoid（MobileNetV3 的啟動與 SE gate）換成 Mul/Add/Clip，避開部署端對這兩個 op 的支援疑慮（數值等價）。"""
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            k = type(child).__name__.lower()
            if k in ('hardswish', 'hardswishme'):
                setattr(mod, cname, _HSwish()); n += 1
            elif k in ('hardsigmoid', 'hardsigmoidme'):
                setattr(mod, cname, _HSig()); n += 1
    print(f'basic_acts: replaced {n} modules')
    return model
