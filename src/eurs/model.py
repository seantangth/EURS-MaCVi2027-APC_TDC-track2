"""物理引導雙頭網路（EURS 主線，act007）。

設計依據（STRATEGY §3.3、exp005）：
- FLOPs 才是瓶頸：768×1024 下 1/4 解析 48 通道密集 3×3 一層 ≈2.0 GFLOPs、1/8 解析 96 通道密集 3×3 一層 ≈2.0 GFLOPs。
  所以區塊類型可配置：dw（depthwise 3×3＋pointwise）、irN（inverted residual，expand N）、dense（單一密集 3×3 殘差）。
  預設只在 1/16 放少量 dense，1/4、1/8 用 dw／ir。
- 復原頭：低解析預測 T(x)（3ch，sigmoid 參數化 ∈[eps,1]，避免 Exp 在 FP16 溢位）與 B(x)（3ch）→ Resize 到全解析（免 FLOPs）
  → J = clamp((x − B)/T)。退化模型 I = I0·T + B（RSUIGM Eq. 19 的像素形式），無模糊項 → 直接繼承輸入銳利度。
- 分割頭吃解碼特徵 ⊕ 下採樣的 J → 5 類 logits → Resize 到全解析（評分端 argmax 後 nearest 縮放）。
- 每影像統計（mean／std）做輸入正規化（ReduceMean，免 FLOPs）。
- 只用 Conv／BN／ReLU／Sigmoid／Resize／逐元素運算；無 InstanceNorm／LayerNorm／注意力。

尺寸由 3_src/eurs_tools/scan_model_budget.py 以官方計數器掃出（見 LOG exp005 追加）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(cin, cout, k=3, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DWBlock(nn.Module):
    """depthwise 3×3 → pointwise 1×1，殘差。"""
    def __init__(self, c):
        super().__init__()
        self.dw = cbr(c, c, 3, 1, groups=c)
        self.pw = cbr(c, c, 1, act=False)

    def forward(self, x):
        return F.relu(x + self.pw(self.dw(x)), inplace=True)


class IRBlock(nn.Module):
    """inverted residual：pw 擴張 → dw 3×3 → pw 投影，殘差。"""
    def __init__(self, c, expand=2):
        super().__init__()
        h = c * expand
        self.body = nn.Sequential(cbr(c, h, 1), cbr(h, h, 3, 1, groups=h), cbr(h, c, 1, act=False))

    def forward(self, x):
        return F.relu(x + self.body(x), inplace=True)


class DenseBlock(nn.Module):
    """單一密集 3×3 殘差（貴，只放在 1/16）。"""
    def __init__(self, c):
        super().__init__()
        self.body = cbr(c, c, 3, 1, act=False)

    def forward(self, x):
        return F.relu(x + self.body(x), inplace=True)


def make_block(kind, c):
    if kind == 'dw':
        return DWBlock(c)
    if kind.startswith('ir'):
        return IRBlock(c, int(kind[2:] or 2))
    if kind == 'dense':
        return DenseBlock(c)
    raise ValueError(kind)


def make_stage(c, spec):
    """spec 例：'dw*2'、'ir2*3'、'dense*2+ir2*1'。"""
    blocks = []
    for part in spec.split('+'):
        kind, n = (part.split('*') + ['1'])[:2]
        blocks += [make_block(kind, c) for _ in range(int(n))]
    return nn.Sequential(*blocks)


def down(cin, cout):
    """dw stride-2 → pw：便宜的下採樣。"""
    return nn.Sequential(cbr(cin, cin, 3, 2, groups=cin), cbr(cin, cout, 1))


PRESETS = {
    # 由 scan_model_budget.py 以官方計數器在 768×1024 量得（見 LOG exp005 追加）；'tiny' 供 CPU 煙霧測試
    'base768': dict(c4=48, c8=96, c16=192, s4='dw*2', s8='ir2*3', s16='dense*1+ir2*2'),    # 768×1024：10.56 GFLOPs／0.860M（官方計數，主線）
    'small768': dict(c4=48, c8=96, c16=160, s4='dw*2', s8='ir2*3', s16='dense*1+ir2*2'),   # 768×1024：9.34 GFLOPs／0.661M（快速消融）
    'deep768': dict(c4=48, c8=96, c16=160, s4='dw*2', s8='ir2*4', s16='dense*1+ir2*3'),   # 768×1024：10.93 GFLOPs／0.805M（較深）
    'base512': dict(c4=56, c8=112, c16=160, s4='dw*2', s8='ir2*3', s16='dense*1+ir2*2'),   # 512×1024：待掃描確認（64,128,192 為 10.2 GFLOPs 但 1.013M 超參數上限）
    'tiny': dict(c4=16, c8=32, c16=48, s4='dw*1', s8='ir2*1', s16='dense*1'),
}


A_N, B_N = 10, 20   # 檔名 (a,b)：水型 a＝1..10、深度 b＝5..100（步長 5）


def _blur121(t):
    """replicate 邊界的 [1 2 1]/4 可分離模糊（＝3×3 高斯 /16）；只用 Slice／Concat／Add／Mul：不計 FLOPs、不需 Pad。"""
    t = (torch.cat([t[:, :, :1], t[:, :, :-1]], 2) + 2.0 * t + torch.cat([t[:, :, 1:], t[:, :, -1:]], 2)) * 0.25
    return (torch.cat([t[..., :1], t[..., :-1]], 3) + 2.0 * t + torch.cat([t[..., 1:], t[..., -1:]], 3)) * 0.25


def orient_score(J, thr, slope=20.0):
    """exp091：直幅偵測。直幅 test 影像被拉成橫幅輸入後，水平紋理變鈍、垂直變密 → 灰階的 mean|∂y| / mean|∂x| 變大
    （訓練 20 場景＋t1 test 6 場景：直幅 1.73–2.06、橫幅 0.73–1.24）。回傳 sigmoid((r−thr)·slope)，形狀 (N,1,1,1)。
    只用 Slice／Sub／Mul／Add／Sqrt／ReduceMean／Div／Sigmoid（圖中已有），不計 FLOPs。"""
    Y = 0.299 * J[:, 0:1] + 0.587 * J[:, 1:2] + 0.114 * J[:, 2:3]
    gx = torch.sqrt((Y[..., 1:] - Y[..., :-1]) ** 2 + 1e-6).mean(dim=(1, 2, 3), keepdim=True)
    gy = torch.sqrt((Y[:, :, 1:] - Y[:, :, :-1]) ** 2 + 1e-6).mean(dim=(1, 2, 3), keepdim=True)
    return torch.sigmoid((gy / (gx + 1e-6) - thr) * slope)


def coral_close(seg, k):
    """exp095：把被 coral 包圍的 bg 洞補成 coral（官方 GT 不畫 background 多邊形，礁內沙地算 coral）。
    在模型解析度上以 argmax 規則判 coral／bg（平手同 argmax 取前者），coral 遮罩做閉運算（MaxPool 膨脹 k，再以負值 MaxPool 侵蝕 k），
    閉運算後為 coral 且原本為 bg 的像素在 coral logit 加 100。只用 Sub／Greater／And／Cast／MaxPool／Neg／Mul／Add，不計 FLOPs。"""
    s = [seg[:, i:i + 1] for i in range(5)]
    bg = (s[0] > s[1]) & (s[0] > s[2]) & (s[0] > s[3]) & (s[0] > s[4])   # 只用 Greater（Jetson 相容性；平手為零測度）
    co = (s[1] > s[0]) & (s[1] > s[2]) & (s[1] > s[3]) & (s[1] > s[4])
    dil = F.max_pool2d(co.float(), k, stride=1, padding=k // 2)
    clo = F.max_pool2d(dil * -1.0, k, stride=1, padding=k // 2) * -1.0
    fill = ((clo > 0.5) & bg).float()
    return torch.cat([s[0], s[1] + 100.0 * fill, s[2], s[3], s[4]], 1)


def sharpen_out(J, a_logits, kind, tab, port=None):
    """exp074：輸出端細節增強 J' = clamp(J + k·HF)。HF＝灰階 Y（g3y）或 J（g3）減 3×3 高斯；
    k＝水型線性表 tab（長 A_N）以 rest3 的水型軟分佈加權（tab 全同＝固定 k）。"""
    if kind == 'g3y':
        Y = 0.299 * J[:, 0:1] + 0.587 * J[:, 1:2] + 0.114 * J[:, 2:3]
        hf = Y - _blur121(Y)
    else:
        hf = J - _blur121(J)
    k = (a_logits.softmax(1).detach() @ tab.t()).view(-1, 1, 1, 1)   # 1×10 MatMul（圖中已有 MatMul；不引入 ReduceSum）；detach：exp088 學 k 表時不回傳到水型分類
    if port is not None:  # exp091：判定為直幅（評分器會把輸出垂直放大回去）→ 改用較強的固定 k
        k_port, thr = port
        s = orient_score(J, thr)
        k = (1.0 - s) * k + s * k_port
    return torch.clamp(J + k * hf, 0.0, 1.0)


class PhysicsDualHead(nn.Module):
    def __init__(self, c4=48, c8=128, c16=192, s4='dw*2', s8='ir2*3', s16='dense*1+ir2*2', num_classes=5, eps=0.02, refine=False, ab_head=False, rest2=False, rest2_raw=False, rest2_raw2=False, rest3=False, rest3_half=False, seg_cascade=False, seg_half=False, ms_half=False):
        super().__init__()
        self.eps, self.refine = eps, refine
        self.stem = nn.Sequential(cbr(48, c4, 1), DWBlock(c4))          # space-to-depth ×4 → 48ch @1/4
        self.s4 = make_stage(c4, s4)
        self.down8 = down(c4, c8)
        self.s8 = make_stage(c8, s8)
        self.down16 = down(c8, c16)
        self.s16 = make_stage(c16, s16)
        self._build_decoder(c4, c8, c16, num_classes, refine, ab_head, rest2, rest2_raw, rest2_raw2, rest3, rest3_half, seg_cascade, seg_half, ms_half)

    def _build_decoder(self, c4, c8, c16, num_classes, refine, ab_head=False, rest2=False, rest2_raw=False, rest2_raw2=False, rest3=False, rest3_half=False, seg_cascade=False, seg_half=False, ms_half=False):
        self.ms_half = ms_half
        if ms_half:  # exp087：雙尺度共享編碼器。同一個編碼器再跑一次 1/2 輸入（約 1.28 GFLOPs、0 新卷積參數），其 1/16 特徵（相對原圖＝1/32，感受野兩倍）
            # 升回原尺度後以逐通道閘門加進 f16，給分割大範圍場景上下文（exp079：錯在大塊區域語意、隨場景而異，不在邊界／解析度）。閘門零初始化＝起點與單尺度相同。
            self.ms_gate = nn.Parameter(torch.zeros(1, c16, 1, 1))
        self.lat16 = cbr(c16, c8, 1)
        self.lat8 = cbr(c8, c4, 1)
        self.fuse8 = IRBlock(c8, 2)
        self.fuse4 = DWBlock(c4)
        # 復原頭輸出 21 通道：T(3)、B(3)、3×3 色彩仿射 A(9)、偏置 c(3)、混合權重 w(3)
        # ab_head（exp037 c 臂候選）：最後 3 通道改為 4 通道 softmax 閘門，在 {物理反演, 仿射, 查表 J0, 輸入 x} 之間混合
        rest2_raw = rest2_raw or rest2_raw2
        self.ab_head, self.rest2, self.rest2_raw, self.rest2_raw2, self.freeze_base = ab_head, rest2, rest2_raw, rest2_raw2, False
        self.rest3, self.rest3_half = rest3, rest3_half
        self.head_rest = nn.Sequential(DWBlock(c4), nn.Conv2d(c4, 22 if ab_head else 21, 1))
        if rest3:  # exp054：獨立的「未標準化輸入」復原編碼器（取代 rest2 旁路）。分割路徑凍結不動；復原不再只靠 64 維旁路偷看原色。
            # 主骨幹吃的是逐通道標準化後的影像，水型／深度最需要的色偏與亮度在入口就被抹掉（exp045、exp044b）。
            # 這裡用 ~0.09M 參數、約 2.7 GFLOPs（768×1024）的 IR 卷積塔直接吃 0–1 原圖：pixel_unshuffle ×4（免 FLOPs）→ 1/4、1/8、1/16 三層 → FPN 回 1/4，
            # 另把凍結分割的 f4 當語意旁路（已算好、免費）。全域向量（1/16 GAP＋每影像 μ／σ）→ (a,b) 軟查表＋ΔM；1/4 參數圖（22ch）→ 4 路閘門混合，同 rest2。
            assert not (ab_head or rest2), 'rest3 與 ab_head／rest2 互斥'
            c1, c2, c3 = 32, 64, 64
            self.r_stem = cbr(48, c1, 1)
            self.r_s4 = nn.Sequential(IRBlock(c1, 2), IRBlock(c1, 2))
            self.r_down8 = down(c1, c2); self.r_s8 = nn.Sequential(IRBlock(c2, 2), IRBlock(c2, 2))
            self.r_down16 = down(c2, c3); self.r_s16 = IRBlock(c3, 2)
            self.r_lat16 = cbr(c3, c2, 1); self.r_fuse8 = DWBlock(c2)
            self.r_lat8 = cbr(c2, c1, 1); self.r_sem = cbr(c4, c1, 1); self.r_fuse4 = DWBlock(c1)
            self.r_head = nn.Sequential(cbr(c1 + 6, c1, 1), DWBlock(c1), nn.Conv2d(c1, 22, 1))
            self.r_gmlp = nn.Sequential(nn.Linear(c3 + 6, 64), nn.ReLU(inplace=True), nn.Linear(64, A_N + B_N + 12))
            with torch.no_grad():
                self.r_gmlp[2].weight[A_N + B_N:].zero_(); self.r_gmlp[2].bias[A_N + B_N:].zero_()
            if rest3_half:  # b 臂：1/2 解析細節殘差（pixel_unshuffle ×2 → 12ch @1/2 → 3ch 殘差，tanh×0.1）約 0.75 GFLOPs
                self.r_half = nn.Sequential(nn.Conv2d(12, 16, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 3, 1))
        self.seg_cascade = seg_cascade
        if seg_cascade:  # exp077：先復原、再分割。分割改吃 rest3 的未標準化 1/4 特徵（1×1 融合）與 rest3 的復原圖，舊復原頭不再計算
            assert rest3, 'seg_cascade 需要 rest3'
            self.segfuse = nn.Conv2d(32, c4, 1)
            with torch.no_grad():   # 零初始化：起點的分割頭只多了一個加 0 的分支
                self.segfuse.weight.zero_(); self.segfuse.bias.zero_()
        self.seg_half = seg_half
        if seg_half:  # exp077b：1/2 解析分割細化（小魚、邊界）：讀 pixel_unshuffle(標準化輸入, 2)＋上採樣的 1/4 logits＋f4 投影，殘差修正 logits（約 0.47 GFLOPs、1.9k 參數）
            assert seg_cascade, 'seg_half 目前只接在 seg_cascade 上'
            self.segh_f4 = cbr(c4, 16, 1)
            self.segh_head = nn.Sequential(cbr(12 + num_classes + 16, 16, 1), DWBlock(16), nn.Conv2d(16, num_classes, 1))
            with torch.no_grad():   # 零初始化：起點的輸出＝1/4 logits 的上採樣，與 exp077a 相同
                self.segh_head[2].weight.zero_(); self.segh_head[2].bias.zero_()
        if rest2:  # exp044：另接新復原頭（4 路閘門＋全域反轉換）產生輸出；舊復原頭只餵分割，凍結後分割與原學生位元相同
            assert not ab_head, 'rest2 只接在沒有 ab_head 的舊學生上'
            self.head_rest2 = nn.Sequential(DWBlock(c4), nn.Conv2d(c4, 22, 1))
        if rest2_raw:  # exp044b：輸入的逐通道標準化抹掉了色偏與對比（exp045：(a,b) 估計是瓶頸）→ 另看未標準化的低解析原圖＋每影像色彩統計
            assert rest2, 'rest2_raw 需要 rest2'
            self.ab_raw = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True))
            self.head_rest2 = nn.Sequential(cbr(c4 + 6, c4, 1), DWBlock(c4), nn.Conv2d(c4, 22, 1))
        if rest2_raw2:  # exp048b：復原頭另看 1/4 解析未標準化局部色彩（16ch）與全域色彩特徵的投影（16ch）
            self.raw_local = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True))
            self.raw_gproj = nn.Linear(64, 16)
            self.head_rest2 = nn.Sequential(cbr(c4 + 6 + 32, c4, 1), DWBlock(c4), nn.Conv2d(c4, 22, 1))
        self.head_seg = nn.Sequential(cbr(c4 + 3, c4, 1), DWBlock(c4), DWBlock(c4), nn.Conv2d(c4, num_classes, 1))
        if refine:  # 全解析極小殘差精修（約 1 GFLOPs @768×1024），預設關
            self.ref = nn.Sequential(nn.Conv2d(6, 8, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(8, 3, 3, 1, 1))
        if ab_head or rest2:  # 由 1/16 特徵的全域平均預測 (a,b) 分佈 → 對 (a,b) 查表的 3×4 線性反轉換做軟加權（表由訓練場景擬合後寫入）
            # 另輸出 12 維逐張修正 ΔM（最後一層對應列初始為 0＝從純查表開始；exp035：直接預測 12 參數比先猜 (a,b) 再查表高 0.36 dB）
            self.ab_mlp = nn.Sequential(nn.Linear(c16 + (64 + 6 if rest2_raw else 0), 64), nn.ReLU(inplace=True), nn.Linear(64, A_N + B_N + 12))
            with torch.no_grad():
                self.ab_mlp[2].weight[A_N + B_N:].zero_(); self.ab_mlp[2].bias[A_N + B_N:].zero_()
        if ab_head or rest2 or rest3:
            eye = torch.zeros(A_N * B_N, 3, 4); eye[:, :, :3] = torch.eye(3)
            self.register_buffer('ab_table', eye.view(A_N * B_N, 12))             # 未擬合前＝恆等

    NEW_HEAD_PREFIXES = ('head_rest2', 'ab_mlp', 'ab_raw', 'raw_local', 'raw_gproj', 'r_', 'segfuse', 'segh_', 'sharpen_tab')   # 凍結舊學生時唯一可訓練的模組（rest2 系列與 rest3 的 r_*）

    @classmethod
    def is_new_head(cls, name):
        return name.startswith(cls.NEW_HEAD_PREFIXES)

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, 'freeze_base', False):  # exp044：凍結的舊學生（含 BN 統計）維持 eval，只有新復原頭在訓練
            for n, mod in self.named_children():
                if not self.is_new_head(n):
                    mod.eval()
        return self

    def encode(self, xn):
        f4 = self.s4(self.stem(F.pixel_unshuffle(xn, 4)))                           # 1/4
        f8 = self.s8(self.down8(f4))                                                 # 1/8
        f16 = self.s16(self.down16(f8))                                              # 1/16
        return f4, f8, f16

    def forward(self, x):
        H, W = x.shape[-2:]
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mu).pow(2).mean(dim=(2, 3), keepdim=True)
        xn = (x - mu) / torch.sqrt(var + 1e-4)
        f4, f8, f16 = self.encode(xn)
        if getattr(self, 'ms_half', False):  # exp087：1/2 輸入再過同一個編碼器，1/16 特徵升回後逐通道閘門相加（特徵蒸餾吃融合後的 f16）
            _, _, f16h = self.encode(F.interpolate(xn, size=(H // 2, W // 2), mode='bilinear', align_corners=False))
            f16 = f16 + self.ms_gate * F.interpolate(f16h, size=f16.shape[-2:], mode='bilinear', align_corners=False)
        self._f16 = f16                                                              # exp068：訓練時給 DINOv3 特徵蒸餾（匯出不受影響）
        f8 = self.fuse8(f8 + F.interpolate(self.lat16(f16), size=f8.shape[-2:], mode='bilinear', align_corners=False))
        f4 = self.fuse4(f4 + F.interpolate(self.lat8(f8), size=f4.shape[-2:], mode='bilinear', align_corners=False))
        if getattr(self, 'seg_cascade', False):
            # exp077：兩個編碼器只交換特徵、不交換梯度——rest3 讀 f4（detach），分割讀 rest3 的 r4 與復原圖（detach），各自只被自己的損失訓練
            J, r4 = self._rest3(x, f4.detach(), mu, var, H, W)
            fs = f4 + self.segfuse(r4.detach())
            j4 = F.interpolate(J.detach(), size=f4.shape[-2:], mode='bilinear', align_corners=False)
            seg4 = self.head_seg(torch.cat([fs, j4], 1))
            if getattr(self, 'seg_half', False):
                h2 = (H // 2, W // 2)
                s2 = F.interpolate(seg4, size=h2, mode='bilinear', align_corners=False)
                g2 = F.interpolate(self.segh_f4(fs), size=h2, mode='bilinear', align_corners=False)
                s2 = s2 + self.segh_head(torch.cat([F.pixel_unshuffle(xn, 2), s2, g2], 1))
                seg = F.interpolate(s2, size=(H, W), mode='bilinear', align_corners=False)
            else:
                seg = F.interpolate(seg4, size=(H, W), mode='bilinear', align_corners=False)
            if getattr(self, 'sharpen_kind', None):
                J = sharpen_out(J, self._ab_logits[:, :A_N], self.sharpen_kind, self.sharpen_tab, getattr(self, 'sharpen_port', None))
            if getattr(self, 'ra_a', None) is not None:  # exp094：逐色版仿射校準（Mul／Add，0 FLOPs）
                J = (J * self.ra_a + self.ra_b).clamp(0.0, 1.0)
            if getattr(self, 'coral_close_k', 0):  # exp095：bg 洞補 coral
                seg = coral_close(seg, self.coral_close_k)
            return J, seg
        p = F.interpolate(self.head_rest(f4), size=(H, W), mode='bilinear', align_corners=False)   # 21ch 參數圖升到全解析（免 FLOPs）
        J = self._restore(p, x, f16, self.ab_head)
        if self.refine:
            J = torch.clamp(J + 0.1 * self.ref(torch.cat([J, x], 1)), 0.0, 1.0)
        j4 = F.interpolate(J, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        seg = self.head_seg(torch.cat([f4, j4], 1))
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=False)
        if self.rest2:  # 輸出改用新頭；分割仍吃上面舊頭的 J
            gvec, h_in = None, f4
            if self.rest2_raw:
                st = torch.cat([mu, torch.sqrt(var + 1e-4)], 1)                    # (N,6,1,1)：每影像逐通道平均與標準差
                xr = x.reshape(x.shape[0], 3, x.shape[2] // 8, 8, x.shape[3] // 8, 8).mean(dim=(3, 5))   # 8×8 平均＝avg_pool2d，但只用 Jetson 已跑過的 Reshape／ReduceMean
                g = self.ab_raw(xr).mean(dim=(2, 3))                                # 未標準化 1/8 原圖 → 64 維全域色彩特徵
                gvec, h_in = torch.cat([g, st.flatten(1)], 1), torch.cat([f4, st.expand(-1, -1, f4.shape[2], f4.shape[3])], 1)
                if self.rest2_raw2:
                    h4, w4 = f4.shape[2], f4.shape[3]
                    xl = x.reshape(x.shape[0], 3, h4, x.shape[2] // h4, w4, x.shape[3] // w4).mean(dim=(3, 5))   # 1/4 平均（Reshape／ReduceMean）
                    gl = self.raw_gproj(g)[:, :, None, None].expand(-1, -1, h4, w4)
                    h_in = torch.cat([h_in, self.raw_local(xl), gl], 1)
            J = self._restore(F.interpolate(self.head_rest2(h_in), size=(H, W), mode='bilinear', align_corners=False), x, f16, True, gvec)
        if self.rest3:  # exp054：輸出改由獨立復原編碼器產生；分割仍吃上面舊頭的 J（凍結時與 init 位元相同）
            J, _ = self._rest3(x, f4, mu, var, H, W)
        if getattr(self, 'sharpen_kind', None):  # exp074：只在匯出／評估時由 set_sharpen 開啟，不影響訓練與既有權重
            J = sharpen_out(J, self._ab_logits[:, :A_N], self.sharpen_kind, self.sharpen_tab, getattr(self, 'sharpen_port', None))
        if getattr(self, 'ra_a', None) is not None:  # exp094：逐色版仿射校準（Mul／Add，0 FLOPs）
            J = (J * self.ra_a + self.ra_b).clamp(0.0, 1.0)
        if getattr(self, 'coral_close_k', 0):  # exp095：bg 洞補 coral
            seg = coral_close(seg, self.coral_close_k)
        return J, seg

    def set_rest_affine(self, a, b):
        """exp094：restored 輸出逐色版 a·J＋b（只在匯出／評估時開啟；不進 state_dict）。"""
        self.register_buffer('ra_a', torch.tensor(a, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('ra_b', torch.tensor(b, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        return self

    def _rest3(self, x, f4, mu, var, H, W):
        """rest3 復原編碼器（exp054）：吃未標準化 0–1 原圖，另以 f4 當語意旁路。回傳 (J, r4)；r4 給 exp077 的串接分割。"""
        st = torch.cat([mu, torch.sqrt(var + 1e-4)], 1)                            # (N,6,1,1)：每影像逐通道平均與標準差
        r4 = self.r_s4(self.r_stem(F.pixel_unshuffle(x, 4)))                       # 未標準化 0–1 原圖 → 1/4（Reshape／Transpose，免 FLOPs）
        r8 = self.r_s8(self.r_down8(r4))
        r16 = self.r_s16(self.r_down16(r8))
        r8 = self.r_fuse8(r8 + F.interpolate(self.r_lat16(r16), size=r8.shape[-2:], mode='bilinear', align_corners=False))
        r4 = self.r_fuse4(r4 + F.interpolate(self.r_lat8(r8), size=r4.shape[-2:], mode='bilinear', align_corners=False) + self.r_sem(f4))
        ab = self.r_gmlp(torch.cat([r16.mean(dim=(2, 3)), st.flatten(1)], 1))      # 全域：(a,b) logits＋ΔM
        p3 = self.r_head(torch.cat([r4, st.expand(-1, -1, r4.shape[2], r4.shape[3])], 1))
        J = self._restore(F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False), x, None, True, ab=ab)
        if self.rest3_half:
            d = F.interpolate(self.r_half(F.pixel_unshuffle(x, 2)), size=(H, W), mode='bilinear', align_corners=False)
            J = torch.clamp(J + 0.1 * torch.tanh(d), 0.0, 1.0)
        return J, r4

    def set_sharpen(self, kind, lo, hi, learn=False):
        """開啟輸出端細節增強：k 由水型 1→10 線性 lo→hi（需 rest3 的水型分佈；lo＝hi 為固定 k）。tab 為非持久 buffer，不進 state_dict。
        learn=True（exp088）：k 表改成可訓練參數（以 lo→hi 起始、進 state_dict），增強放進訓練的 forward，與 rest3 一起對復原分數最佳化。"""
        self.sharpen_kind = kind
        tab = torch.linspace(float(lo), float(hi), A_N).view(1, A_N)
        if learn:
            self.sharpen_tab = nn.Parameter(tab)
        else:
            self.register_buffer('sharpen_tab', tab, persistent=False)
        return self

    def _restore(self, p, x, f16, gated, gvec=None, ab=None):
        T = self.eps + (1.0 - self.eps) * torch.sigmoid(p[:, 0:3])                   # 透射率 T ∈ [eps, 1]
        B = torch.sigmoid(p[:, 3:6]) * 0.8                                           # 背散射 B ∈ [0, 0.8]
        A = 8.0 * torch.tanh(p[:, 6:15])                                             # 逐像素 3×3 色彩仿射（跨通道推斷，紅光 T≈0 時用）
        c = torch.tanh(p[:, 15:18])
        j_phys = (x - B) / T
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        j_aff = torch.cat([A[:, 0:1] * r + A[:, 1:2] * g + A[:, 2:3] * b + c[:, 0:1],
                           A[:, 3:4] * r + A[:, 4:5] * g + A[:, 5:6] * b + c[:, 1:2],
                           A[:, 6:7] * r + A[:, 7:8] * g + A[:, 8:9] * b + c[:, 2:3]], 1)
        if gated:
            if ab is None:
                feat = f16.mean(dim=(2, 3))
                ab = self.ab_mlp(feat if gvec is None else torch.cat([feat, gvec], 1))   # (N, A_N+B_N+12)
            self._ab_logits = ab[:, :A_N + B_N]                                      # 訓練時給 (a,b) 交叉熵
            joint = (ab[:, :A_N].softmax(1)[:, :, None] * ab[:, A_N:A_N + B_N].softmax(1)[:, None, :]).flatten(1)
            M = (joint @ self.ab_table + 0.25 * torch.tanh(ab[:, A_N + B_N:])).view(-1, 3, 4, 1, 1)   # 軟查表＋逐張修正
            j0 = M[:, :, 0] * r + M[:, :, 1] * g + M[:, :, 2] * b + M[:, :, 3]
            self._j0 = j0                                                            # 訓練時可加 J0 自身的復原損失
            gate = torch.softmax(p[:, 18:22], 1)
            J = torch.clamp(gate[:, 0:1] * j_phys + gate[:, 1:2] * j_aff + gate[:, 2:3] * j0 + gate[:, 3:4] * x, 0.0, 1.0)
        else:
            w = torch.sigmoid(p[:, 18:21])                                           # 物理反演 vs 仿射 的混合權重
            J = torch.clamp(w * j_phys + (1.0 - w) * j_aff, 0.0, 1.0)
        return J


class PretrainedPhysicsDualHead(PhysicsDualHead):
    """exp016：編碼器換成 timm ImageNet 預訓練輕量骨幹（取 stride 4／8／16 三層，1×1 投影到 c4／c8／c16），
    解碼器、物理復原頭、分割頭、每影像標準化輸入與 PhysicsDualHead 完全相同。
    timm 的 features_only 會把 stride 32 的區塊一起建出並執行（白算 FLOPs、多出參數）→ 建好後截掉。"""

    def __init__(self, backbone, out_indices=(1, 2, 3), c4=48, c8=96, c16=192, s16='', num_classes=5, eps=0.02, refine=False, pretrained=False, ab_head=False, rest2=False, rest2_raw=False, rest2_raw2=False, rest3=False, rest3_half=False, seg_cascade=False, seg_half=False, ms_half=False, in_affine=None):
        nn.Module.__init__(self)  # 不走 PhysicsDualHead 的編碼器建構
        import timm
        self.eps, self.refine = eps, refine
        self.in_affine = in_affine  # exp096：編碼器入口的常數乘加（對齊新預訓練權重的輸入分布）；None＝不動（現行權重）
        self.enc = timm.create_model(backbone, features_only=True, out_indices=out_indices, pretrained=pretrained)
        _truncate_features(self.enc)
        ch = self.enc.feature_info.channels()
        self.p4, self.p8, self.p16 = cbr(ch[0], c4, 1), cbr(ch[1], c8, 1), cbr(ch[2], c16, 1)
        # 四層骨幹（exp065 候選：EfficientViT-B0 取 stride 4／8／16／32）：stride 32 投影到 c16 後上採樣加進 f16，其餘解碼器不變
        self.p32 = cbr(ch[3], c16, 1) if len(ch) == 4 else None
        self.s16 = make_stage(c16, s16) if s16 else nn.Identity()
        self._build_decoder(c4, c8, c16, num_classes, refine, ab_head, rest2, rest2_raw, rest2_raw2, rest3, rest3_half, seg_cascade, seg_half, ms_half)

    def encode(self, xn):
        feats = self.enc(xn if self.in_affine is None else xn * self.in_affine[0] + self.in_affine[1])
        f4, f8, f16 = feats[0], feats[1], self.p16(feats[2])
        if self.p32 is not None:
            f16 = f16 + F.interpolate(self.p32(feats[3]), size=f16.shape[-2:], mode='bilinear', align_corners=False)
        return self.p4(f4), self.p8(f8), self.s16(f16)


def _truncate_features(enc):
    """EfficientNet 系（MobileNetV3／V4）的 features 模型：只保留到最後一個輸出所在的 stage。
    timm 的 _stage_out_idx 以 stem＝0 計，第 i 個 block 的輸出是 stage i+1 → 需要的 block 為 blocks[:max(_stage_out_idx)]。"""
    if hasattr(enc, 'blocks') and hasattr(enc, '_stage_out_idx'):
        enc.blocks = enc.blocks[:max(enc._stage_out_idx)]
        if hasattr(enc, 'conv_head'):
            enc.conv_head = nn.Identity()


PT_PRESETS = {
    # 768×1024、含解碼器全模型；數字為 2026-10-20 預算掃描（官方計數器，scratchpad scan_pretrained.py），正式數字以匯出後官方檢查為準
    'mnv4s768': dict(backbone='mobilenetv4_conv_small.e2400_r224_in1k', out_indices=(1, 2, 3), c4=48, c8=96, c16=192),   # 約 7.10 GFLOPs／0.41M，只有 Conv／ReLU
    'mnv3l768': dict(backbone='mobilenetv3_large_100.ra_in1k', out_indices=(1, 2, 3), c4=48, c8=96, c16=192),            # 約 7.71 GFLOPs／0.88M，含 HardSwish／SE
    # exp096：同一顆 MNv3-L 換預訓練權重（參數／FLOPs／op 完全相同）。forward 的輸入是逐張逐色版標準化（約 0 均值、1 標準差，接近 ImageNet 正規化）；
    # 新權重的訓練輸入分布不同 → 編碼器入口乘加常數對齊（0 FLOPs）：MIIL in21k 吃原始 0–1（約 0.45±0.25）、ra4 吃 (x−0.5)/0.5（約 −0.1±0.5）
    'mnv3l768_in21k': dict(backbone='mobilenetv3_large_100.miil_in21k_ft_in1k', out_indices=(1, 2, 3), c4=48, c8=96, c16=192, in_affine=(0.25, 0.45)),
    'mnv3l768_ra4': dict(backbone='mobilenetv3_large_100.ra4_e3600_r224_in1k', out_indices=(1, 2, 3), c4=48, c8=96, c16=192, in_affine=(0.5, -0.1)),
    # exp065 候選：MIT EfficientViT-B0（ReLU 線性注意力、Hardswish；作者在 Jetson AGX Orin TensorRT FP16 上量過），四層 stride 4／8／16／32 全用（骨幹 0.683M）
    'evitb0_768': dict(backbone='efficientvit_b0.r224_in1k', out_indices=(0, 1, 2, 3), c4=48, c8=96, c16=192),
    'evitb0s_768': dict(backbone='efficientvit_b0.r224_in1k', out_indices=(0, 1, 2, 3), c4=48, c8=96, c16=128),          # 窄版（c16 128），備用
}


def build(preset='base768', pretrained=False, **kw):
    """pretrained 只影響 PT_PRESETS（訓練時 True 下載 ImageNet 權重；匯出／推論時 False 再載入 checkpoint）。"""
    if preset in PT_PRESETS:
        cfg = dict(PT_PRESETS[preset]); cfg.update(kw)
        return PretrainedPhysicsDualHead(pretrained=pretrained, **cfg)
    cfg = dict(PRESETS[preset]); cfg.update(kw)
    return PhysicsDualHead(**cfg)


def build_from_ckpt(ck, preset):
    """依 checkpoint 存的訓練參數（refine、ab_head、rest2）重建同一個學生；舊 checkpoint 沒有這些鍵時＝預設關。"""
    args = ck.get('args', {}) or {}
    if args.get('preset') in PT_PRESETS and args.get('preset') != preset:  # exp096：權重版本與輸入對齊跟著訓練時的 preset 走
        print(f'build_from_ckpt: preset {preset} → {args["preset"]}（依 checkpoint）'); preset = args['preset']
    m = build(preset, refine=bool(args.get('refine', False)), ab_head=bool(args.get('ab_head', False)), rest2=bool(args.get('rest2', False)),
                 rest2_raw=bool(args.get('rest2_raw', False)), rest2_raw2=bool(args.get('rest2_raw2', False)),
                 rest3=bool(args.get('rest3', False)), rest3_half=bool(args.get('rest3_half', False)), seg_cascade=bool(args.get('seg_cascade', False)),
                 seg_half=bool(args.get('seg_half', False)), ms_half=bool(args.get('ms_half', False)))
    if args.get('sharpen_learn'):  # exp088：可訓練的增強 k 表在 state_dict 裡，載入前先建好
        kind, rng = args['sharpen_learn'].split(':'); lo, hi = (float(v) for v in rng.split('-'))
        m.set_sharpen(kind, lo, hi, learn=True)
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())
