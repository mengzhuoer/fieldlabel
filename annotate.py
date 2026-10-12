# -*- coding: utf-8 -*-
"""annotate.py - 耕图 FieldLabel 遥感影像多类别标注工具

在 v1（田块多边形标注）基础上的强化版本：

新增能力
    1. 网页内文件浏览器：无需命令行，直接在页面里浏览磁盘、切换要标注的影像
    2. 多类别标注：类别可增删改、改名改色，多边形/笔刷都可指定类别
    3. 涂抹式笔刷标注：像画笔一样涂抹生成语义标签图，支持橡皮擦、笔刷大小
    4. 自动保存：矢量与标签图在编辑后自动落盘，无需手动点击保存

输出（与既有 outputs/label_*.npy 格式一致）
    <影像名>.parcels.geojson   矢量田块（带 cls 类别字段）
    <影像名>.label.npy         uint8 语义标签图（0=背景，1..N=类别），自动保存
    <影像名>_label.tif         按需导出的 GeoTIFF 标签图
    <影像名>_label_stats.csv   按需导出的类别统计

依赖：rasterio, Pillow, numpy
用法：python annotate.py [影像或目录] [--port 8765] [--no-browser] [--local-only]
"""

import argparse
import io
import json
import math
import os
import re
import socket
import string
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio import features as rio_features

# --------------------------------------------------------------------------
# 路径 / 配置
# --------------------------------------------------------------------------

APP_NAME = "FieldLabel"
LOADED_FROM = None


def app_dir():
    """可写目录：打包后为 exe 所在目录，否则为脚本目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def res_dir():
    """只读资源目录（index.html 等）。"""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


CONFIG_PATH = os.path.join(app_dir(), "annotator_config.json")

DEFAULT_CLASSES = [
    {"id": 1, "name": "耕地", "color": "#4ade80"},
    {"id": 2, "name": "林地", "color": "#166534"},
    {"id": 3, "name": "草地", "color": "#a3e635"},
    {"id": 4, "name": "水体", "color": "#38bdf8"},
    {"id": 5, "name": "建设用地", "color": "#f87171"},
    {"id": 6, "name": "裸地/其他", "color": "#d1a054"},
]

RASTER_EXTS = (".tif", ".tiff", ".img", ".vrt", ".jp2", ".png", ".jpg", ".jpeg", ".bmp", ".dat")

_cfg_lock = threading.Lock()


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_config():
    cfg = _read_json(CONFIG_PATH, {})
    if not isinstance(cfg, dict):
        cfg = {}
    classes = cfg.get("classes")
    if not isinstance(classes, list) or not classes:
        classes = [dict(c) for c in DEFAULT_CLASSES]
    # 归一化
    clean, used = [], set()
    for c in classes:
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        if cid <= 0 or cid > 255 or cid in used:
            continue
        used.add(cid)
        clean.append({
            "id": cid,
            "name": str(c.get("name") or f"类别{cid}")[:24],
            "color": str(c.get("color") or "#888888"),
        })
    if not clean:
        clean = [dict(c) for c in DEFAULT_CLASSES]
    cfg["classes"] = clean
    cfg.setdefault("recent", [])
    return cfg


def save_config(cfg):
    with _cfg_lock:
        _write_json_atomic(CONFIG_PATH, cfg)


def hex_to_rgb(h):
    h = (h or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except Exception:
        return 136, 136, 136


# --------------------------------------------------------------------------
# 场景
# --------------------------------------------------------------------------

def build_ndvi_lut():
    t = np.linspace(0.0, 1.0, 256)
    c0 = np.array([165, 145, 115])
    c1 = np.array([230, 220, 120])
    c2 = np.array([20, 105, 35])
    lut = np.zeros((256, 3), dtype=np.uint8)
    half = 128
    for ch in range(3):
        lut[:half, ch] = (c0[ch] + (c1[ch] - c0[ch]) * (t[:half] * 2)).astype(np.uint8)
        lut[half:, ch] = (c1[ch] + (c2[ch] - c1[ch]) * ((t[half:] - 0.5) * 2)).astype(np.uint8)
    return lut


NDVI_LUT = build_ndvi_lut()


def build_ndwi_lut():
    """NDWI 三段色带：陆地 -> 过渡 -> 水体（蓝）。"""
    t = np.linspace(0.0, 1.0, 256)
    c0 = np.array([170, 160, 140])       # 陆地
    c1 = np.array([140, 190, 210])       # 过渡
    c2 = np.array([25, 95, 190])         # 水体
    lut = np.zeros((256, 3), dtype=np.uint8)
    half = 128
    for ch in range(3):
        lut[:half, ch] = (c0[ch] + (c1[ch] - c0[ch]) * (t[:half] * 2)).astype(np.uint8)
        lut[half:, ch] = (c1[ch] + (c2[ch] - c1[ch]) * ((t[half:] - 0.5) * 2)).astype(np.uint8)
    return lut


NDWI_LUT = build_ndwi_lut()


# --------------------------------------------------------------------------
# 几何工具（魔棒轮廓简化）
# --------------------------------------------------------------------------

def _shoelace(ring):
    """平面多边形面积（shoelace），ring 可为闭合或开放。"""
    s = 0.0
    n = len(ring)
    if n < 3:
        return 0.0
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _rdp(points, eps):
    """Douglas-Peucker 折线简化（迭代版）。points 为开放折线 [(x, y), ...]。"""
    if len(points) < 3 or eps <= 0:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = points[i]
        bx, by = points[j]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        dmax, idx = -1.0, -1
        for k in range(i + 1, j):
            px, py = points[k]
            if norm == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                d = abs(dy * px - dx * py + bx * ay - by * ax) / norm
            if d > dmax:
                dmax, idx = d, k
        if dmax > eps and idx > 0:
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return [p for p, k in zip(points, keep) if k]


# --------------------------------------------------------------------------
# 图像显示调整 / 多光谱表达式（会话级，可在页面里实时调）
# --------------------------------------------------------------------------

DEFAULT_ADJ = {
    "brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "gamma": 1.0,
    "sharpen": 0.0, "wb": 0, "lo": 2, "hi": 98, "expr": "",
}
ADJ = dict(DEFAULT_ADJ)

_EXPR_OK = re.compile(r"^[A-Za-z0-9_().,\s+\-*/]*$")


def apply_adjust(rgb, gains=None):
    """uint8 (H,W,3) 显示调整：白平衡 -> gamma -> 亮度/对比度 -> 饱和度 -> 锐化。"""
    a = ADJ
    x = rgb.astype(np.float32)
    if gains is not None and a.get("wb"):
        x = x * np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    g = float(a.get("gamma", 1.0))
    if abs(g - 1.0) > 1e-3:
        x = np.clip(x, 0, 255)
        x = 255.0 * np.power(x / 255.0, 1.0 / g)
    b = float(a.get("brightness", 1.0))
    c = float(a.get("contrast", 1.0))
    if abs(b - 1.0) > 1e-3 or abs(c - 1.0) > 1e-3:
        x = (x - 128.0) * c + 128.0 * b
    s = float(a.get("saturation", 1.0))
    if abs(s - 1.0) > 1e-3:
        gray = x.mean(axis=2, keepdims=True)
        x = gray + (x - gray) * s
    x = np.clip(x, 0, 255)
    sh = float(a.get("sharpen", 0.0))
    if sh > 1e-3:
        pad = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blur = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
                pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]) / 9.0
        x = x + (x - blur) * sh * 3.0
    return np.clip(x, 0, 255).astype(np.uint8)


def sanitize_adj(payload):
    """只接受白名单内的调整项，按类型钳制。"""
    out = dict(ADJ)
    if not isinstance(payload, dict):
        return out
    for k, kind in (("brightness", "f"), ("contrast", "f"), ("saturation", "f"),
                    ("gamma", "f"), ("sharpen", "f"), ("wb", "b"),
                    ("lo", "i"), ("hi", "i"), ("expr", "s")):
        if k not in payload:
            continue
        v = payload[k]
        try:
            if kind == "f":
                lo = 0.0 if k == "sharpen" else 0.05
                out[k] = max(lo, min(4.0, float(v)))
            elif kind == "b":
                out[k] = 1 if v in (1, True, "1") else 0
            elif kind == "i":
                out[k] = max(0, min(100, int(v)))
            else:
                out[k] = str(v)[:400]
        except (TypeError, ValueError):
            continue
    if out["hi"] - out["lo"] < 1:
        out["hi"] = min(100, out["lo"] + 1)
    expr = out["expr"].strip()
    if expr and not _EXPR_OK.match(expr):
        raise ValueError("表达式含非法字符")
    out["expr"] = expr
    return out


def eval_expr(arr, count, expr):
    """在波段数组上求值：b1..bN 对应波段 1..N，可用 np 与少量数学函数。"""
    if not expr:
        raise ValueError("表达式为空")
    if not _EXPR_OK.match(expr) or "__" in expr:
        raise ValueError("表达式含非法字符")
    env = {
        "np": np, "abs": abs, "min": np.minimum, "max": np.maximum,
        "sqrt": np.sqrt, "log": np.log, "exp": np.exp, "where": np.where,
        "clip": np.clip, "nan_to_num": np.nan_to_num,
    }
    for i in range(count):
        env["b%d" % (i + 1)] = arr[i].astype(np.float32)
    res = eval(expr, {"__builtins__": {}}, env)          # noqa: S307 受白名单约束
    out = np.asarray(res, dtype=np.float32)
    if out.ndim == 0:
        out = np.full(arr.shape[1:], float(out), dtype=np.float32)
    if out.shape != arr.shape[1:]:
        raise ValueError("表达式输出尺寸与影像不符")
    return out


class RasterScene:
    """一个已打开的栅格：元信息、拉伸统计、瓦片渲染、标签图读写。"""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.ds = rasterio.open(path)
        self.width = self.ds.width
        self.height = self.ds.height
        self.count = self.ds.count
        self.crs = self.ds.crs
        self.transform = self.ds.transform
        self.res = self.ds.res
        self.nodata = self.ds.nodata
        t = self.transform
        self.affine = [t.a, t.b, t.c, t.d, t.e, t.f]
        self.is_geographic = bool(self.crs and self.crs.is_geographic)
        self.epsg = self.crs.to_epsg() if self.crs else None
        self.descriptions = list(self.ds.descriptions or ())
        self.fill = int(self.nodata) if self.nodata is not None else 0
        self._ds_lock = threading.Lock()          # 必须先于 _compute_stretch 创建
        self._stretch = self._compute_stretch()

        # ---- 标签图（涂抹结果）----
        self._mask_lock = threading.RLock()
        self._pyr = {}
        self._lut = np.zeros((256, 4), dtype=np.uint8)
        self._counts = None
        self._mask_dirty = False
        self._mask_save_timer = None
        self.undo_stack = []
        self.redo_stack = []
        self.stroke_count = 0
        self.mask_rev = 0              # 标签图修订号（笔画/撤销/清空 +1）
        self.user_stats = {}           # client -> {name, n, px, by_cls:{cls:px}}
        self.last_saved = None
        self.mask = self._load_mask()

    # -- 拉伸 --------------------------------------------------------------
    def _compute_stretch(self, sample=768):
        """2%-98%（可调 lo/hi）百分位拉伸，另存各波段均值供白平衡用。"""
        out_w = min(sample, self.width)
        out_h = max(1, int(round(self.height * out_w / self.width)))
        lo_pct = float(ADJ.get("lo", 2))
        hi_pct = float(ADJ.get("hi", 98))
        with self._ds_lock:
            try:
                data = self.ds.read(out_shape=(self.count, out_h, out_w),
                                    boundless=True, fill_value=self.fill)
            except Exception:
                data = self.ds.read()
        stats = []
        self._means = []
        for i in range(self.count):
            b = data[i].astype(np.float64)
            valid = np.isfinite(b) & (b != self.fill)
            if valid.sum() < 16:
                valid = np.isfinite(b)
            if valid.sum() < 16:
                stats.append((0.0, 1.0))
                self._means.append(1.0)
                continue
            lo, hi = np.percentile(b[valid], (lo_pct, hi_pct))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = 0.0, 1.0
            if hi - lo < 1e-6:
                hi = lo + 1.0
            stats.append((float(lo), float(hi)))
            self._means.append(float(np.mean(b[valid])))
        return stats

    def wb_gains(self):
        """灰度世界白平衡增益（真彩色 R,G,B 三通道，基于抽样均值）。"""
        if not getattr(self, "_wb", None):
            m = list(getattr(self, "_means", []) or [])
            if len(m) >= 3:
                rgb_means = [m[2], m[1], m[0]]          # R, G, B（真彩色顺序）
                avg = sum(rgb_means) / 3.0
                if avg > 1e-9:
                    self._wb = [float(np.clip(avg / max(v, 1e-9), 0.4, 2.5))
                                for v in rgb_means]
                else:
                    self._wb = [1.0, 1.0, 1.0]
            else:
                self._wb = [1.0, 1.0, 1.0]
        return self._wb

    def recompute_stretch(self):
        """拉伸范围（lo/hi）或表达式变化后调用。"""
        self._stretch = self._compute_stretch()
        self._expr_stats = None
        if ADJ.get("expr"):
            self._expr_stats = self._compute_expr_stats()

    def _compute_expr_stats(self, sample=768):
        out_w = min(sample, self.width)
        out_h = max(1, int(round(self.height * out_w / self.width)))
        with self._ds_lock:
            try:
                data = self.ds.read(out_shape=(self.count, out_h, out_w),
                                    boundless=True, fill_value=self.fill)
            except Exception:
                data = self.ds.read()
        try:
            res = eval_expr(data, self.count, ADJ.get("expr", ""))
        except Exception:
            return (0.0, 1.0)
        r = res[np.isfinite(res)]
        if r.size < 16:
            return (0.0, 1.0)
        lo, hi = np.percentile(r, (float(ADJ.get("lo", 2)), float(ADJ.get("hi", 98))))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            lo, hi = float(np.min(r)), float(np.max(r)) + 1e-6
        return (float(lo), float(hi))

    def close(self):
        try:
            self.flush_mask()          # 仅在确有改动时落盘，避免产生空标签文件
        except Exception:
            pass
        try:
            with self._ds_lock:        # 等在途读取完成后再关，避免并发访问崩溃
                self.ds.close()
        except Exception:
            pass

    # -- 标签图 ------------------------------------------------------------
    @property
    def mask_path(self):
        stem, _ = os.path.splitext(self.path)
        return stem + ".label.npy"

    def _load_mask(self):
        p = self.mask_path
        if os.path.isfile(p):
            try:
                arr = np.load(p)
                if arr.shape == (self.height, self.width):
                    return np.array(arr, dtype=np.uint8, copy=True, order="C")
                print(f"[annotate] 标签图尺寸不匹配，已忽略: {p} {arr.shape} != {(self.height, self.width)}")
            except Exception as e:
                print(f"[annotate] 读取标签图失败 {p}: {e}")
        return np.zeros((self.height, self.width), dtype=np.uint8)

    def set_classes(self, classes):
        lut = np.zeros((256, 4), dtype=np.uint8)
        for c in classes:
            cid = int(c["id"])
            if 0 < cid < 256:
                r, g, b = hex_to_rgb(c.get("color"))
                lut[cid] = (r, g, b, 150)
        self._lut = lut

    def invalidate(self):
        self._pyr.clear()
        self._counts = None

    def mask_counts(self):
        if self._counts is None:
            with self._mask_lock:
                self._counts = np.bincount(self.mask.reshape(-1), minlength=256)[:256]
        return self._counts

    # -- 笔画绘制 ----------------------------------------------------------
    def _draw_stroke(self, arr, stroke, clip=None):
        """把一条笔画画进 arr，返回 (x0, y0, x1, y1, before_patch)。

        画笔采用超采样覆盖渲染：先在 S 倍分辨率上画“线段+端点圆”，缩回
        原尺寸后按 50% 覆盖阈值判定是否着色——圆形更圆、斜线更顺滑，
        同时保持标签图的类别值不被插值污染（擦除同样走该覆盖逻辑）。

        clip 给定时绘制范围与之求交（用于按人撤销时的区域重放，
        避免覆盖该区域之外他人后续笔画的修改）。
        """
        pts = stroke.get("points") or []
        if not pts:
            return None
        size = int(round(float(stroke.get("size", 8))))
        size = max(1, min(400, size))
        val = 0 if stroke.get("erase") else int(stroke.get("cls", 1))
        val = max(0, min(255, val))
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        pad = size
        x0 = max(0, int(math.floor(min(xs) - pad)))
        y0 = max(0, int(math.floor(min(ys) - pad)))
        x1 = min(self.width, int(math.ceil(max(xs) + pad)) + 1)
        y1 = min(self.height, int(math.ceil(max(ys) + pad)) + 1)
        if clip is not None:
            x0 = max(x0, int(clip[0]))
            y0 = max(y0, int(clip[1]))
            x1 = min(x1, int(clip[2]))
            y1 = min(y1, int(clip[3]))
        if x1 <= x0 or y1 <= y0:
            return None
        sub = arr[y0:y1, x0:x1]
        before = np.array(sub, copy=True)
        w, h = x1 - x0, y1 - y0
        area = w * h
        # 超采样倍数按块面积自适应，兼顾质量与大笔画性能
        if area <= 65536:
            S = 4
        elif area <= 4194304:
            S = 2
        else:
            S = 1
        cov = Image.new("L", (w * S, h * S), 0)
        dr = ImageDraw.Draw(cov)
        local = [((float(p[0]) - x0) * S, (float(p[1]) - y0) * S) for p in pts]
        r = size * S / 2.0
        if len(local) == 1:
            px, py = local[0]
            dr.ellipse([px - r, py - r, px + r, py + r], fill=255)
        else:
            lw = max(1, int(round(size * S)))
            dr.line(local, fill=255, width=lw, joint="curve")
            for px, py in (local[0], local[-1]):
                dr.ellipse([px - r, py - r, px + r, py + r], fill=255)
        if S > 1:
            cov = cov.resize((w, h), Image.BILINEAR)
        m = np.asarray(cov, dtype=np.uint8) >= 127
        sub[m] = val
        arr[y0:y1, x0:x1] = sub
        return (x0, y0, x1, y1, before)

    def _ensure_writable(self):
        if (not self.mask.flags.writeable) or (not self.mask.flags.c_contiguous):
            self.mask = np.array(self.mask, dtype=np.uint8, copy=True, order="C")

    def apply_stroke(self, stroke, record=True, client="", name=""):
        """应用一条笔画；stroke 记录归属人，供按人撤销与统计。"""
        with self._mask_lock:
            self._ensure_writable()
            patch = self._draw_stroke(self.mask, stroke)
            if patch is None:
                return False
            if record:
                stroke = dict(stroke)
                stroke["client"] = client or ""
                stroke["name"] = name or ""
                x0, y0, x1, y1, before_patch = patch
                after = self.mask[y0:y1, x0:x1]
                if stroke.get("erase"):
                    # 擦除统计：bbox 内被清零的原非零像素
                    changed = int(np.count_nonzero(
                        (before_patch != 0) & (after == 0)))
                else:
                    # 净新增：本笔画着色且非“重涂已有同类像素”
                    cls_v = int(stroke.get("cls", 1))
                    changed = int(np.count_nonzero(
                        (after == cls_v) & (before_patch != cls_v)))
                stroke["_px"] = changed
                self.undo_stack.append((stroke, patch))
                if len(self.undo_stack) > 200:
                    self.undo_stack.pop(0)
                self.redo_stack.clear()
                self.stroke_count += 1
                self.mask_rev += 1
                if client:
                    st = self.user_stats.setdefault(
                        client, {"name": name or client, "n": 0, "px": 0, "by_cls": {}})
                    if name:
                        st["name"] = name[:24]
                    st["n"] += 1
                    st["px"] += changed
                    cls_k = str(int(stroke.get("cls", 1))) if not stroke.get("erase") else "erase"
                    st["by_cls"][cls_k] = st["by_cls"].get(cls_k, 0) + changed
            self.invalidate()
            self._mask_dirty = True
        self.schedule_mask_save()
        return True

    # -- 按人撤销：从栈里摘出目标笔画，恢复其快照后重放其上的笔画 -----
    def _replay_entry(self, entry, region=None):
        """在当前 mask 上按原顺序重放一条历史条目，返回其新 patch。

        region=None 时全图重放（redo 场景）；给定 region 时仅重放该区域
        （按人撤销场景）——区域外保持现状，避免覆盖他人后续笔画的修改。
        """
        stroke, _old = entry
        if region is None:
            if stroke.get("_clear"):
                cls = stroke.get("cls")
                before = np.array(self.mask, copy=True)
                if cls is None:
                    self.mask[:] = 0
                else:
                    self.mask[self.mask == int(cls)] = 0
                return (0, 0, self.width, self.height, before)
            if stroke.get("_bake"):
                feats = stroke.get("features")
                if not feats:
                    return None
                before = np.array(self.mask, copy=True)
                self._bake_apply(feats, stroke.get("cls", 1))
                return (0, 0, self.width, self.height, before)
            return self._draw_stroke(self.mask, stroke)

        rx0, ry0, rx1, ry1 = (int(region[0]), int(region[1]),
                              int(region[2]), int(region[3]))
        if stroke.get("_clear"):
            cls = stroke.get("cls")
            sub = self.mask[ry0:ry1, rx0:rx1]
            before = np.array(sub, copy=True)
            if cls is None:
                sub[:] = 0
            else:
                sub[sub == int(cls)] = 0
            return (rx0, ry0, rx1, ry1, before)
        if stroke.get("_bake"):
            feats = stroke.get("features")
            if not feats:
                return None
            before = np.array(self.mask[ry0:ry1, rx0:rx1], copy=True)
            guard = np.array(self.mask, copy=True)      # 保护区域外不被改写
            self._bake_apply(feats, stroke.get("cls", 1))
            m = np.ones(self.mask.shape, dtype=bool)
            m[ry0:ry1, rx0:rx1] = False
            self.mask[m] = guard[m]
            return (rx0, ry0, rx1, ry1, before)
        return self._draw_stroke(self.mask, stroke, clip=region)

    def _replay_merge(self, ent, region):
        """重放一条后续条目并合成其 before 快照。

        合成规则：old patch 在 region 之外的取值仍然有效（该区域时间线未被
        本次撤销改变），region 之内改用重放时捕获的最新状态——两者取并，
        保证后续任何一次撤销恢复到正确的“无目标笔画”状态。
        不与 region 相交的条目原样保留（mask 与 patch 均未受影响）。
        """
        stroke, old_patch = ent
        fresh = self._replay_entry(ent, region)
        if fresh is None:
            return ent
        fx0, fy0, fx1, fy1, fbefore = fresh
        ox0, oy0, ox1, oy1, obefore = old_patch
        ix0, iy0 = max(fx0, ox0), max(fy0, oy0)
        ix1, iy1 = min(fx1, ox1), min(fy1, oy1)
        if ix0 < ix1 and iy0 < iy1:
            obefore[iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] = \
                fbefore[iy0 - fy0:iy1 - fy0, ix0 - fx0:ix1 - fx0]
        return (stroke, (ox0, oy0, ox1, oy1, obefore))

    def _find_last(self, stack, client, own_only):
        """own_only 时只找 client 自己的最近一条；否则取栈顶。"""
        if not stack:
            return -1
        if not own_only:
            return len(stack) - 1
        for i in range(len(stack) - 1, -1, -1):
            if (stack[i][0].get("client") or "") == client:
                return i
        return -1

    def mask_undo(self, client=None):
        """撤销。client 给定时只撤销该用户自己的最近一条笔画。"""
        own_only = bool(client)
        with self._mask_lock:
            self._ensure_writable()
            i = self._find_last(self.undo_stack, client, own_only)
            if i < 0:
                return False
            if i == len(self.undo_stack) - 1:
                stroke, (x0, y0, x1, y1, before) = self.undo_stack.pop()
                self.mask[y0:y1, x0:x1] = before
            else:
                # 目标不在栈顶：恢复其快照后，将与其区域相交的后续条目
                # 按序裁剪重放，并合成各自的 before 快照（见 _replay_merge）
                stroke, (x0, y0, x1, y1, before) = self.undo_stack.pop(i)
                self.mask[y0:y1, x0:x1] = before
                region = (x0, y0, x1, y1)
                rest = self.undo_stack[i:]
                del self.undo_stack[i:]
                for ent in rest:
                    self.undo_stack.append(self._replay_merge(ent, region))
            self.redo_stack.append((stroke, (x0, y0, x1, y1, before)))
            if len(self.redo_stack) > 200:
                self.redo_stack.pop(0)
            self._bump_stats_undo(stroke)
            self.mask_rev += 1
            self.invalidate()
            self._mask_dirty = True
        self.schedule_mask_save()
        return True

    def mask_redo(self, client=None):
        own_only = bool(client)
        with self._mask_lock:
            self._ensure_writable()
            i = self._find_last(self.redo_stack, client, own_only)
            if i < 0:
                return False
            stroke, patch = self.redo_stack.pop(i)
            if stroke.get("_clear") or stroke.get("_bake"):
                np_ = self._replay_entry((stroke, patch))
                if np_ is None:
                    self.redo_stack.insert(i, (stroke, patch))   # 无法重放则原样放回
                    return False
                patch = np_
            else:
                np_ = self._draw_stroke(self.mask, stroke)
                if np_ is None:
                    return False
                patch = np_
            self.undo_stack.append((stroke, patch))
            if len(self.undo_stack) > 200:
                self.undo_stack.pop(0)
            self._bump_stats_redo(stroke)
            self.mask_rev += 1
            self.invalidate()
            self._mask_dirty = True
        self.schedule_mask_save()
        return True

    def mask_clear(self, cls=None, client="", name=""):
        with self._mask_lock:
            self._ensure_writable()
            before = np.array(self.mask, copy=True)
            if cls is None:
                self.mask[:] = 0
            else:
                self.mask[self.mask == int(cls)] = 0
            entry = {"points": [], "_clear": True, "cls": cls,
                     "client": client or "", "name": name or ""}
            self.undo_stack.append((entry,
                                    (0, 0, self.width, self.height, before)))
            if len(self.undo_stack) > 200:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self.mask_rev += 1
            if client and cls is None:
                st = self.user_stats.get(client)
                if st:
                    st["px"] = 0
                    st["by_cls"] = {}
            self.invalidate()
            self._mask_dirty = True
        self.schedule_mask_save()
        return True

    def _bump_stats_undo(self, stroke):
        client = stroke.get("client") or ""
        if not client:
            return
        st = self.user_stats.get(client)
        if not st:
            return
        px = int(stroke.get("_px") or 0)
        st["px"] = max(0, st["px"] - px)
        if stroke.get("_clear"):
            if stroke.get("cls") is None:
                st["px"] = 0
                st["by_cls"] = {}
            return
        if stroke.get("erase"):
            return
        k = str(int(stroke.get("cls", 1)))
        st["by_cls"][k] = max(0, st["by_cls"].get(k, 0) - px)
        st["n"] = max(0, st["n"] - 1)

    def _bump_stats_redo(self, stroke):
        client = stroke.get("client") or ""
        if not client:
            return
        st = self.user_stats.setdefault(
            client, {"name": stroke.get("name") or client, "n": 0, "px": 0, "by_cls": {}})
        px = int(stroke.get("_px") or 0)
        st["px"] += px
        if not stroke.get("erase") and not stroke.get("_clear"):
            k = str(int(stroke.get("cls", 1)))
            st["by_cls"][k] = st["by_cls"].get(k, 0) + px
        st["n"] += 1

    def user_mask_stats(self, client):
        """按人统计（像素为近似净值：扣除后被他人覆盖的可能未计）。"""
        with self._mask_lock:
            st = self.user_stats.get(client)
            return dict(st) if st else {"name": client, "n": 0, "px": 0, "by_cls": {}}

    def bake_polygons(self, features, default_cls=1, client="", name=""):
        """把多边形（按其 cls）栅格化进标签图，返回处理块数。"""
        if not features:
            return 0
        with self._mask_lock:
            self._ensure_writable()
            before = np.array(self.mask, copy=True)
            n = self._bake_apply(features, default_cls)
            if n:
                self.undo_stack.append((
                    {"points": [], "_bake": True, "features": features,
                     "cls": default_cls, "client": client or "",
                     "name": name or ""},
                    (0, 0, self.width, self.height, before)))
                if len(self.undo_stack) > 200:
                    self.undo_stack.pop(0)
                self.redo_stack.clear()
                self.mask_rev += 1
                self.invalidate()
                self._mask_dirty = True
        if n:
            self.schedule_mask_save()
        return n

    def _bake_apply(self, features, default_cls):
        """在 self.mask 上栅格化多边形（调用方须持 _mask_lock）。"""
        try:
            inv = ~self.transform
        except Exception:
            inv = None
        canvas = Image.fromarray(self.mask)
        draw = ImageDraw.Draw(canvas)
        n = 0
        for feat in features:
            geom = (feat or {}).get("geometry") or {}
            gtype = geom.get("type")
            props = feat.get("properties") or {}
            try:
                cls = int(props.get("cls", default_cls))
            except Exception:
                cls = int(default_cls)
            cls = max(0, min(255, cls))
            rings = []
            if gtype == "Polygon":
                rings = [geom.get("coordinates", [])]
            elif gtype == "MultiPolygon":
                rings = geom.get("coordinates", [])
            for poly in rings:
                if not poly:
                    continue
                ring = poly[0] if isinstance(poly[0], (list, tuple)) and poly and \
                    isinstance(poly[0][0], (list, tuple)) else poly
                px = []
                for c in ring:
                    if not isinstance(c, (list, tuple)) or len(c) < 2:
                        continue
                    lon, lat = float(c[0]), float(c[1])
                    if inv is not None:
                        col, row = inv * (lon, lat)
                    else:
                        col, row = lon, lat
                    px.append((float(col), float(row)))
                if len(px) >= 3:
                    draw.polygon(px, fill=cls)
                    n += 1
        self.mask = np.array(canvas, dtype=np.uint8, copy=True, order="C")
        return n

    # -- 自动保存 ----------------------------------------------------------
    def schedule_mask_save(self, delay=1.2):
        with self._mask_lock:
            if self._mask_save_timer is not None:
                try:
                    self._mask_save_timer.cancel()
                except Exception:
                    pass
            t = threading.Timer(delay, self.flush_mask)
            t.daemon = True
            self._mask_save_timer = t
            t.start()

    def flush_mask(self, force=False):
        with self._mask_lock:
            if not self._mask_dirty and not force:
                return None
            path = self.mask_path
            try:
                np.save(path, self.mask)
                self._mask_dirty = False
                self.last_saved = time.strftime("%H:%M:%S")
                return path
            except Exception as e:
                print(f"[annotate] 标签图保存失败: {e}")
                return None

    # -- 瓦片渲染 ----------------------------------------------------------
    def _read_window(self, z, x, y, size=256):
        k = 2.0 ** (-z)
        col0 = x * size * k
        row0 = y * size * k
        span = size * k
        W, H = self.width, self.height
        with self._ds_lock:
            inside = (col0 >= -1e-9 and row0 >= -1e-9 and
                      col0 + span <= W + 1e-9 and row0 + span <= H + 1e-9)
            north_up = abs(self.transform.b) < 1e-12 and abs(self.transform.d) < 1e-12
            if inside or north_up:
                try:
                    return self.ds.read(window=Window(col0, row0, span, span),
                                        boundless=True, fill_value=self.fill,
                                        out_shape=(self.count, size, size))
                except ZeroDivisionError:
                    pass
            # 旋转网格（如 90° 转出来的 a=0 仿射）：rasterio boundless 不可用，
            # 手工做最近邻填充。
            out = np.full((self.count, size, size), self.fill,
                          dtype=self.ds.dtypes[0])
            step = span / size
            cols = col0 + (np.arange(size) + 0.5) * step
            rows = row0 + (np.arange(size) + 0.5) * step
            ci = np.floor(cols).astype(np.int64)
            ri = np.floor(rows).astype(np.int64)
            m_c = (ci >= 0) & (ci < W)
            m_r = (ri >= 0) & (ri < H)
            if m_c.any() and m_r.any():
                cc = ci[m_c]
                rr = ri[m_r]
                cpos = np.where(m_c)[0]
                rpos = np.where(m_r)[0]
                sub = self.ds.read(window=Window(int(cc[0]), int(rr[0]),
                                                 int(cc[-1] - cc[0] + 1),
                                                 int(rr[-1] - rr[0] + 1)))
                vals = sub[:, rr - rr[0], :][:, :, cc - cc[0]]
                out[:, rpos[:, None], cpos[None, :]] = vals
            return out

    def _stretch_band(self, i, arr):
        lo, hi = self._stretch[i]
        v = (arr.astype(np.float32) - lo) / (hi - lo) * 255.0
        v = np.nan_to_num(v, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(v, 0, 255).astype(np.uint8)

    def render_tile(self, z, x, y, combo="true", band=1, size=256):
        data = self._read_window(z, x, y, size)
        gains = None
        if combo == "ndvi" and self.count >= 4:
            b8 = data[3].astype(np.float32)
            b4 = data[2].astype(np.float32)
            den = b8 + b4
            nd = np.where(np.abs(den) > 1e-6, (b8 - b4) / np.maximum(den, 1e-6), 0.0)
            t = np.clip((nd + 0.2) / 1.0, 0.0, 1.0)
            rgb = NDVI_LUT[(t * 255).astype(np.uint8)]
        elif combo == "ndwi" and self.count >= 4:
            b3 = data[1].astype(np.float32)            # 绿
            b8 = data[3].astype(np.float32)            # 近红外
            den = b3 + b8
            nd = np.where(np.abs(den) > 1e-6, (b3 - b8) / np.maximum(den, 1e-6), 0.0)
            t = np.clip((nd + 0.2) / 1.0, 0.0, 1.0)
            rgb = NDWI_LUT[(t * 255).astype(np.uint8)]
        elif combo == "ndbi" and self.count >= 5:
            sw = data[4].astype(np.float32)            # 短波红外
            b8 = data[3].astype(np.float32)
            den = sw + b8
            nd = np.where(np.abs(den) > 1e-6, (sw - b8) / np.maximum(den, 1e-6), 0.0)
            t = np.clip((nd + 0.2) / 1.0, 0.0, 1.0)
            rgb = NDVI_LUT[(t * 255).astype(np.uint8)]
        elif combo == "custom" and ADJ.get("expr"):
            if getattr(self, "_expr_stats", None) is None:
                self._expr_stats = self._compute_expr_stats()
            try:
                res = eval_expr(data, self.count, ADJ["expr"])
            except Exception:
                res = None
            if res is not None:
                lo, hi = self._expr_stats
                res = np.nan_to_num(res, nan=lo, posinf=hi, neginf=lo)
                g = np.clip((res - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
                rgb = np.stack([g, g, g], axis=-1)
            else:
                i = min(max(int(band), 1), self.count) - 1
                g = self._stretch_band(i, data[i])
                rgb = np.stack([g, g, g], axis=-1)
        elif combo == "false" and self.count >= 4:
            rgb = np.stack([self._stretch_band(3, data[3]),
                            self._stretch_band(2, data[2]),
                            self._stretch_band(1, data[1])], axis=-1)
        elif combo == "true" and self.count >= 3:
            gains = self.wb_gains()
            rgb = np.stack([self._stretch_band(2, data[2]),
                            self._stretch_band(1, data[1]),
                            self._stretch_band(0, data[0])], axis=-1)
        else:
            i = min(max(int(band), 1), self.count) - 1
            g = self._stretch_band(i, data[i])
            rgb = np.stack([g, g, g], axis=-1)
        rgb = apply_adjust(rgb, gains)
        img = Image.fromarray(rgb, "RGB")
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=False, compress_level=1)
        return buf.getvalue()

    def _pyr_level(self, L):
        with self._mask_lock:
            if L in self._pyr:
                return self._pyr[L]
            if L <= 0:
                img = Image.fromarray(self.mask)
                self._pyr[0] = img
                return img
            prev = self._pyr_level(L - 1)
            w = max(1, prev.width // 2)
            h = max(1, prev.height // 2)
            img = prev.resize((w, h), Image.NEAREST)
            self._pyr[L] = img
            return img

    def render_overlay(self, z, x, y, size=256, hide=None):
        """按瓦片输出带透明的类别着色图层。"""
        with self._mask_lock:
            L = max(0, -int(z))
            img = self._pyr_level(L)
            k = 2.0 ** (-z)
            col0 = x * size * k
            row0 = y * size * k
            span = size * k
            s = float(2 ** L)
            lx0, ly0, lspan = col0 / s, row0 / s, span / s
            cx0 = max(0, int(math.floor(lx0)))
            cy0 = max(0, int(math.floor(ly0)))
            cx1 = min(img.width, int(math.ceil(lx0 + lspan)))
            cy1 = min(img.height, int(math.ceil(ly0 + lspan)))
            if cx1 <= cx0 or cy1 <= cy0:
                return self._empty_overlay(size)
            crop = img.crop((cx0, cy0, cx1, cy1))
            dx0 = int(round((cx0 * s - col0) / span * size))
            dy0 = int(round((cy0 * s - row0) / span * size))
            dw = max(1, int(round((cx1 * s - col0) / span * size)) - dx0)
            dh = max(1, int(round((cy1 * s - row0) / span * size)) - dy0)
            small = crop.resize((dw, dh), Image.NEAREST)
            canvas = Image.new("L", (size, size), 0)
            canvas.paste(small, (dx0, dy0))
            idx = np.asarray(canvas, dtype=np.uint8)
            rgba = self._lut[idx]
            if hide:
                for h in hide:
                    rgba = np.where(idx[..., None] == int(h), 0, rgba)
                rgba = rgba.astype(np.uint8)
        out = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        out.save(buf, "PNG", optimize=False, compress_level=1)
        return buf.getvalue()

    @staticmethod
    def _empty_overlay(size=256):
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        buf = io.BytesIO()
        out.save(buf, "PNG", optimize=False, compress_level=1)
        return buf.getvalue()

    # -- 魔棒（按容差泛洪选择）-------------------------------------------
    def _rgb_window(self, win_o):
        """读取窗口并按拉伸后的真彩色显示值转 uint8 RGB（与页面所见一致）。"""
        key = (int(win_o.col_off), int(win_o.row_off), int(win_o.width), int(win_o.height))
        cached = getattr(self, "_rgb_cache", None)
        if cached is None:
            cached = self._rgb_cache = {}
        if key in cached:
            return cached[key]
        with self._ds_lock:
            if self.count >= 3:
                data = self.ds.read(window=win_o, indexes=[3, 2, 1])
            else:
                data = self.ds.read(window=win_o, indexes=[1])
        if data.shape[0] == 1:
            g = self._stretch_band(0, data[0])
            rgb = np.stack([g, g, g], axis=-1)
        else:
            rgb = np.stack([self._stretch_band(2, data[0]),
                            self._stretch_band(1, data[1]),
                            self._stretch_band(0, data[2])], axis=-1)
        # 与页面显示一致：应用当前显示调整（含白平衡）
        rgb = apply_adjust(rgb, self.wb_gains())
        if len(cached) >= 3:
            cached.pop(next(iter(cached)))      # 小容量 LRU
        cached[key] = rgb
        return rgb

    def magic_wand(self, x, y, tol=30, win=1200):
        """魔棒：在显示色上按容差泛洪，返回简化轮廓（图像像素坐标）。

        tol 为 1-100（对应单通道差 0-255），与 PS 的容差语义一致。
        计算在以种子为中心的窗口内进行；区域触到窗口边界时自动扩大窗口重算，
        仍触边则标记 partial（截断）。
        """
        ix, iy = int(round(float(x))), int(round(float(y)))
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            raise ValueError("魔棒取点超出影像范围")
        tol = max(1, min(100, int(tol)))
        thresh = tol * 2.55

        sizes = [win]
        if max(self.width, self.height) > win + 400:
            sizes.append(min(2400, max(self.width, self.height)))

        mask, wt, partial = None, None, False
        for size in sizes:
            w = min(int(size), self.width)
            h = min(int(size), self.height)
            col0 = min(max(ix - w // 2, 0), self.width - w)
            row0 = min(max(iy - h // 2, 0), self.height - h)
            win_o = Window(col0, row0, w, h)
            rgb = self._rgb_window(win_o)
            seed_rgb = rgb[iy - row0, ix - col0].astype(np.int16)

            try:
                from scipy import ndimage
                diff = np.abs(rgb.astype(np.int16) - seed_rgb).max(axis=2)
                valid = diff <= thresh
                lbl, _ = ndimage.label(valid, structure=np.ones((3, 3), dtype=np.uint8))
                sid = lbl[iy - row0, ix - col0]
                m = (lbl == sid) if sid else valid
                # 开运算去掉 1px 毛刺，显著减少轮廓顶点；
                # 保留与种子连通的分量，确保选区一定包含点击点
                if m.any():
                    m2 = ndimage.binary_opening(m, structure=np.ones((3, 3), dtype=bool))
                    l2, _ = ndimage.label(m2, structure=np.ones((3, 3), dtype=np.uint8))
                    sid2 = int(l2[iy - row0, ix - col0])
                    keep = (l2 == sid2) if sid2 else np.zeros_like(m)
                    if keep.any():
                        m = keep
            except Exception:        # 无 scipy 时退回 PIL 泛洪
                img = Image.fromarray(rgb)
                try:
                    ImageDraw.floodfill(img, (ix - col0, iy - row0), (0, 1, 2),
                                        thresh=thresh * 3)
                    m = np.all(np.asarray(img) == (0, 1, 2), axis=-1)
                except Exception:
                    m = np.zeros(rgb.shape[:2], dtype=bool)

            touched = False
            if col0 > 0 and m[:, 0].any():
                touched = True
            if col0 + w < self.width and m[:, -1].any():
                touched = True
            if row0 > 0 and m[0, :].any():
                touched = True
            if row0 + h < self.height and m[-1, :].any():
                touched = True

            mask, wt, partial = m, window_transform(win_o, self.transform), touched
            if not touched:
                break

        if mask is None or not mask.any():
            return {"polygon": [], "area_px": 0, "n": 0, "partial": partial,
                    "tol": tol, "window": [int(sizes[-1])] * 2}

        # 矢量化：GDAL polygonize 取最大连通域轮廓
        best, best_a = None, -1.0
        for geom, _val in rio_features.shapes(mask.astype(np.uint8), mask=mask,
                                              transform=wt, connectivity=8):
            ring = geom.get("coordinates") or []
            if not ring:
                continue
            a = _shoelace(ring[0])
            if a > best_a:
                best, best_a = ring[0], a
        if not best:
            return {"polygon": [], "area_px": 0, "n": 0, "partial": partial,
                    "tol": tol, "window": [int(sizes[-1])] * 2}

        # 地理坐标 -> 图像像素坐标
        try:
            inv = ~self.transform
        except Exception:
            inv = None
        px = []
        for c in best:
            if inv is not None:
                col, row = inv * (float(c[0]), float(c[1]))
            else:
                col, row = float(c[0]), float(c[1])
            px.append((float(col), float(row)))
        # 去掉首尾重复点
        if len(px) >= 2 and math.hypot(px[0][0] - px[-1][0], px[0][1] - px[-1][1]) < 1e-6:
            px.pop()
        # Douglas-Peucker 简化（逐步加大容差以控制顶点数）
        simp = px
        for eps in (1.0, 1.5, 2.5, 4.0, 7.0, 12.0):
            simp = _rdp(px, eps)
            if len(simp) <= 1500:
                break
        if len(simp) < 3:
            return {"polygon": [], "area_px": 0, "n": len(simp), "partial": partial,
                    "tol": tol, "window": [int(sizes[-1])] * 2}
        return {
            "polygon": [[round(p[0], 1), round(p[1], 1)] for p in simp],
            "area_px": int(round(_shoelace(simp))),
            "n": len(simp),
            "partial": bool(partial),
            "tol": tol,
            "window": [int(sizes[-1])] * 2,
        }

    # -- 导出 --------------------------------------------------------------
    def export_mask_tif(self, out_path=None):
        stem, _ = os.path.splitext(self.path)
        out_path = out_path or (stem + "_label.tif")
        profile = {
            "driver": "GTiff", "height": self.height, "width": self.width,
            "count": 1, "dtype": "uint8", "crs": self.crs,
            "transform": self.transform, "nodata": 0, "compress": "deflate",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(self.mask, 1)
        return out_path

    def export_stats_csv(self, classes, out_path=None):
        stem, _ = os.path.splitext(self.path)
        out_path = out_path or (stem + "_label_stats.csv")
        counts = self.mask_counts()
        px_area = abs(self.transform.a * self.transform.e) if self.transform else 0
        if self.is_geographic:
            cy = self.transform.f + self.transform.e * self.height / 2.0
            px_area = abs(self.transform.a * 111320 * math.cos(math.radians(cy))) * \
                abs(self.transform.e * 110540)
        total = int(self.mask.size)
        named = {int(c["id"]): c["name"] for c in classes}
        lines = ["class_id,class_name,pixels,ratio,area_ha"]
        for cid in sorted(named):
            n = int(counts[cid]) if cid < len(counts) else 0
            lines.append(f"{cid},{named[cid]},{n},{n / max(1, total):.6f},{n * px_area / 10000:.4f}")
        bg = int(counts[0]) if len(counts) else 0
        lines.append(f"0,background,{bg},{bg / max(1, total):.6f},{bg * px_area / 10000:.4f}")
        with open(out_path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(lines) + "\n")
        return out_path


# --------------------------------------------------------------------------
# 多用户协作状态
# --------------------------------------------------------------------------

class Collab:
    """在线用户、事件流与修订号（内存态，随进程生命周期）。

    - users: client_id -> {id, name, color, last(时间戳), cursor:[x,y,z,x0,y0,x1,y1]}
    - events: 最近若干条变更事件 {seq, type, client, name, data, t}
    - rev: 矢量修订号，每次保存 +1，供前端比对是否需要拉取
    """

    USER_TTL = 25.0          # 心跳超时（秒）视为离线
    MAX_EVENTS = 400

    def __init__(self):
        self.lock = threading.RLock()    # 可重入：list_users 剪枝时会再调 emit
        self.users = {}
        self.events = []
        self.seq = 0
        self.rev = 0

    # -- 用户存在感 --------------------------------------------------------
    def touch(self, client, name="", color="", cursor=None):
        if not client:
            return None
        with self.lock:
            u = self.users.get(client)
            now = time.time()
            if u is None:
                u = {"id": client, "name": name or client[:8],
                     "color": color or "#4f8cff", "first": now}
                self.users[client] = u
                joined = True
            else:
                joined = False
            if name:
                u["name"] = str(name)[:24]
            if color:
                u["color"] = str(color)[:24]
            if cursor is not None and isinstance(cursor, (list, tuple)) and len(cursor) >= 2:
                try:
                    u["cursor"] = [float(v) for v in cursor[:6]]
                except (TypeError, ValueError):
                    pass
            u["last"] = now
            return u, joined

    def leave(self, client):
        with self.lock:
            return self.users.pop(client, None)

    def list_users(self, now=None):
        now = now if now is not None else time.time()
        with self.lock:
            stale = [k for k, u in self.users.items()
                     if now - u.get("last", 0) > self.USER_TTL]
            for k in stale:
                u = self.users.pop(k, None)
                if u:
                    self.seq += 1
                    self.events.append({"seq": self.seq, "type": "leave",
                                        "client": k, "name": u.get("name", ""),
                                        "data": {}, "t": now})
            if len(self.events) > self.MAX_EVENTS:
                del self.events[:len(self.events) - self.MAX_EVENTS]
            return [dict(u) for u in self.users.values()]

    # -- 事件流 ------------------------------------------------------------
    def emit(self, etype, client="", name="", data=None):
        with self.lock:
            self.seq += 1
            ev = {"seq": self.seq, "type": etype, "client": client or "",
                  "name": name or "", "data": data or {}, "t": time.time()}
            self.events.append(ev)
            if len(self.events) > self.MAX_EVENTS:
                del self.events[:len(self.events) - self.MAX_EVENTS]
            return ev

    def events_since(self, n):
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 0
        with self.lock:
            if n <= 0:
                return list(self.events), self.seq
            return [e for e in self.events if e["seq"] > n], self.seq

    def bump_rev(self):
        with self.lock:
            self.rev += 1
            return self.rev

    def get_rev(self):
        with self.lock:
            return self.rev


# --------------------------------------------------------------------------
# 应用状态
# --------------------------------------------------------------------------

class AppState:
    def __init__(self, entry_path):
        self.lock = threading.Lock()
        self._gj_lock = threading.Lock()     # 矢量合并保存
        self.collab = Collab()          # 多用户协作状态（在线/事件/修订号）
        self.cfg = load_config()
        self.scene = None
        self.last_dir = os.path.abspath(entry_path) if entry_path else ""
        if entry_path and os.path.isfile(entry_path):
            self.last_dir = os.path.dirname(os.path.abspath(entry_path))
            self.scene = self._make_scene(entry_path)
        elif entry_path and os.path.isdir(entry_path):
            first = self.list_rasters(entry_path)
            if first:
                self.scene = self._make_scene(first[0])

    def _make_scene(self, path):
        sc = RasterScene(path)
        sc.set_classes(self.cfg["classes"])
        return sc

    @staticmethod
    def list_rasters(d):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return []
        return [os.path.join(d, n) for n in names if n.lower().endswith(RASTER_EXTS)]

    def open_raster(self, path):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with self.lock:
            new = self._make_scene(path)
            old = self.scene
            self.scene = new
            self.last_dir = os.path.dirname(path)
        if old:
            old.close()
        self.remember(path)
        return self.scene

    def remember(self, path):
        d = os.path.dirname(os.path.abspath(path))
        rec = self.cfg.setdefault("recent", [])
        rec = [r for r in rec if r != d]
        rec.insert(0, d)
        self.cfg["recent"] = rec[:12]
        try:
            save_config(self.cfg)
        except Exception:
            pass

    def set_classes(self, classes):
        self.cfg["classes"] = classes
        save_config(self.cfg)
        if self.scene:
            self.scene.set_classes(classes)

    # -- 矢量文件 ----------------------------------------------------------
    def geojson_path(self, path=None):
        p = path or (self.scene.path if self.scene else None)
        if not p:
            return None
        stem, _ = os.path.splitext(p)
        return stem + ".parcels.geojson"

    def load_geojson(self):
        p = self.geojson_path()
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"type": "FeatureCollection", "features": []}

    def save_geojson(self, fc):
        p = self.geojson_path()
        if not p:
            raise RuntimeError("no raster open")
        _write_json_atomic(p, fc)
        return p

    # -- 多人协作：矢量合并保存 ------------------------------------------
    @staticmethod
    def _feat_key(f):
        """要素唯一键：文本与田块 id 空间独立，故带类型前缀（与前端 fkey 一致）。"""
        p = (f or {}).get("properties") or {}
        kind = "t" if p.get("type") == "text" else "p"
        return "%s:%s" % (kind, p.get("id"))

    def merge_geojson(self, payload):
        """按人增量保存。

        mode="full"    : 整体替换（旧版客户端兼容）
        mode="merge"   : upsert(按 key 替换/追加) + remove(按 key 删除)
        返回 {path, rev, count}。upsert 的要素会盖上 owner/editor 溯源信息。
        """
        mode = str(payload.get("mode") or "full")
        client = str(payload.get("client") or "")
        name = str(payload.get("name") or "")[:24]
        with self._gj_lock:
            cur = self.load_geojson()
            feats = list(cur.get("features") or [])
            if mode == "merge":
                index = {self._feat_key(f): i for i, f in enumerate(feats)}
                now = int(time.time())
                for f in (payload.get("remove") or []):
                    k = self._feat_key(f) if isinstance(f, dict) else str(f)
                    i = index.pop(k, None)
                    if i is not None:
                        feats[i] = None
                        # 删除后重建索引中后续项会错位，故收集完再压平
                for f in (payload.get("upsert") or []):
                    if not isinstance(f, dict):
                        continue
                    props = f.setdefault("properties", {}) or {}
                    props["editor"] = name or client or props.get("editor", "")
                    props["mtime"] = now
                    k = self._feat_key(f)
                    i = index.get(k)
                    if i is None:
                        if not props.get("owner"):
                            props["owner"] = name or client
                        index[k] = len(feats)
                        feats.append(f)
                    else:
                        old_owner = (feats[i].get("properties") or {}).get("owner")
                        if not props.get("owner"):
                            props["owner"] = old_owner or props.get("editor", "")
                        feats[i] = f
                feats = [f for f in feats if f is not None]
            fc = {"type": "FeatureCollection", "features": feats}
            path = self.save_geojson(fc)
            rev = self.collab.bump_rev()
            return {"path": path, "rev": rev, "count": len(feats)}

    # -- 几何变换（裁剪/旋转/翻转）--------------------------------------
    def derive(self, kind, x=0, y=0, w=0, h=0):
        """写出派生影像 + 标签图并切换过去；原文件保持不变。

        像素映射（原 -> 新），与前端 mapping 保持一致：
          rot90   (顺时针90°): x' = H-1-y, y' = x
          rot90ccw(逆时针90°): x' = y,     y' = W-1-x
          rot180            : x' = W-1-x, y' = H-1-y
          fliph             : x' = W-1-x, y' = y
          flipv             : x' = x,     y' = H-1-y
          crop              : x' = x-x0,  y' = y-y0
        """
        sc = self.scene
        if not sc:
            raise RuntimeError("no raster open")
        from rasterio import Affine
        T = sc.transform
        W, H = sc.width, sc.height
        if kind == "crop":
            x = int(x); y = int(y); w = int(w); h = int(h)
            x = max(0, min(x, W - 8)); y = max(0, min(y, H - 8))
            w = max(8, min(w, W - x)); h = max(8, min(h, H - y))
            M = Affine(1, 0, x, 0, 1, y)
            new_w, new_h = w, h
            suffix = "_crop"
        elif kind == "rot90":
            # 逆映射 old = (y', H-1-x')，与 np.rot90(k=-1) 数据旋转一致
            M = Affine(0, 1, 0, -1, 0, H - 1)
            new_w, new_h = H, W
            suffix = "_rot90"
        elif kind == "rot90ccw":
            # 逆映射 old = (W-1-y', x')，与 np.rot90(k=1) 数据旋转一致
            M = Affine(0, -1, W - 1, 1, 0, 0)
            new_w, new_h = H, W
            suffix = "_rot90ccw"
        elif kind == "rot180":
            M = Affine(-1, 0, W - 1, 0, -1, H - 1)
            new_w, new_h = W, H
            suffix = "_rot180"
        elif kind == "fliph":
            M = Affine(-1, 0, W - 1, 0, 1, 0)
            new_w, new_h = W, H
            suffix = "_fliph"
        elif kind == "flipv":
            M = Affine(1, 0, 0, 0, -1, H - 1)
            new_w, new_h = W, H
            suffix = "_flipv"
        else:
            raise ValueError("unknown kind: %s" % kind)
        newT = T * M

        stem, _ = os.path.splitext(sc.path)
        out = stem + suffix + ".tif"
        n = 1
        while os.path.exists(out):
            n += 1
            out = "%s_%d.tif" % (stem + suffix, n)
            if n > 500:
                raise RuntimeError("派生文件命名失败")
        newstem = os.path.splitext(out)[0]
        mask_out = newstem + ".label.npy"

        profile = {"driver": "GTiff", "height": new_h, "width": new_w,
                   "count": sc.count, "dtype": sc.ds.dtypes[0], "crs": sc.crs,
                   "transform": newT, "nodata": sc.nodata,
                   "compress": "deflate", "ZLEVEL": 1, "BIGTIFF": "IF_SAFER"}
        with rasterio.open(out, "w", **profile) as dst:
            if kind == "crop":
                for r0 in range(0, h, 1024):
                    hh = min(1024, h - r0)
                    with sc._ds_lock:
                        arr = sc.ds.read(window=Window(x, y + r0, w, hh))
                    dst.write(arr, indexes=list(range(1, sc.count + 1)),
                              window=Window(0, r0, w, hh))
                with sc._mask_lock:
                    new_mask = np.array(sc.mask[y:y + h, x:x + w], copy=True)
            else:
                for i in range(1, sc.count + 1):
                    with sc._ds_lock:
                        arr = sc.ds.read(i)
                    if kind == "rot90":
                        arr = np.rot90(arr, -1)
                    elif kind == "rot90ccw":
                        arr = np.rot90(arr, 1)
                    elif kind == "rot180":
                        arr = np.rot90(arr, 2)
                    elif kind == "fliph":
                        arr = arr[:, ::-1]
                    else:                      # flipv
                        arr = arr[::-1, :]
                    dst.write(np.ascontiguousarray(arr), i)
                with sc._mask_lock:
                    m = sc.mask
                    if kind == "rot90":
                        new_mask = np.ascontiguousarray(np.rot90(m, -1))
                    elif kind == "rot90ccw":
                        new_mask = np.ascontiguousarray(np.rot90(m, 1))
                    elif kind == "rot180":
                        new_mask = np.ascontiguousarray(np.rot90(m, 2))
                    elif kind == "fliph":
                        new_mask = np.ascontiguousarray(m[:, ::-1])
                    else:
                        new_mask = np.ascontiguousarray(m[::-1, :])
        np.save(mask_out, new_mask)

        new_sc = self.open_raster(out)        # 切换场景（自动加载刚写的标签）
        return {
            "kind": kind, "path": out, "mask_path": mask_out,
            "orig": {"width": W, "height": H},
            "rect": ({"x": x, "y": y, "w": w, "h": h} if kind == "crop" else None),
            "info": _scene_info(new_sc), "mask": _mask_stats(new_sc),
        }


STATE = None
PAGE = None
_page_lock = threading.Lock()


def load_page():
    global PAGE
    with _page_lock:
        if PAGE is None:
            with open(os.path.join(res_dir(), "index.html"), "r", encoding="utf-8") as f:
                PAGE = f.read()
    return PAGE


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

SAFE_SEG = re.compile(r"[\x00-\x1f]")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = APP_NAME + "/2.0"

    def log_message(self, fmt, *args):
        pass

    # -- 基础工具 ----------------------------------------------------------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _q(self):
        return parse_qs(urlparse(self.path).query)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _scene(self):
        return STATE.scene

    # -- GET ---------------------------------------------------------------
    def do_GET(self):
        try:
            url = urlparse(self.path)
            q = self._q()
            if url.path in ("/", "/index.html"):
                self._send(200, load_page(), "text/html; charset=utf-8")
            elif url.path == "/info":
                self._info()
            elif url.path == "/browse":
                self._browse(q)
            elif url.path == "/tile":
                self._tile(q, overlay=False)
            elif url.path == "/overlay":
                self._tile(q, overlay=True)
            elif url.path == "/parcels":
                fc = STATE.load_geojson()
                if isinstance(fc, dict):
                    fc["rev"] = STATE.collab.get_rev()
                self._send(200, fc)
            elif url.path == "/users":
                self._users(q)
            elif url.path == "/maskval":
                self._maskval(q)
            elif url.path == "/maskinfo":
                self._maskinfo()
            elif url.path == "/classes":
                self._send(200, {"classes": STATE.cfg["classes"]})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    # -- POST --------------------------------------------------------------
    def do_POST(self):
        try:
            url = urlparse(self.path)
            payload = self._body()
            p = url.path
            if p == "/open":
                sc = STATE.open_raster(payload.get("path", ""))
                STATE.collab.emit("open",
                                  str(payload.get("client") or ""),
                                  str(payload.get("name") or ""),
                                  {"name": os.path.basename(sc.path)})
                self._send(200, {
                    "ok": True,
                    "info": _scene_info(sc),
                    "classes": STATE.cfg["classes"],
                    "labels_path": STATE.geojson_path(),
                    "mask": _mask_stats(sc),
                    "files": STATE.list_rasters(os.path.dirname(sc.path)),
                })
            elif p == "/save":
                mode = str(payload.get("mode") or "full")
                client = str(payload.get("client") or "")
                name = str(payload.get("name") or "")
                if mode == "merge":
                    res = STATE.merge_geojson(payload)
                    STATE.collab.emit("save", client, name,
                                      {"mode": "merge", "rev": res["rev"],
                                       "n": len(payload.get("upsert") or [])})
                    self._send(200, {"ok": True, **res})
                else:
                    feats = payload.get("features", [])
                    fc = {"type": "FeatureCollection",
                          "features": feats if isinstance(feats, list) else []}
                    path = STATE.save_geojson(fc)
                    rev = STATE.collab.bump_rev()
                    STATE.collab.emit("save", client, name,
                                      {"mode": "full", "rev": rev})
                    self._send(200, {"ok": True, "path": path, "rev": rev})
            elif p == "/users":
                self._post_users(payload)
            elif p == "/classes":
                self._set_classes(payload)
            elif p == "/adj":
                prev = dict(ADJ)
                ADJ.update(sanitize_adj(payload.get("adj", {})))
                sc = STATE.scene
                if ADJ.get("expr") and ADJ["expr"] != prev.get("expr"):
                    # 用微型数组先验证表达式可用（波段名/语法/输出形状）
                    nb = sc.count if sc else 4
                    try:
                        eval_expr(np.zeros((nb, 4, 4), dtype=np.float32), nb, ADJ["expr"])
                    except Exception as e:
                        ADJ["expr"] = prev.get("expr", "")
                        raise ValueError("表达式错误: %s" % e)
                if sc is not None:
                    if getattr(sc, "_rgb_cache", None):
                        sc._rgb_cache.clear()
                    if (ADJ.get("lo") != prev.get("lo") or
                            ADJ.get("hi") != prev.get("hi") or
                            ADJ.get("expr") != prev.get("expr")):
                        sc.recompute_stretch()
                self._send(200, {"ok": True, "adj": ADJ})
            elif p == "/derive":
                kind = payload.get("kind", "")
                res = STATE.derive(
                    kind,
                    payload.get("x", 0), payload.get("y", 0),
                    payload.get("w", 0), payload.get("h", 0))
                STATE.collab.emit("derive",
                                  str(payload.get("client") or ""),
                                  str(payload.get("name") or ""),
                                  {"kind": kind})
                self._send(200, {"ok": True, **res})
            elif p == "/wand":
                sc = self._require_scene()
                res = sc.magic_wand(payload.get("x", 0), payload.get("y", 0),
                                    tol=payload.get("tol", 30),
                                    win=int(payload.get("win", 1200) or 1200))
                self._send(200, {"ok": True, **res})
            elif p == "/stroke":
                sc = self._require_scene()
                client = str(payload.get("client") or "")
                name = str(payload.get("name") or "")
                ok = sc.apply_stroke({
                    "points": payload.get("points") or [],
                    "size": payload.get("size", 8),
                    "cls": payload.get("cls", 1),
                    "erase": bool(payload.get("erase", False)),
                }, client=client, name=name)
                if ok:
                    STATE.collab.emit("stroke", client, name, {
                        "cls": payload.get("cls", 1),
                        "erase": bool(payload.get("erase", False)),
                        "mask_rev": sc.mask_rev})
                self._send(200, {"ok": ok, "stats": _mask_stats(sc),
                                 "mask_rev": sc.mask_rev})
            elif p == "/mask_undo":
                sc = self._require_scene()
                client = str(payload.get("client") or "")
                ok = sc.mask_undo(client or None)
                if ok:
                    STATE.collab.emit("undo", client,
                                      str(payload.get("name") or ""),
                                      {"mask_rev": sc.mask_rev})
                self._send(200, {"ok": ok, "stats": _mask_stats(sc),
                                 "mask_rev": sc.mask_rev})
            elif p == "/mask_redo":
                sc = self._require_scene()
                client = str(payload.get("client") or "")
                ok = sc.mask_redo(client or None)
                if ok:
                    STATE.collab.emit("redo", client,
                                      str(payload.get("name") or ""),
                                      {"mask_rev": sc.mask_rev})
                self._send(200, {"ok": ok, "stats": _mask_stats(sc),
                                 "mask_rev": sc.mask_rev})
            elif p == "/mask_clear":
                sc = self._require_scene()
                cls = payload.get("cls")
                client = str(payload.get("client") or "")
                name = str(payload.get("name") or "")
                sc.mask_clear(None if cls in (None, "", -1) else int(cls),
                              client=client, name=name)
                STATE.collab.emit("clear", client, name,
                                  {"cls": cls, "mask_rev": sc.mask_rev})
                self._send(200, {"ok": True, "stats": _mask_stats(sc),
                                 "mask_rev": sc.mask_rev})
            elif p == "/bake":
                sc = self._require_scene()
                n = sc.bake_polygons(payload.get("features") or [], payload.get("cls", 1),
                                     client=str(payload.get("client") or ""),
                                     name=str(payload.get("name") or ""))
                if n:
                    STATE.collab.emit("bake",
                                      str(payload.get("client") or ""),
                                      str(payload.get("name") or ""),
                                      {"count": n, "mask_rev": sc.mask_rev})
                self._send(200, {"ok": True, "count": n, "stats": _mask_stats(sc)})
            elif p == "/export_tif":
                sc = self._require_scene()
                path = sc.export_mask_tif(payload.get("path") or None)
                self._send(200, {"ok": True, "path": path})
            elif p == "/export_csv":
                sc = self._require_scene()
                path = sc.export_stats_csv(STATE.cfg["classes"], payload.get("path") or None)
                self._send(200, {"ok": True, "path": path})
            elif p == "/flush":
                sc = self._require_scene()
                path = sc.flush_mask(force=True)
                self._send(200, {"ok": True, "path": path, "stats": _mask_stats(sc)})
            elif p == "/shutdown":
                self._send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    # -- 路由实现 ----------------------------------------------------------
    def _require_scene(self):
        sc = self._scene()
        if not sc:
            raise RuntimeError("no raster open")
        return sc

    def _set_classes(self, payload):
        classes = payload.get("classes") or []
        clean, used = [], set()
        for c in classes:
            try:
                cid = int(c.get("id"))
            except Exception:
                continue
            if cid <= 0 or cid > 255 or cid in used:
                continue
            used.add(cid)
            clean.append({"id": cid, "name": str(c.get("name") or f"类别{cid}")[:24],
                          "color": str(c.get("color") or "#888888")})
        if not clean:
            clean = [dict(c) for c in DEFAULT_CLASSES]
        STATE.set_classes(clean)
        STATE.collab.emit("classes",
                          str(payload.get("client") or ""),
                          str(payload.get("name") or ""), {})
        self._send(200, {"ok": True, "classes": clean})

    def _info(self):
        sc = self._scene()
        if not sc:
            self._send(200, {"scene": None, "classes": STATE.cfg["classes"],
                             "adj": ADJ,
                             "last_dir": STATE.last_dir,
                             "recent": STATE.cfg.get("recent", [])})
            return
        self._send(200, {
            "scene": _scene_info(sc),
            "classes": STATE.cfg["classes"],
            "adj": ADJ,
            "labels_path": STATE.geojson_path(),
            "mask_path": sc.mask_path,
            "mask": _mask_stats(sc),
            "last_dir": STATE.last_dir,
            "recent": STATE.cfg.get("recent", []),
        })

    def _maskinfo(self):
        sc = self._scene()
        if not sc:
            self._send(200, {"mask": None})
            return
        self._send(200, {"mask": _mask_stats(sc)})

    def _users(self, q):
        """GET /users?since=<seq>：在线用户 + 增量事件 + 矢量修订号。"""
        since = (q.get("since", ["0"])[0] or "0")
        users = STATE.collab.list_users()
        events, seq = STATE.collab.events_since(since)
        self._send(200, {"users": users, "events": events, "seq": seq,
                         "rev": STATE.collab.get_rev()})

    def _post_users(self, payload):
        """POST /users：身份注册/心跳/光标上报，新用户广播 join 事件。"""
        client = str(payload.get("client") or "").strip()[:64]
        if not client:
            self._send(400, {"error": "missing client id"})
            return
        if payload.get("action") == "leave":
            u = STATE.collab.leave(client)
            if u:
                STATE.collab.emit("leave", client, u.get("name", ""), {})
            self._send(200, {"ok": True, "users": STATE.collab.list_users(),
                             "seq": STATE.collab.seq,
                             "rev": STATE.collab.get_rev()})
            return
        res = STATE.collab.touch(
            client,
            name=str(payload.get("name") or "")[:24],
            color=str(payload.get("color") or "")[:24],
            cursor=payload.get("cursor"),
        )
        joined = bool(res and res[1])
        if joined:
            STATE.collab.emit("join", client, res[0].get("name", ""),
                              {"color": res[0].get("color", "")})
        users = STATE.collab.list_users()
        events, seq = STATE.collab.events_since(payload.get("since", 0))
        self._send(200, {"ok": True, "users": users, "events": events,
                         "seq": seq, "rev": STATE.collab.get_rev()})

    def _maskval(self, q):
        """GET /maskval?x=&y= 取像素类别（吸管）；?client= 按人统计。"""
        sc = self._scene()
        if not sc:
            self._send(200, {"cls": None, "stat": None})
            return
        client = (q.get("client", [""])[0] or "").strip()
        if "x" in q and "y" in q:
            try:
                x = int(float(q["x"][0]))
                y = int(float(q["y"][0]))
            except (ValueError, IndexError):
                self._send(400, {"error": "bad coords"})
                return
            if not (0 <= x < sc.width and 0 <= y < sc.height):
                self._send(200, {"x": x, "y": y, "cls": 0})
                return
            with sc._mask_lock:
                v = int(sc.mask[y, x])
            self._send(200, {"x": x, "y": y, "cls": v})
            return
        self._send(200, {"stat": sc.user_mask_stats(client),
                         "mask_rev": sc.mask_rev})

    def _browse(self, q):
        raw = (q.get("path", [""])[0] or "").strip().strip('"')
        show_all = q.get("all", ["0"])[0] == "1"
        if not raw:
            drives = ["%s:\\" % c for c in string.ascii_uppercase if os.path.isdir("%s:\\" % c)]
            self._send(200, {"path": "", "parent": None, "dirs":
                             [{"name": d, "path": d} for d in drives],
                             "files": [], "recent": STATE.cfg.get("recent", []),
                             "home": os.path.expanduser("~").replace("/", os.sep)})
            return
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isdir(path):
            self._send(400, {"error": f"目录不存在: {path}"})
            return
        dirs, files = [], []
        try:
            with os.scandir(path) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            dirs.append({"name": e.name, "path": e.path})
                        elif e.is_file(follow_symlinks=False):
                            low = e.name.lower()
                            if show_all or low.endswith(RASTER_EXTS):
                                try:
                                    size = e.stat().st_size
                                except OSError:
                                    size = 0
                                files.append({"name": e.name, "path": e.path, "size": size})
                    except OSError:
                        continue
        except PermissionError:
            self._send(403, {"error": f"无权限访问: {path}"})
            return
        dirs.sort(key=lambda d: d["name"].lower())
        files.sort(key=lambda f: f["name"].lower())
        parent = os.path.dirname(path.rstrip("\\/")) or None
        if parent and os.path.normcase(parent) == os.path.normcase(path):
            parent = None
        self._send(200, {"path": path, "parent": parent, "dirs": dirs, "files": files,
                         "recent": STATE.cfg.get("recent", []),
                         "home": os.path.expanduser("~").replace("/", os.sep)})

    def _tile(self, q, overlay=False):
        sc = self._scene()
        if not sc:
            self._send(404, {"error": "no raster"})
            return
        try:
            z = int(q.get("z", ["0"])[0])
            x = int(q.get("x", ["0"])[0])
            y = int(q.get("y", ["0"])[0])
            combo = (q.get("combo", ["true"])[0])[:8]
            band = int(q.get("band", ["1"])[0])
        except (ValueError, IndexError):
            self._send(400, {"error": "bad tile params"})
            return
        if abs(z) > 12 or abs(x) > 100000 or abs(y) > 100000:
            self._send(400, {"error": "tile out of range"})
            return
        if overlay:
            hide = [v for v in (q.get("hide", [""])[0] or "").split(",") if v.strip().isdigit()]
            png = sc.render_overlay(z, x, y, hide=[int(h) for h in hide] or None)
        else:
            png = sc.render_tile(z, x, y, combo=combo, band=band)
        self._send(200, png, "image/png")


def _scene_info(sc):
    return {
        "path": sc.path, "name": os.path.basename(sc.path),
        "width": sc.width, "height": sc.height, "count": sc.count,
        "dtype": sc.ds.dtypes[0],
        "crs": str(sc.crs) if sc.crs else None, "epsg": sc.epsg,
        "is_geographic": sc.is_geographic, "affine": sc.affine,
        "res": list(sc.res), "nodata": sc.nodata,
        "descriptions": sc.descriptions,
    }


def _mask_stats(sc):
    counts = sc.mask_counts()
    return {
        "path": sc.mask_path,
        "width": sc.width, "height": sc.height,
        "strokes": sc.stroke_count,
        "can_undo": len(sc.undo_stack) > 0,
        "can_redo": len(sc.redo_stack) > 0,
        "mask_rev": sc.mask_rev,
        "last_saved": getattr(sc, "last_saved", None),
        "counts": {str(i): int(counts[i]) for i in range(256) if counts[i]},
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _warmup():
    """后台预热 scipy，避免首次魔棒调用承担模块导入耗时。"""
    try:
        from scipy import ndimage
        ndimage.label(np.zeros((4, 4), dtype=np.uint8),
                      structure=np.ones((3, 3), dtype=np.uint8))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="耕图 FieldLabel - 遥感影像多类别标注工具")
    ap.add_argument("path", nargs="?", default=None, help="影像文件或目录（可留空，页面里再选）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0", help="绑定地址（默认 0.0.0.0，可局域网访问）")
    ap.add_argument("--local-only", action="store_true", help="仅绑定 127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    global STATE
    entry = args.path
    if entry is None:
        entry = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else os.getcwd()
    STATE = AppState(entry)

    print(f"[annotate] {APP_NAME} v1.0 (FieldLabel / 耕图)")
    print(f"[annotate] 配置  : {CONFIG_PATH}")
    if STATE.scene:
        print(f"[annotate] 影像  : {STATE.scene.path}")
        print(f"[annotate] 矢量  : {STATE.geojson_path()}")
        print(f"[annotate] 标签  : {STATE.scene.mask_path}")
    else:
        print(f"[annotate] 未预载影像，请在网页中点击「打开文件」选择")

    bind = "127.0.0.1" if args.local_only else args.host
    try:
        server = ThreadingHTTPServer((bind, args.port), Handler)
    except OSError as e:
        print(f"[annotate] 端口 {args.port} 不可用: {e}")
        raise SystemExit(1)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[annotate] 服务  : {url}   (Ctrl+C 退出)")
    threading.Timer(0.6, _warmup).start()          # 后台预热

    if not args.local_only:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            print(f"[annotate] 局域网: http://{ip}:{args.port}/   <- 其他电脑用这个")
        except OSError:
            pass
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if not ip.startswith("127."):
                    print(f"[annotate]         http://{ip}:{args.port}/")
        except OSError:
            pass

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[annotate] 退出中……")
    finally:
        server.server_close()
        if STATE.scene:
            STATE.scene.close()
        print("[annotate] 已保存并退出")


if __name__ == "__main__":
    main()
