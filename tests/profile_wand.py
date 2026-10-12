# -*- coding: utf-8 -*-
"""魔棒耗时剖析：定位读窗 / label / polygonize / 简化 各阶段耗时。"""
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform
from rasterio import features as rio_features
from scipy import ndimage

PATH = r"d:\UNET\Sentinel2_Zhangzi_Jul_2023.tif"
TOL = 35


def t(label, t0):
    dt = time.time() - t0
    print(f"  {label:<28} {dt*1000:8.0f} ms")
    return time.time()


def main():
    ds = rasterio.open(PATH)
    W, H = ds.width, ds.height
    ix, iy = W // 2, H // 2
    thresh = TOL * 2.55
    stats = None   # 真实代码用预计算的拉伸统计

    # 两遍窗口（与 magic_wand 一致）
    for size in (1200, 3600):
        w, h = min(size, W), min(size, H)
        col0 = min(max(ix - w // 2, 0), W - w)
        row0 = min(max(iy - h // 2, 0), H - h)
        win = Window(col0, row0, w, h)
        print(f"窗口 {w}x{h}:")
        t0 = time.time()
        data = ds.read(window=win, indexes=[3, 2, 1])
        t0 = t("读窗", t0)

        # stretch（模拟真实代码：预计算统计，仅按窗口映射）
        if stats is None:
            sub = data[:, ::4, ::4]
            stats = [(float(np.percentile(sub[i], 2)), float(np.percentile(sub[i], 98))) for i in range(3)]
        out = np.empty(data.shape, dtype=np.uint8)
        for i in range(3):
            b = data[i]
            lo, hi = stats[i]
            out[i] = np.clip((b.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        t0 = t("拉伸", t0)

        seed = out[:, iy - row0, ix - col0].astype(np.int16)
        diff = np.abs(out.astype(np.int16) - seed[:, None, None]).max(axis=0)
        valid = diff <= thresh
        t0 = t("diff+valid", t0)

        lbl, _ = ndimage.label(valid, structure=np.ones((3, 3), dtype=np.uint8))
        m = lbl == lbl[iy - row0, ix - col0]
        t0 = t(f"label ({int(m.sum())} px)", t0)

        # 形态学开运算去 1px 噪声
        m2 = ndimage.binary_opening(m, structure=np.ones((3, 3), dtype=bool))
        t0 = t("binary_opening", t0)

        wt = window_transform(win, ds.transform)
        best_a = -1
        nring = 0
        nverts = 0
        for geom, _v in rio_features.shapes(m2.astype(np.uint8), mask=m2, transform=wt, connectivity=8):
            nring += 1
            ring = geom["coordinates"][0]
            nverts = max(nverts, len(ring))
            a = abs(sum(ring[i][0] * ring[(i+1) % len(ring)][1] - ring[(i+1) % len(ring)][0] * ring[i][1]
                        for i in range(len(ring)))) / 2
            if a > best_a:
                best_a = a
        t0 = t(f"polygonize (rings={nring}, maxverts={nverts})", t0)
        print()
        if m[:, 0].any() or m[:, -1].any() or m[0, :].any() or m[-1, :].any():
            print("  触边 -> 升级窗口重算")
        else:
            break
    ds.close()


if __name__ == "__main__":
    main()
