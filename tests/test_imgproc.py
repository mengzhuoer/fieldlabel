# -*- coding: utf-8 -*-
"""图像处理后端测试：/adj 显示调整、多光谱组合、/derive 裁剪旋转。

用法: python tests/test_imgproc.py [port]
需先启动 annotate.py（建议指定一张 tif）。
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import rasterio

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8801
BASE = f"http://127.0.0.1:{PORT}"
PASS, FAIL = [], []
CREATED = []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (f"   {extra}" if extra and not cond else ""))


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def get(path, raw=False):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        d = r.read()
        return d if raw else json.loads(d.decode())


def tile(combo="true"):
    return np.asarray(Image.open(io.BytesIO(
        get(f"/tile?z=-3&x=1&y=1&combo={combo}", raw=True))).convert("RGB"))


def main():
    info = get("/info")
    sc = info["scene"]
    check("场景就绪", bool(sc), str(sc and sc["name"]))
    check("/info 返回 adj", "adj" in info, "")
    orig = (sc["width"], sc["height"], sc["name"])

    print("\n== 显示调整 ==")
    base = tile()
    j = post("/adj", {"adj": {"brightness": 1.6, "contrast": 1.2, "saturation": 1.5,
                              "gamma": 1.2, "sharpen": 1.0, "wb": 1}})
    check("POST /adj 接受参数", j.get("ok") and abs(j["adj"]["brightness"] - 1.6) < 1e-6, str(j.get("adj")))
    time.sleep(0.2)
    bright = tile()
    diff = np.abs(bright.astype(int) - base.astype(int)).mean()
    check("亮度/调整改变像素", diff > 1.0, f"mean diff={diff:.2f}")

    j = post("/adj", {"adj": {"lo": 5, "hi": 90}})
    check("拉伸范围可改", j.get("ok") and j["adj"]["lo"] == 5, "")
    time.sleep(0.2)
    t2 = tile()
    check("拉伸范围改变像素", np.abs(t2.astype(int) - bright.astype(int)).mean() > 0.3,
          f"{np.abs(t2.astype(int) - bright.astype(int)).mean():.2f}")

    j = post("/adj", {"adj": {"brightness": 1, "contrast": 1, "saturation": 1, "gamma": 1,
                              "sharpen": 0, "wb": 0, "lo": 2, "hi": 98}})
    check("重置参数", j.get("ok") and j["adj"]["brightness"] == 1, "")

    print("\n== 多光谱组合 ==")
    for combo in ("ndvi", "ndwi", "ndbi", "false", "gray"):
        png = get(f"/tile?z=-3&x=1&y=1&combo={combo}", raw=True)
        ok = png[:4] == b"\x89PNG" and len(png) > 300
        check(f"组合 {combo} 出图", ok, f"len={len(png)}")
    uniq = len(np.unique(tile("ndwi").reshape(-1, 3), axis=0))
    check("NDWI 色带非灰", uniq > 10, f"colors={uniq}")

    print("\n== 自定义波段表达式 ==")
    j = post("/adj", {"adj": {"expr": "(b4-b3)/(b4+b3)"}})
    check("合法表达式接受", j.get("ok") and j["adj"]["expr"] == "(b4-b3)/(b4+b3)", str(j.get("adj", {}).get("expr")))
    png = get("/tile?z=-3&x=1&y=1&combo=custom", raw=True)
    check("custom 组合出图", png[:4] == b"\x89PNG", "")
    try:
        post("/adj", {"adj": {"expr": "b7*__import__('os')"}})
        check("非法表达式拒绝", False, "未拒绝")
    except urllib.error.HTTPError as e:
        check("非法表达式拒绝", e.code == 500, f"HTTP {e.code}")
    j = post("/adj", {"adj": {"expr": ""}})
    check("清空表达式", j.get("ok") and j["adj"]["expr"] == "", "")

    print("\n== 裁剪 ==")
    cw, ch = 600, 400
    j = post("/derive", {"kind": "crop", "x": 100, "y": 80, "w": cw, "h": ch})
    if j.get("ok") is None and j.get("error"):
        check("裁剪执行", False, str(j))
    else:
        info2 = j["info"]
        CREATED.append(j["path"])
        CREATED.append(j["mask_path"])
        check("裁剪尺寸正确", (info2["width"], info2["height"]) == (cw, ch),
              f"{info2['width']}x{info2['height']}")
        # 新图 (0,0) 的地理坐标 == 原图 (100,80)
        inv_o = ~rasterio.Affine(*sc["affine"])
        lon0, lat0 = inv_o * (100, 80)   # 原图 100,80 -> 需要正算
        T0 = rasterio.Affine(*sc["affine"])
        lon1, lat1 = T0 * (100, 80)
        T1 = rasterio.Affine(*info2["affine"])
        lon2, lat2 = T1 * (0, 0)
        check("裁剪 georef 正确", abs(lon1 - lon2) < 1e-9 and abs(lat1 - lat2) < 1e-9,
              f"{lon1},{lat1} vs {lon2},{lat2}")
        check("裁剪标签文件生成", os.path.isfile(j["mask_path"]), str(j["mask_path"]))
        m = np.load(j["mask_path"])
        check("裁剪标签尺寸", m.shape == (ch, cw), str(m.shape))

    print("\n== 旋转 ==")
    orig_path = sc["path"]

    # 顺时针90°（从原图重新打开做基线）
    post("/open", {"path": orig_path})
    info = get("/info")["scene"]
    T0 = rasterio.Affine(*info["affine"])
    H0, W0 = info["height"], info["width"]
    j = post("/derive", {"kind": "rot90"})
    info2 = j["info"]
    CREATED.append(j["path"]); CREATED.append(j["mask_path"])
    T1 = rasterio.Affine(*info2["affine"])
    check("顺时针90° 尺寸互换", (info2["width"], info2["height"]) == (H0, W0),
          f"{info2['width']}x{info2['height']} 期望 {H0}x{W0}")
    # 映射 new(0,0) <-> old(0, H-1)；new(H-1,0) <-> old(0,0)
    lon1, lat1 = T0 * (0, H0 - 1)
    lon2, lat2 = T1 * (0, 0)
    check("顺时针90° 角点A georef", abs(lon1 - lon2) < 1e-9 and abs(lat1 - lat2) < 1e-9,
          f"{lon1},{lat1} vs {lon2},{lat2}")
    lon1, lat1 = T0 * (0, 0)
    lon2, lat2 = T1 * (H0 - 1, 0)
    check("顺时针90° 角点B georef", abs(lon1 - lon2) < 1e-9 and abs(lat1 - lat2) < 1e-9,
          f"{lon1},{lat1} vs {lon2},{lat2}")

    # 逆时针90°（重新打开原图）
    post("/open", {"path": orig_path})
    info = get("/info")["scene"]
    T0 = rasterio.Affine(*info["affine"])
    H0, W0 = info["height"], info["width"]
    j = post("/derive", {"kind": "rot90ccw"})
    info3 = j["info"]
    CREATED.append(j["path"]); CREATED.append(j["mask_path"])
    T3 = rasterio.Affine(*info3["affine"])
    check("逆时针90° 尺寸互换", (info3["width"], info3["height"]) == (H0, W0),
          f"{info3['width']}x{info3['height']}")
    # 映射 new(0,0) <-> old(W-1, 0)；new(0, W-1) <-> old(0,0)
    lon3, lat3 = T0 * (W0 - 1, 0)
    lon4, lat4 = T3 * (0, 0)
    check("逆时针90° 角点A georef", abs(lon3 - lon4) < 1e-9 and abs(lat3 - lat4) < 1e-9,
          f"{lon3},{lat3} vs {lon4},{lat4}")
    lon3, lat3 = T0 * (0, 0)
    lon4, lat4 = T3 * (0, W0 - 1)
    check("逆时针90° 角点B georef", abs(lon3 - lon4) < 1e-9 and abs(lat3 - lat4) < 1e-9,
          f"{lon3},{lat3} vs {lon4},{lat4}")

    print("\n== 瓦片继续可用 ==")
    png = get("/tile?z=-3&x=1&y=1", raw=True)
    check("切换后瓦片正常", png[:4] == b"\x89PNG", "")
    png = get("/overlay?z=-3&x=1&y=1", raw=True)
    check("切换后叠加层正常", png[:4] == b"\x89PNG", "")

    print("\n== 清理派生文件 ==")
    # 先切回原图（释放对派生文件的占用），再删除
    try:
        post("/open", {"path": orig_path})
    except Exception as e:
        print("   ! 切回原图失败:", e)
    for p in CREATED:
        try:
            if os.path.isfile(p):
                os.remove(p)
                print("   removed", os.path.basename(p))
        except OSError as e:
            print("   ! fail", p, e)

    print(f"\n===== 图像处理后端: {len(PASS)} 通过 / {len(FAIL)} 失败 =====")
    for f in FAIL:
        print("  失败:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
