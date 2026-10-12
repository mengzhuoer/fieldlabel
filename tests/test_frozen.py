# -*- coding: utf-8 -*-
"""对「打包后的 exe」做一次完整功能回归（服务已在运行，只发请求）。

用法：
    dist\FieldLabel\FieldLabel.exe <影像.tif> --port 8794 --local-only --no-browser
    python tests\test_frozen.py            # 端口默认 8794，可用 TA_PORT 覆盖
覆盖：核心 19 项 + 多人协作/画笔 12 项 ≈ 31 项。
"""
import io
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

PORT = int(os.environ.get("TA_PORT", 8794))
BASE = f"http://127.0.0.1:{PORT}"
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (f"   {extra}" if extra and not cond else ""))


def get(path, raw=False):
    with urllib.request.urlopen(BASE + path, timeout=90) as r:
        d = r.read()
        return r.status, (d if raw else json.loads(d.decode("utf-8"))), dict(r.headers)


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def main():
    print(f"== 打包版回归 (端口 {PORT}) ==")

    open_path = os.environ.get("TA_OPEN", "")
    if open_path:
        st, j = post("/open", {"path": open_path})
        print("   已载入:", j["info"]["name"])

    st, page, hd = get("/", raw=True)
    txt = page.decode("utf-8", "replace")
    check("首页可访问且为 FieldLabel", st == 200 and "耕图 FieldLabel" in txt and "classbar" in txt, f"len={len(page)}")
    check("首页包含自动保存与笔刷 UI", "自动保存" in txt and "brushSize" in txt and "田块→标签图" in txt)

    st, j, _ = get("/info")
    sc = j.get("scene") or {}
    img = os.path.splitext(sc.get("name", "img"))[0]
    check("场景信息", bool(sc.get("name")), str(sc.get("name")))
    check("默认类别 6 类", len(j.get("classes", [])) == 6, str([c["name"] for c in j.get("classes", [])]))

    st, j, _ = get("/browse?path=" + urllib.parse.quote(os.path.dirname(open_path) or r"d:\UNET"))
    check("文件浏览器可用", any(f["name"].endswith(".tif") for f in j.get("files", [])), str(len(j.get("files", []))))

    st, png, _ = get("/tile?z=-3&x=0&y=0", raw=True)
    check("底图瓦片", png[:4] == b"\x89PNG" and len(png) > 5000, f"len={len(png)}")

    st, j, _ = get("/maskinfo")
    mc = ((j.get("mask") or {}).get("counts")) or {}
    if mc:
        # 外置服务场景：上一轮 cleanup 删掉 label.npy 后，autosave 可能又把它
        # 复活（带历史笔画）——先全清，保证测试从干净 mask 开始
        post("/mask_clear", {})
        st, j, _ = get("/maskinfo")
        mc = ((j.get("mask") or {}).get("counts")) or {}
    check("起始 mask 干净", int(mc.get("1", 0)) == 0 and int(mc.get("2", 0)) == 0,
          f"初始类1={mc.get('1')} 类2={mc.get('2')} 路径={((j.get('mask') or {}).get('path'))}")

    st, j = post("/stroke", {"points": [[500, 500], [560, 520], [620, 500]], "size": 20, "cls": 1})
    c1 = int(j["stats"]["counts"].get("1", 0))
    check("涂抹落笔", j.get("ok") and c1 > 500, f"c1={c1}")
    st, j = post("/stroke", {"points": [[520, 520]], "size": 32, "cls": 2})
    c2 = int(j["stats"]["counts"].get("2", 0))
    check("点划（单击圆点）", c2 > 200, f"c2={c2}")

    st, png, _ = get("/overlay?z=-3&x=0&y=0", raw=True)
    ov = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    nz = int((ov[..., 3] > 0).sum())
    check("标签图层叠加渲染", nz > 0, f"非透明像素={nz}")

    st, j = post("/mask_undo", {})
    c2b = int(j["stats"]["counts"].get("2", 0))
    check("涂抹撤销", c2b == 0, f"{c2} -> {c2b}")
    st, j = post("/mask_redo", {})
    check("涂抹重做", int(j["stats"]["counts"].get("2", 0)) == c2, "")

    st, j = post("/save", {"features": [{
        "type": "Feature", "properties": {"id": 7, "cls": 5, "image_px": [[600, 600], [700, 600], [700, 700], [600, 700]]},
        "geometry": {"type": "Polygon", "coordinates": [[[112.7, 36.3], [112.71, 36.3], [112.71, 36.31], [112.7, 36.31], [112.7, 36.3]]]}}]})
    geo = j.get("path")
    check("矢量自动/手动保存", j.get("ok") and geo and os.path.isfile(geo), str(geo))

    st, j, _ = get("/parcels")
    check("矢量读回含 cls", len(j.get("features", [])) == 1 and j["features"][0]["properties"]["cls"] == 5, "")
    check("parcels 附 rev 修订号", "rev" in j, str(j.get("rev")))

    # 用影像中心构造多边形，保证落在图内
    a, b, c, d, e, f = sc["affine"]
    lonc = c + a * sc["width"] / 2.0
    latc = f + e * sc["height"] / 2.0
    ring = [[lonc - 0.002, latc - 0.002], [lonc + 0.002, latc - 0.002],
            [lonc + 0.002, latc + 0.002], [lonc - 0.002, latc + 0.002], [lonc - 0.002, latc - 0.002]]
    st, j = post("/bake", {"features": [{"type": "Feature", "properties": {"cls": 5},
                                         "geometry": {"type": "Polygon", "coordinates": [ring]}}]})
    check("多边形→标签图栅格化", j.get("count") == 1 and int(j["stats"]["counts"].get("5", 0)) > 100,
          str(j.get("stats", {}).get("counts")))

    st, j = post("/flush", {})
    npy = j.get("path")
    check("标签图落盘 .npy", npy and os.path.isfile(npy), str(npy))
    if npy and os.path.isfile(npy):
        a2 = np.load(npy)
        check("npy 形状/类型与影像一致", a2.shape == (3343, 4453) and a2.dtype == np.uint8,
              str(a2.shape) + str(a2.dtype))

    st, j = post("/export_tif", {})
    import rasterio
    with rasterio.open(j["path"]) as ds:
        check("导出 GeoTIFF 正确", (ds.width, ds.height, ds.count) == (4453, 3343, 1), str(ds.shape))
    st, j = post("/export_csv", {})
    check("导出统计 CSV", os.path.isfile(j["path"]), str(j.get("path")))

    st, j = post("/classes", {"classes": [{"id": i, "name": n, "color": col} for i, (n, col) in enumerate(
        [("耕地", "#4ade80"), ("林地", "#166534"), ("草地", "#a3e635"), ("水体", "#38bdf8"),
         ("建设用地", "#f87171"), ("裸地/其他", "#d1a054")], start=1)]})
    check("类别配置写回", len(j.get("classes", [])) == 6, "")

    # ---------------- 多人协作与画笔（v2.2 新增） ----------------
    print("\n-- 协作与画笔 --")
    st, j = post("/users", {"client": "fzA", "name": "甲", "color": "#4f8cff",
                            "cursor": [100.0, 120.0]})
    ids = [u["id"] for u in j.get("users", [])]
    check("POST /users 注册上线", "fzA" in ids, str(ids))
    check("心跳返回 seq/rev", "seq" in j and "rev" in j, str(j)[:80])

    st, j = post("/users", {"client": "fzB", "name": "乙", "color": "#3ecf8e"})
    st, j, _ = get("/users?since=0")
    types = [e["type"] for e in j.get("events", [])]
    check("GET /users 含 join 事件", "join" in types, str(types))
    check("在线两人", len(j.get("users", [])) == 2, str(len(j.get("users", []))))

    st, j, _ = get("/parcels")
    base_rev = j.get("rev", 0)
    feat = {"type": "Feature",
            "properties": {"id": 42, "cls": 3, "image_px": [[800, 800], [860, 800], [860, 860], [800, 860]]},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[112.705, 36.305], [112.706, 36.305],
                                          [112.706, 36.306], [112.705, 36.306],
                                          [112.705, 36.305]]]}}
    st, j = post("/save", {"mode": "merge", "upsert": [feat], "remove": ["p:7"],
                           "client": "fzA", "name": "甲"})
    check("协作式增量保存", j.get("ok") and j.get("rev", 0) > base_rev, str(j))
    st, j, _ = get("/parcels")
    ids = [f["properties"]["id"] for f in j["features"]]
    check("upsert 生效且 remove 删掉旧要素", ids == [42], str(ids))
    owner = j["features"][0]["properties"].get("owner")
    check("owner 溯源字段写入", owner == "甲", str(owner))

    st, j = post("/stroke", {"points": [[1000, 1000], [1060, 1000]], "size": 24,
                             "cls": 4, "erase": False, "client": "fzA", "name": "甲"})
    check("带归属的笔画返回 mask_rev", j.get("ok") and "mask_rev" in j, str(j.get("mask_rev")))
    st, j, _ = get("/maskval?x=1010&y=1000")
    check("吸管取像素", j.get("cls") == 4, str(j))
    st, j, _ = get("/maskval?client=fzA")
    stat = j.get("stat") or {}
    check("按人统计", stat.get("n") == 1 and stat.get("px", 0) > 200, str(stat))
    st, j = post("/mask_undo", {"client": "fzA", "name": "甲"})
    check("按人撤销", j.get("ok") is True, str(j))
    st, j, _ = get("/maskval?x=1010&y=1000")
    check("撤销后像素归零", j.get("cls") == 0, str(j))

    st, j = post("/users", {"client": "fzB", "name": "乙", "action": "leave"})
    ids = [u["id"] for u in j.get("users", [])]
    check("主动离场", "fzB" not in ids and "fzA" in ids, str(ids))

    check("页面含协作 UI（身份弹窗/头像栏/笔刷面板/事件浮层）",
          "modalUser" in txt and "userbar" in txt and "brushpop" in txt and "evtlog" in txt)
    check("页面含新快捷键说明", "Alt + 滚轮" in txt and "多人协作" in txt)

    # 清理本机测试产物
    print("\n-- 清理 --")
    for p in [geo, npy, f"d:\\UNET\\{img}_label.tif", f"d:\\UNET\\{img}_label_stats.csv"]:
        try:
            if p and os.path.isfile(p):
                os.remove(p); print("   removed", p)
        except OSError as e:
            print("   ! fail", p, e)
    # 打包版的 annotator_config.json 落在 exe 目录（类别写回测试产生）
    exe_cfg = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "dist", "FieldLabel", "annotator_config.json")
    try:
        if os.path.isfile(exe_cfg):
            os.remove(exe_cfg); print("   removed", exe_cfg)
    except OSError:
        pass
    print(f"\n===== 打包版结果: {len(PASS)} 通过 / {len(FAIL)} 失败 =====")
    for x in FAIL:
        print("  失败:", x)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
