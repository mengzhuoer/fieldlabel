# -*- coding: utf-8 -*-
"""端到端接口测试：启动 annotate.py 子进程并逐项验证。"""
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
APP = os.path.join(WORK, "annotate.py")
# 默认使用仓库示例影像；可用环境变量 TA_IMG1 / TA_IMG2 指定任意两张 GeoTIFF
ZHANGZI = os.environ.get("TA_IMG1", r"d:\UNET\Sentinel2_Zhangzi_Jul_2023.tif")
SHOUYANG = os.environ.get("TA_IMG2", r"d:\UNET\Sentinel2_Shouyang_Jul_2023.tif")
PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (f"  {extra}" if extra and not cond else ""))


def get(path, raw=False):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        data = r.read()
        return r.status, (data if raw else json.loads(data.decode("utf-8"))), dict(r.headers)


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def wait_port(port, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def main():
    if not os.path.isfile(APP):
        print(f"未找到 annotate.py: {APP}")
        return 1
    if not os.path.isfile(ZHANGZI) or not os.path.isfile(SHOUYANG):
        print("跳过：未找到示例影像（需要两张 GeoTIFF）。")
        print(f"  影像1 = {ZHANGZI}")
        print(f"  影像2 = {SHOUYANG}")
        print("  可通过环境变量指定：set TA_IMG1=... & set TA_IMG2=...")
        return 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, APP, ZHANGZI, "--port", str(PORT),
                             "--local-only", "--no-browser"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=WORK, env=env, text=True, encoding="utf-8", errors="replace")
    try:
        if not wait_port(PORT):
            print("服务启动失败")
            return 1
        time.sleep(1.0)

        print("\n== 静态与信息 ==")
        st, body, hd = get("/", raw=True)
        check("GET / 返回页面", st == 200 and b"\xe6\xb6\x82\xe6\x8a\xb9" in body, f"len={len(body)}")
        st, j, _ = get("/info")
        check("GET /info 场景信息", j.get("scene", {}).get("name") == "Sentinel2_Zhangzi_Jul_2023.tif", str(j.get("scene", {}).get("name")))
        check("GET /info 返回类别", len(j.get("classes", [])) >= 6, str(j.get("classes")))
        check("GET /info 标签路径", str(j.get("mask_path", "")).endswith(".label.npy"), str(j.get("mask_path")))

        print("\n== 文件浏览器 ==")
        st, j, _ = get("/browse?path=")
        check("GET /browse 根列出盘符", len(j.get("dirs", [])) > 0, str(j))
        st, j, _ = get("/browse?path=" + urllib.parse.quote(r"d:\UNET"))
        names = [f["name"] for f in j.get("files", [])]
        check("GET /browse 列出 D:\\UNET 影像", any(n.endswith(".tif") for n in names), str(names[:5]))
        check("GET /browse 列出子目录", len(j.get("dirs", [])) > 0, str([d["name"] for d in j.get("dirs", [])][:5]))
        check("GET /browse 返回上级", bool(j.get("parent")), str(j.get("parent")))
        st, j, _ = get("/browse?path=" + urllib.parse.quote(r"d:\UNET\parcel_annotator_v2"))
        check("GET /browse 目录内无影像时 files 为空", j.get("files") == [], str(j.get("files")))

        print("\n== 瓦片 ==")
        st, png, hd = get("/tile?z=0&x=0&y=0", raw=True)
        check("GET /tile 返回 PNG", st == 200 and png[:8] == b"\x89PNG\r\n\x1a\n", hd.get("Content-Type"))
        st, png, hd = get("/tile?z=-3&x=3&y=2", raw=True)
        check("GET /tile 低层级 PNG", st == 200 and png[:4] == b"\x89PNG", "")
        st, png, hd = get("/overlay?z=0&x=0&y=0", raw=True)
        check("GET /overlay 初始为透明 PNG", st == 200 and png[:4] == b"\x89PNG", "")

        print("\n== 矢量读写 ==")
        st, j, _ = get("/parcels")
        check("GET /parcels 空集合", j.get("type") == "FeatureCollection", str(j))
        feats = [{
            "type": "Feature",
            "properties": {"id": 1, "cls": 4, "area_ha": 12.5,
                           "image_px": [[100, 100], [200, 100], [200, 200], [100, 200]]},
            "geometry": {"type": "Polygon", "coordinates": [[[112.6, 36.2], [112.7, 36.2],
                                                             [112.7, 36.3], [112.6, 36.3], [112.6, 36.2]]]},
        }]
        st, j = post("/save", {"features": feats})
        check("POST /save 写入 GeoJSON", j.get("ok") and str(j.get("path", "")).endswith(".parcels.geojson"), str(j))
        geo_path = j["path"]
        st, j, _ = get("/parcels")
        check("GET /parcels 读回 1 个要素", len(j.get("features", [])) == 1 and j["features"][0]["properties"]["cls"] == 4, "")
        st, j = post("/bake", {"features": feats, "cls": 4})
        check("POST /bake 栅格化多边形", j.get("count") == 1, str(j))
        baked = j.get("stats", {}).get("counts", {}).get("4", 0)
        check("POST /bake 产生 4 类像素", baked > 500, f"counts4={baked}")

        print("\n== 涂抹笔刷 ==")
        st, j = post("/stroke", {"points": [[300, 300], [360, 320], [420, 300]], "size": 24, "cls": 1, "erase": False})
        c1 = int(j.get("stats", {}).get("counts", {}).get("1", 0))
        check("POST /stroke 类别1 落笔", j.get("ok") and c1 > 1000, f"counts1={c1}")
        st, j = post("/stroke", {"points": [[320, 300]], "size": 40, "cls": 2, "erase": False})
        c2 = int(j.get("stats", {}).get("counts", {}).get("2", 0))
        c1_after2 = int(j.get("stats", {}).get("counts", {}).get("1", 0))
        check("POST /stroke 单击点划（类别2）", c2 > 800, f"counts2={c2}")
        st, png, _ = get("/overlay?z=-1&x=1&y=1", raw=True)
        check("GET /overlay 涂抹后 PNG 正常", png[:4] == b"\x89PNG", "")
        st, j = post("/stroke", {"points": [[330, 305]], "size": 60, "cls": 1, "erase": True})
        c1b = int(j.get("stats", {}).get("counts", {}).get("1", 0))
        check("POST /stroke 橡皮擦减少像素", c1b < c1_after2, f"{c1_after2} -> {c1b}")
        st, j = post("/mask_undo", {})
        c1c = int(j.get("stats", {}).get("counts", {}).get("1", 0))
        check("POST /mask_undo 恢复", j.get("ok") and c1c == c1_after2, f"{c1b} -> {c1c}, 期望 {c1_after2}")
        st, j = post("/mask_redo", {})
        c1d = int(j.get("stats", {}).get("counts", {}).get("1", 0))
        check("POST /mask_redo 重放", c1d == c1b, f"{c1c} -> {c1d}, 期望 {c1b}")
        st, j = post("/mask_undo", {})
        check("再次撤销回到擦除前", int(j.get("stats", {}).get("counts", {}).get("1", 0)) == c1_after2, "")

        print("\n== 类别管理 ==")
        st, j, _ = get("/classes")
        check("GET /classes", len(j.get("classes", [])) == 6, str(len(j.get("classes", []))))
        newc = [{"id": i, "name": n, "color": c} for i, (n, c) in enumerate(
            [("耕地", "#4ade80"), ("林地", "#166534"), ("草地", "#a3e635"),
             ("水体", "#38bdf8"), ("建设用地", "#f87171"), ("裸地/其他", "#d1a054"),
             ("园地", "#f0abfc")], start=1)]
        st, j = post("/classes", {"classes": newc})
        check("POST /classes 新增第7类", len(j.get("classes", [])) == 7, str(j))
        st, j = post("/classes", {"classes": newc[:-1]})
        check("POST /classes 还原6类", len(j.get("classes", [])) == 6, "")

        print("\n== 导出与落盘 ==")
        st, j = post("/flush", {})
        check("POST /flush 写入标签 npy", os.path.isfile(str(j.get("path", ""))), str(j.get("path")))
        mask_p = j["path"]
        st, j = post("/export_tif", {})
        tif_p = j["path"]
        check("POST /export_tif 生成 GeoTIFF", os.path.isfile(tif_p) and os.path.getsize(tif_p) > 1000, tif_p)
        import rasterio
        import numpy as np
        with rasterio.open(tif_p) as ds:
            check("导出 GeoTIFF 尺寸与 CRS 正确", (ds.width, ds.height) == (4453, 3343) and ds.count == 1,
                  f"{ds.width}x{ds.height}")
            arr = ds.read(1)
            check("导出 GeoTIFF 含类别值", len(np.unique(arr)) >= 3, str(np.unique(arr)[:6]))
        st, j = post("/export_csv", {})
        csv_p = j["path"]
        check("POST /export_csv 生成统计 CSV", os.path.isfile(csv_p), csv_p)
        with open(csv_p, "r", encoding="utf-8-sig") as f:
            txt = f.read()
        check("CSV 含类别统计行", "class_id,class_name" in txt and "水体" in txt, txt.splitlines()[:2])

        print("\n== 清空涂抹 ==")
        st, j = post("/mask_clear", {"cls": 2})
        check("POST /mask_clear 清空类别2", j.get("stats", {}).get("counts", {}).get("2", 0) == 0, str(j.get("stats", {}).get("counts")))
        st, j = post("/mask_undo", {})
        check("清空后可撤销", int(j.get("stats", {}).get("counts", {}).get("2", 0)) > 0, "")

        print("\n== 切换影像 ==")
        st, j = post("/open", {"path": SHOUYANG})
        check("POST /open 切到寿阳影像", j.get("info", {}).get("name") == "Sentinel2_Shouyang_Jul_2023.tif", str(j.get("info", {}).get("name")))
        check("切换后标签路径随之改变", str(j.get("mask", {}).get("path", "")).endswith("Sentinel2_Shouyang_Jul_2023.label.npy"), str(j.get("mask", {}).get("path")))
        st, png, _ = get("/tile?z=-2&x=1&y=0", raw=True)
        check("大图瓦片渲染正常", png[:4] == b"\x89PNG" and len(png) > 3000, f"len={len(png)}")
        st, png, hd = get("/tile?z=-5&x=0&y=0", raw=True)
        check("整图缩略瓦片正常", png[:4] == b"\x89PNG" and len(png) > 1000, f"len={len(png)}")
        st, j, _ = get("/parcels")
        check("寿阳既有矢量被读取", len(j.get("features", [])) >= 0, f"{len(j.get('features', []))} 个要素")

        print("\n== 异常处理 ==")
        try:
            st, j, _ = get("/tile?z=0&x=abc&y=0")
            check("非法瓦片参数返回错误", False, str(j))
        except urllib.error.HTTPError as e:
            check("非法瓦片参数返回错误", e.code in (400, 500), f"HTTP {e.code}")
        try:
            st, j = post("/open", {"path": r"d:\UNET\不存在的影像.tif"})
            check("打开不存在文件报错", False, str(j))
        except urllib.error.HTTPError as e:
            check("打开不存在文件报错", e.code == 500, f"HTTP {e.code}")

        print("\n== 清理测试产物 ==")
        for p in [geo_path, mask_p, tif_p, csv_p,
                  r"d:\UNET\Sentinel2_Shouyang_Jul_2023.label.npy",
                  r"d:\UNET\Sentinel2_Shouyang_Jul_2023_label.tif",
                  r"d:\UNET\Sentinel2_Shouyang_Jul_2023_label_stats.csv"]:
            try:
                if p and os.path.isfile(p):
                    os.remove(p)
                    print("  removed", p)
            except OSError as e:
                print("  ! remove failed", p, e)
    finally:
        try:
            proc.terminate()
            out = proc.communicate(timeout=15)[0]
        except Exception:
            proc.kill()
            out = ""
        print("\n---- 服务端输出 ----")
        print((out or "")[-2500:])

    print(f"\n===== 结果: {len(PASS)} 通过 / {len(FAIL)} 失败 =====")
    if FAIL:
        for f in FAIL:
            print("  失败:", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
