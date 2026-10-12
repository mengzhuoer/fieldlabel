# -*- coding: utf-8 -*-
"""多人协作 + 画笔优化专项测试。

自动生成一张 512x512 的小型 GeoTIFF 作为测试影像，不依赖真实数据；
覆盖：用户注册/心跳/事件流、矢量增量合并、rev 同步、按人撤销与统计、
吸管取像素、抗锯齿笔刷落点、旧格式兼容等 ≥15 项断言。
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(WORK, "annotate.py")
PORT = 8793
BASE = f"http://127.0.0.1:{PORT}"

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (f"  {extra}" if extra and not cond else ""))


def get(path, raw=False):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        data = r.read()
        return r.status, (data if raw else json.loads(data.decode("utf-8")))


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


def make_test_raster(path):
    """生成 512x512、4 波段的伪 GeoTIFF（EPSG:4326）。"""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    rng = np.random.default_rng(42)
    data = rng.integers(30, 220, size=(4, 512, 512), dtype="uint8")
    data[3] = np.clip(data[3].astype("int16") + 40, 0, 255).astype("uint8")
    with rasterio.open(
        path, "w", driver="GTiff", width=512, height=512, count=4,
        dtype="uint8", crs="EPSG:4326",
        transform=from_bounds(112.0, 36.0, 112.5, 36.5, 512, 512),
    ) as ds:
        ds.write(data)


def feat(pid, cls=1, px=(50, 50)):
    x, y = px
    return {
        "type": "Feature",
        "properties": {"id": pid, "cls": cls, "area_ha": 1.0,
                       "image_px": [[x, y], [x + 40, y], [x + 40, y + 40], [x, y + 40]]},
        "geometry": {"type": "Polygon", "coordinates": [[[112.1, 36.1], [112.12, 36.1],
                                                         [112.12, 36.12], [112.1, 36.12],
                                                         [112.1, 36.1]]]},
    }


def main():
    if not os.path.isfile(APP):
        print(f"未找到 annotate.py: {APP}")
        return 1
    tmpdir = tempfile.mkdtemp(prefix="ta_multi_")
    tif = os.path.join(tmpdir, "multi_test.tif")
    try:
        make_test_raster(tif)
    except Exception as e:
        print("无法生成测试影像:", e)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return 1

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, APP, tif, "--port", str(PORT),
                             "--local-only", "--no-browser"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=WORK, env=env, text=True, encoding="utf-8",
                            errors="replace")
    try:
        if not wait_port(PORT):
            print("服务启动失败")
            return 1
        time.sleep(1.0)

        A = {"client": "userA", "name": "张三"}
        B = {"client": "userB", "name": "李四"}

        print("\n== 1. 用户注册与在线状态 ==")
        st, j = post("/users", dict(A, color="#ff0000"))
        ids = [u["id"] for u in j.get("users", [])]
        check("注册用户A出现在在线列表", "userA" in ids, str(j.get("users")))
        st, j = post("/users", dict(B, color="#00ff00", cursor=[100.0, 120.0]))
        ids = [u["id"] for u in j.get("users", [])]
        check("注册用户B后两人在线", "userA" in ids and "userB" in ids, str(ids))
        bu = [u for u in j["users"] if u["id"] == "userB"][0]
        check("B 的光标被记录", bu.get("cursor") == [100.0, 120.0], str(bu.get("cursor")))
        check("B 的昵称与颜色保存", bu.get("name") == "李四" and bu.get("color") == "#00ff00",
              str((bu.get("name"), bu.get("color"))))

        print("\n== 2. 事件流与 seq ==")
        st, j = get("/users?since=0")
        types = [e["type"] for e in j.get("events", [])]
        check("GET /users 返回 join 事件", "join" in types, str(types))
        check("事件 seq 单调递增", all(j["events"][i]["seq"] < j["events"][i + 1]["seq"]
                                  for i in range(len(j["events"]) - 1)), "")
        seq0 = j.get("seq", 0)
        st, j2 = get(f"/users?since={seq0}")
        check("since 过滤后无新事件", j2.get("events") == [], str(j2.get("events")))
        st, j2 = post("/users", dict(A, since=seq0))
        check("POST /users 心跳返回 seq/rev", "seq" in j2 and "rev" in j2, str(j2))
        join_ev = [e for e in j["events"] if e["type"] == "join"]
        check("join 事件带 client 与 name", join_ev[0]["client"] == "userA"
              and join_ev[0]["name"] == "张三", str(join_ev[0]))

        print("\n== 3. 矢量增量合并（mode=merge） ==")
        st, j = get("/parcels")
        check("GET /parcels 带 rev 字段", "rev" in j, str(j.get("rev")))
        base_rev = j["rev"]
        st, j = post("/save", {"mode": "merge", "upsert": [feat(1)], "remove": [],
                               **A})
        check("A upsert #1 成功并返回新 rev", j.get("ok") and j.get("rev", 0) > base_rev, str(j))
        st, j = post("/save", {"mode": "merge", "upsert": [feat(2, cls=2, px=(200, 200))],
                               "remove": [], **B})
        check("B upsert #2 成功", j.get("ok"), str(j))
        st, j, = get("/parcels")
        ids = [f["properties"]["id"] for f in j["features"]]
        check("两人各自 upsert 后两要素并存", sorted(ids) == [1, 2], str(ids))
        st, j = post("/save", {"mode": "merge",
                               "upsert": [dict(feat(1), properties={
                                   **feat(1)["properties"], "cls": 5})],
                               "remove": [], **A})
        st, j, = get("/parcels")
        f1 = [f for f in j["features"] if f["properties"]["id"] == 1][0]
        check("同 id upsert 为更新而非新增", len(j["features"]) == 2
              and f1["properties"]["cls"] == 5, str(len(j["features"])))
        check("owner 溯源字段写入", f1["properties"].get("owner") == "张三",
              str(f1["properties"].get("owner")))
        check("editor 最后编辑者记录", f1["properties"].get("editor") == "张三",
              str(f1["properties"].get("editor")))
        st, j = post("/save", {"mode": "merge", "upsert": [],
                               "remove": ["p:2"], **B})
        st, j, = get("/parcels")
        ids = [f["properties"]["id"] for f in j["features"]]
        check("remove 按 key 删除要素", ids == [1], str(ids))
        rev_after = j["rev"]
        check("每次合并保存 rev 递增", rev_after >= base_rev + 4, f"{base_rev} -> {rev_after}")

        print("\n== 4. 旧版全量保存兼容 ==")
        st, j = post("/save", {"features": [feat(9)], **A})
        check("mode=full（缺省）整体替换仍可用", j.get("ok"), str(j))
        st, j, = get("/parcels")
        check("全量保存后仅剩 1 个要素", len(j["features"]) == 1
              and j["features"][0]["properties"]["id"] == 9, str(len(j["features"])))
        # 清空，进入涂抹测试
        post("/save", {"mode": "merge", "upsert": [], "remove": ["p:9"], **A})

        print("\n== 5. 笔画归属与按人统计 ==")
        st, j = post("/stroke", {"points": [[60, 60], [120, 60]], "size": 20,
                                 "cls": 1, "erase": False, **A})
        check("A 落笔成功并返回 mask_rev", j.get("ok") and "mask_rev" in j, str(j.get("mask_rev")))
        mask_rev0 = j["mask_rev"]
        st, j = post("/stroke", {"points": [[300, 300], [360, 300]], "size": 20,
                                 "cls": 3, "erase": False, **B})
        check("B 落笔成功", j.get("ok"), "")
        check("mask_rev 随笔画递增", j["mask_rev"] > mask_rev0, f"{mask_rev0} -> {j['mask_rev']}")
        st, j = get("/maskval?client=userA")
        sa = j.get("stat") or {}
        check("按人统计 A：笔画数=1", sa.get("n") == 1, str(sa))
        check("按人统计 A：像素>0", sa.get("px", 0) > 500, str(sa.get("px")))
        st, j = get("/maskval?client=userB")
        sb = j.get("stat") or {}
        check("按人统计 B 独立计数", sb.get("n") == 1 and sb.get("px", 0) > 500, str(sb))

        print("\n== 6. 吸管取像素 ==")
        st, j = get("/maskval?x=70&y=60")
        check("吸管取到 A 的类别1", j.get("cls") == 1, str(j))
        st, j = get("/maskval?x=310&y=300")
        check("吸管取到 B 的类别3", j.get("cls") == 3, str(j))
        st, j = get("/maskval?x=500&y=500")
        check("吸管取空白为背景0", j.get("cls") == 0, str(j))

        print("\n== 7. 按人撤销（栈中摘除+重放） ==")
        st, j = post("/mask_undo", dict(A))
        check("A 撤销自己笔画成功", j.get("ok") is True, str(j))
        st, j = get("/maskval?x=70&y=60")
        check("A 撤销后自己的点归零", j.get("cls") == 0, str(j))
        st, j = get("/maskval?x=310&y=300")
        check("A 撤销不影响 B 的笔画", j.get("cls") == 3, str(j))
        st, j = post("/mask_undo", dict(B))
        check("B 撤销成功（栈中非顶部摘除后仍可用）", j.get("ok") is True, str(j))
        st, j = get("/maskval?x=310&y=300")
        check("B 的笔画归零", j.get("cls") == 0, str(j))
        st, j = post("/mask_undo", dict(A))
        check("无可撤销时返回 ok=false", j.get("ok") is False, str(j))
        st, j = post("/mask_redo", dict(A))
        check("A 重做成功", j.get("ok") is True, str(j))
        st, j = get("/maskval?x=70&y=60")
        check("重做后 A 的点恢复类别1", j.get("cls") == 1, str(j))
        st, j = post("/mask_clear", {"cls": 1, **A})
        check("按人清空带归属", j.get("ok"), str(j))
        st, j = get("/maskval?x=70&y=60")
        check("清空后像素归零", j.get("cls") == 0, str(j))
        st, j = post("/mask_undo", dict(A))
        check("清空可撤销", j.get("ok") is True, str(j))
        st, j = get("/maskval?x=70&y=60")
        check("撤销清空后恢复", j.get("cls") == 1, str(j))

        print("\n== 7b. 相交笔画的撤销重放合成 ==")
        post("/mask_clear", {"cls": None, **A})
        st, j = post("/stroke", {"points": [[60, 60], [140, 60]], "size": 20,
                                 "cls": 1, "erase": False, **A})
        st, j = post("/stroke", {"points": [[100, 60], [180, 60]], "size": 20,
                                 "cls": 3, "erase": False, **B})
        st, j = post("/mask_undo", dict(A))
        check("相交场景 A 撤销成功", j.get("ok") is True, str(j))
        st, j = get("/maskval?x=70&y=60")
        check("A 独有区域归零", j.get("cls") == 0, str(j))
        st, j = get("/maskval?x=120&y=60")
        check("重叠区由 B 的笔画补回（重放生效）", j.get("cls") == 3, str(j))
        st, j = get("/maskval?x=160&y=60")
        check("B 独有区域保留", j.get("cls") == 3, str(j))
        st, j = post("/mask_undo", dict(B))
        check("相交场景 B 撤销成功", j.get("ok") is True, str(j))
        st, j = get("/maskval?x=120&y=60")
        check("B 撤销后重叠区归零（合成 patch 正确）", j.get("cls") == 0, str(j))
        st, j = get("/maskval?x=160&y=60")
        check("B 撤销后独有区归零", j.get("cls") == 0, str(j))

        print("\n== 8. 广播事件（stroke/save 等） ==")
        st, j = get("/users?since=0")
        types = [e["type"] for e in j["events"]]
        check("stroke 事件已广播", "stroke" in types, str(types))
        check("save 事件已广播", "save" in types, str(types))
        check("undo/clear 事件已广播", "undo" in types and "clear" in types, str(types))
        sv = [e for e in j["events"] if e["type"] == "save"][0]
        check("save 事件带 client 与 rev", sv["client"] == "userA" and "rev" in sv["data"],
              str(sv))

        print("\n== 9. 用户离开 ==")
        st, j = post("/users", dict(B, action="leave"))
        ids = [u["id"] for u in j.get("users", [])]
        check("leave 后 B 从在线列表消失", "userB" not in ids and "userA" in ids, str(ids))
        st, j = get("/users?since=0")
        check("leave 事件已广播", any(e["type"] == "leave" and e["client"] == "userB"
                                  for e in j["events"]), str([e["type"] for e in j["events"]]))

        print("\n== 10. 抗锯齿笔刷（超采样圆刷） ==")
        st, j = post("/mask_clear", {"cls": None, **A})
        st, j = post("/stroke", {"points": [[256, 256]], "size": 33, "cls": 4,
                                 "erase": False, **A})
        c4 = int(j.get("stats", {}).get("counts", {}).get("4", 0))
        import math
        expect = math.pi * (33 / 2.0) ** 2
        check("单击圆点面积接近理论值(±12%)",
              abs(c4 - expect) / expect < 0.12, f"{c4} vs {expect:.0f}")
        # 圆度：统计 bbox 内四个角必须为空
        corners = [(256 - 17, 256 - 17), (256 + 16, 256 - 17),
                   (256 - 17, 256 + 16), (256 + 16, 256 + 16)]
        ok_corner = True
        for (x, y) in corners:
            st, j = get(f"/maskval?x={x}&y={y}")
            if j.get("cls") != 0:
                ok_corner = False
        check("圆点四角为空（非方形）", ok_corner, "")

        print("\n== 11. 页面与资源 ==")
        st, body = get("/", raw=True)
        check("页面含协作身份弹窗", "modalUser".encode() in body, "")
        check("页面含笔刷右键面板", "brushpop".encode() in body, "")
        check("页面含在线头像容器", "userbar".encode() in body, "")
        check("页面含事件浮层", "evtlog".encode() in body, "")

    finally:
        try:
            proc.terminate()
            out = proc.communicate(timeout=15)[0]
        except Exception:
            proc.kill()
            out = ""
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("\n---- 服务端输出 ----")
        print((out or "")[-2000:])

    print(f"\n===== 结果: {len(PASS)} 通过 / {len(FAIL)} 失败 =====")
    if FAIL:
        for f in FAIL:
            print("  失败:", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
