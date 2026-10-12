# -*- coding: utf-8 -*-
"""PS 选区四件套 - /wand 魔棒接口测试。

用法: python tests/test_wand.py [port]
需先启动 annotate.py（任意影像）。
"""
import json
import sys
import time
import urllib.error
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8801
BASE = f"http://127.0.0.1:{PORT}"
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK] " if cond else "  [!!] ") + name + (f"   {extra}" if extra and not cond else ""))


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    return data, time.time() - t0


def main():
    with urllib.request.urlopen(BASE + "/info", timeout=30) as r:
        info = json.loads(r.read().decode())["scene"]
    print(f"影像: {info['name']} {info['width']}x{info['height']}")

    cx, cy = info["width"] / 2, info["height"] / 2

    j, dt = post("/wand", {"x": cx, "y": cy, "tol": 35})
    poly = j.get("polygon", [])
    print(f"  tol=35: {len(poly)} 顶点, {j.get('area_px')} px, partial={j.get('partial')}, {dt*1000:.0f}ms")
    check("正常容差返回结构", "polygon" in j and "area_px" in j and "partial" in j, str(j)[:120])
    if poly:
        check("轮廓顶点>=3且在图内",
              len(poly) >= 3 and all(0 <= p[0] <= info["width"] and 0 <= p[1] <= info["height"] for p in poly),
              str(poly[:3]))
        check("轮廓面积>0", j["area_px"] > 0, "")
        check("大区域性能 <5s", dt < 5.0, f"{dt:.2f}s")
    else:
        print("  (该点容差35未选出连通区域，属正常)")

    # 1b) 小容差：典型地块级区域应很快
    j1b, dt1b = post("/wand", {"x": cx, "y": cy, "tol": 5})
    print(f"  tol=5 : {len(j1b.get('polygon', []))} 顶点, {j1b.get('area_px')} px, {dt1b*1000:.0f}ms")
    check("小容差性能 <2s", dt1b < 2.0, f"{dt1b:.2f}s")
    # 1c) 缓存生效：同点重复调用应更快
    j1c, dt1c = post("/wand", {"x": cx, "y": cy, "tol": 5})
    check("重复调用（窗口缓存）更快或相近", dt1c <= dt1b + 0.5, f"{dt1b:.2f}s -> {dt1c:.2f}s")

    j2, dt2 = post("/wand", {"x": cx, "y": cy, "tol": 100})
    a2 = j2.get("area_px", 0)
    print(f"  tol=100: {len(j2.get('polygon', []))} 顶点, {a2} px, partial={j2.get('partial')}, {dt2*1000:.0f}ms")
    check("大容差选出 >100万 px", a2 > 1000000, str(a2))
    check("大容差性能 <6s", dt2 < 6.0, f"{dt2:.2f}s")
    check("大容差轮廓顶点<=1500", len(j2.get("polygon", [])) <= 1500, str(len(j2.get("polygon", []))))

    try:
        post("/wand", {"x": -100, "y": 5, "tol": 30})
        check("越界种子报错", False, "未报错")
    except urllib.error.HTTPError as e:
        check("越界种子报错", e.code == 500, f"HTTP {e.code}")

    ok = True
    for i in range(3):
        try:
            j3, _ = post("/wand", {"x": cx + i * 137, "y": cy + i * 91, "tol": 45})
            ok = ok and "polygon" in j3
        except Exception as e:
            ok = False
            print("   err:", e)
    check("连续 3 次调用稳定", ok, "")

    print(f"\n===== 选区后端结果: {len(PASS)} 通过 / {len(FAIL)} 失败 =====")
    for f in FAIL:
        print("  失败:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
