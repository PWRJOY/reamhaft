"""
铰刀手柄 —— build123d 参数化建模（平放打印姿态）
==================================================

跑这个文件 = 生成 STEP + STL + 打印说明（输出到 ../output/<档案名>/）。

用法::

    python codes/handle_build.py                        # 用默认档案 r01-2.36
    python codes/handle_build.py --reamer r02-xxx       # 换一把刀（reamers/r02-xxx.json）
    python codes/handle_build.py --set length=18        # 临时改一个参数
    python codes/handle_build.py --set bore_clearance=0.04
    python codes/handle_build.py --set teardrop=0       # 关掉孔顶泪滴
    python codes/handle_build.py --no-export            # 只建模不自检导出

档案只写这把刀自己的实测数据（柄径、方头、刀长），外形和打印参数全走默认值，
所以一把刀一个 JSON，互不覆盖。

坐标约定见 params.py：X = 轴向（x=0 是插刀口），Z = 打印高度（z=0 贴床）。
STL 已按平放姿态导出，切片软件里不要再旋转。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

from build123d import (
    Align,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Cone,
    Cylinder,
    Mode,
    Plane,
    Polygon,
    Pos,
    RegularPolygon,
    Rectangle,
    RectangleRounded,
    export_step,
    export_stl,
    extrude,
    loft,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from params import DEFAULT_PROFILE, DEFAULT, MATERIALS, HandleParams, load_profile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output"
STEP_NAME = "reamer_handle.step"
STL_NAME = "reamer_handle.stl"

MIN_WALL = 1.2          # mm，任何位置的最小壁厚警戒线
FIT_MIN = -0.05         # mm，预估装配间隙的合理下限（再负就是压不进的死过盈）
FIT_MAX = 0.10          # mm，预估装配间隙的合理上限（再大就是旷量）


# ----------------------------------------------------------------------
# 参数自检
# ----------------------------------------------------------------------
def validate(p: HandleParams) -> None:
    """把会做出废件的参数组合提前拦下来。"""
    errors: list[str] = []

    if p.wall < MIN_WALL:
        errors.append(
            f"插孔太大：Ø{p.bore_d} 在 R{p.across_flats/2:.2f} 的棱柱上只剩 "
            f"{p.wall:.2f} mm 壁厚（下限 {MIN_WALL}）"
        )
    if p.bore_top_z > p.across_flats - MIN_WALL:
        errors.append(
            f"孔顶 z={p.bore_top_z:.2f} 顶到顶面了（上面只有 "
            f"{p.across_flats - p.bore_top_z:.2f} mm 材料）"
        )
    if p.floor_thickness < 1.5:
        errors.append(f"封闭端壁厚 {p.floor_thickness:.2f} < 1.5 mm，刀柄会把孔底顶穿")
    if p.socket_len <= 3.0:
        errors.append(f"插孔只有 {p.socket_len:.2f} mm 深，太浅")
    if p.length < 10.0:
        errors.append(f"总长 {p.length:.1f} mm 太短，握不住")
    if p.mouth_chamfer > 0 and (
        p.mouth_chamfer >= p.floor_thickness or p.mouth_chamfer > p.socket_len / 2
    ):
        errors.append("孔口倒角过大")
    if p.material.upper() not in MATERIALS:
        errors.append(
            f"未知耗材 {p.material!r}，可选：{', '.join(sorted(MATERIALS))}"
        )
    if p.neck_on:
        if p.mouth_neck_flats >= p.across_flats:
            errors.append(
                f"收口对边距 {p.mouth_neck_flats:.2f} 不小于外形 "
                f"{p.across_flats:.2f}，等于没收口"
            )
        if p.neck_wall < MIN_WALL:
            errors.append(
                f"收口只剩 {p.neck_wall:.2f} mm 壁厚（下限 {MIN_WALL}），"
                f"孔口会崩；mouth_neck_flats 至少 {p.bore_d + 2 * MIN_WALL:.2f}"
            )
        if p.mouth_neck_len <= 0.0:
            errors.append("mouth_neck_len 必须大于 0，否则端面直接变成刀口")
        if p.neck_end > p.socket_len:
            errors.append(
                f"收口总长 {p.neck_end:.2f} 已经越过孔底（孔深 {p.socket_len:.2f}）"
            )
    if p.square_on:
        if p.square_hole <= p.square_size_min:
            errors.append(
                f"方孔 {p.square_hole:.2f} 不大于方头短边 {p.square_size_min:.2f}，"
                "装不进去；square_clearance 必须 > 0"
            )
        if p.pass_margin < -0.04:
            errors.append(
                f"方头对角线 Ø{p.square_envelope_d:.2f}（实测）"
                f"比预估孔径 Ø{p.printed_bore_d:.2f} 还大 "
                f"{-p.pass_margin:.3f} 单边，压不进去。"
                f"要么把 bore_clearance 加到 "
                f"{(p.square_envelope_d - p.shank_d) / 2 + 0.01:.2f} 以上，"
                f"要么重新量方头（对角线不该比柄身粗）"
            )
        if p.square_len <= 0.0:
            errors.append("square_len 必须大于 0")
        if p.square_tip_len > p.square_len + 1e-9:
            short = p.socket_len - p.square_tip_len
            errors.append(
                f"方头长 {p.square_tip_len:.2f} > 方孔深 {p.square_len:.2f}，"
                f"多出的 {p.square_tip_len - p.square_len:.2f} mm 会顶进孔底壁；"
                f"把 square_len 改成 ≥ {p.square_tip_len:.2f}"
                f"（圆孔段会缩到 {short:.2f}，孔底壁仍是 {p.floor_thickness:.2f}）"
            )
        if p.square_start < 1.5:
            errors.append(
                f"方孔入口只剩 x={p.square_start:.2f} 的让位孔，柄身几乎没有导向段；"
                f"把 square_len 减到 {p.socket_len - 1.5:.1f} 以内"
            )
        # 柄身末端 = 方孔入口（由 params.shank_end_x 保证），方头整根落在方孔里，
        # 所以这里不需要再检查「柄身顶在台阶上」——那条约束已经在派生式里消掉了。
        if p.square_top_z > p.across_flats - MIN_WALL:
            errors.append(
                f"方孔尖顶 z={p.square_top_z:.2f} 顶到顶面了（只剩 "
                f"{p.across_flats - p.square_top_z:.2f} mm）"
            )
        sq_wall = p.across_flats / 2.0 - p.square_hole / 2.0
        if sq_wall < MIN_WALL:
            errors.append(
                f"方孔侧壁只剩 {sq_wall:.2f} mm（下限 {MIN_WALL}）"
            )

    if p.set_screw_hole:
        if p.screw_thread_depth < 2.5:
            errors.append(
                f"顶丝孔的螺纹只有 {p.screw_thread_depth:.2f} mm"
                f"（顶面 z={p.across_flats:.2f} 减孔系最高 z={p.bore_top_z:.2f}）"
                f"，M3 咬不住，至少 2.5 mm"
            )
        lo = p.neck_end + 1.5
        hi = p.pass_len - 1.5
        if not (lo <= p.set_screw_x <= hi):
            errors.append(
                f"顶丝位置 x={p.set_screw_x:.2f} 超出让位孔段 [{lo:.2f}, {hi:.2f}]；"
                "压在方孔上会把方孔尖顶切穿"
            )
        if not (2.0 <= p.set_screw_d <= 3.4):
            errors.append(
                f"顶丝孔径 Ø{p.set_screw_d:.2f} 不合理；"
                f"M3 用 2.70（自攻）或 2.50（先攻丝）"
            )
    if p.bore_d <= p.shank_d - 0.05:
        errors.append(
            f"孔 Ø{p.bore_d:.2f} 比柄径 Ø{p.shank_d:.2f} 还小得多，压不进去；"
            "真过盈请控制在 0.05 以内"
        )
    if not p.square_on and not (FIT_MIN <= p.printed_clearance <= FIT_MAX):
        errors.append(
            f"预估装配间隙 {p.printed_clearance:+.3f}（按缩量 {p.print_shrink_d:.2f} 估）"
            f"超出 {FIT_MIN:+.2f} ~ {FIT_MAX:+.2f} 的可用区间"
        )

    if errors:
        raise ValueError("参数不合法：\n  - " + "\n  - ".join(errors))


# ----------------------------------------------------------------------
# 建模
# ----------------------------------------------------------------------
def _sketch_plane(p: HandleParams) -> Plane:
    """YZ 平面，原点抬到孔轴高度：局部 x→全局 Y，局部 y→全局 Z，法向 +X。"""
    return Plane(origin=(0, 0, p.hole_z), x_dir=(0, 1, 0), z_dir=(1, 0, 0))


def _set_screw_hole(p: HandleParams):
    """顶丝孔本体：从孔顶垂直向上贯穿顶面的圆柱（沿 +Z，可无支撑打印）。"""
    bottom, top = p.bore_top_z, p.across_flats
    return Pos(p.set_screw_x, 0, bottom) * Cylinder(
        radius=p.set_screw_d / 2.0,
        height=top - bottom + 1.0,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )


def _set_screw_counterbore(p: HandleParams):
    """顶丝孔入口的 45° 倒角，拧螺钉时不啃边。"""
    top = p.across_flats
    return Pos(p.set_screw_x, 0, top - 0.6) * Cone(
        bottom_radius=p.set_screw_d / 2.0,
        top_radius=p.set_screw_d / 2.0 + 0.6,
        height=0.8,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )


def build_handle(p: HandleParams = DEFAULT):
    """返回手柄实体（Solid）。"""
    validate(p)
    plane = _sketch_plane(p)

    # 1) 正八棱柱：轴线沿 X，一个平面贴 z=0 ---------------------------------
    with BuildPart() as bp:
        # 1) 外形：等截面棱柱，或「小八边形收口 + 45° 锥 + 棱柱」
        if p.neck_on:
            for x, r in p.neck_sections:
                with BuildSketch(plane.offset(x)):
                    RegularPolygon(
                        radius=r,
                        side_count=8,
                        major_radius=True,
                        rotation=p.profile_rotation,
                    )
            loft(ruled=True)
        else:
            with BuildSketch(plane):
                RegularPolygon(
                    radius=p.circumradius,
                    side_count=8,
                    major_radius=True,
                    rotation=p.profile_rotation,
                )
            extrude(amount=p.length)

        # 2) 让位孔：从 x=0 打到方孔入口 -----------------------------------
        with BuildSketch(plane):
            Circle(p.bore_r)
            if p.teardrop:
                k = p.bore_r / math.sqrt(2.0)
                Polygon(
                    (-k, k),
                    (k, k),
                    (0.0, p.bore_r * math.sqrt(2.0)),
                    align=None,
                    mode=Mode.ADD,
                )
        extrude(amount=p.pass_len, mode=Mode.SUBTRACT)

        # 2b) 孔底方孔：正方 + 45° 收顶，与尾部方头咬合传扭 ----------------
        #     45° 尖顶不是为了好看：方孔是 2.2 mm 宽的平顶，51 层里那一段
        #     就是纯桥接，PETG 会塌成腰形，方头根本塞不进去。收成尖顶后
        #     顶面变成可自支撑的斜悬垂，孔形不塌，孔顶还正好落在泪滴高度上。
        if p.square_on:
            half = p.square_hole / 2.0
            with BuildSketch(plane.offset(p.square_start)):
                Polygon(
                    (-half, -half),
                    (half, -half),
                    (half, half),
                    (0.0, 2.0 * half),
                    (-half, half),
                    align=None,
                )
            extrude(amount=p.square_len, mode=Mode.SUBTRACT)

    body = bp.part

    # 3) 孔口 45° 导入倒角（几何减法，不依赖边选择） -----------------------
    if p.mouth_chamfer > 0:
        mouth = Pos(0, 0, p.hole_z) * Cone(
            bottom_radius=p.bore_r + p.mouth_chamfer,
            top_radius=p.bore_r,
            height=p.mouth_chamfer,
            rotation=(0, 90, 0),
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        body = body - mouth

    # 4) 顶丝孔：顶面垂直打到孔顶，螺纹深度 = 顶面到孔顶 --------------------
    if p.set_screw_hole:
        body = body - _set_screw_hole(p)
        body = body - _set_screw_counterbore(p)

    body = body.clean()

    solids = body.solids()
    if len(solids) != 1:
        raise RuntimeError(f"布尔运算后得到 {len(solids)} 个实体，应为 1 个")
    return solids[0]


# ----------------------------------------------------------------------
# 校验 / 报告
# ----------------------------------------------------------------------
def _probe(p: HandleParams):
    """装配探针：Ø2.36 柄身 + 尾部方头（四角按实测对角反推的半径倒圆）。

    方头必须带倒圆，不能用尖角 Box：实物跨角量出来只有 Ø2.36，
    尖角版会造出一圈假干涉，把自检打成 FAIL，而真件根本插得进去。
    """
    x_end = p.shank_end_x
    shank = Pos(0, 0, p.hole_z) * Cylinder(
        radius=p.shank_d / 2.0,
        height=x_end,
        rotation=(0, 90, 0),
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    if not p.square_on:
        return shank
    r = min(p.square_corner_r, min(p.square_size, p.square_size_min) / 2 - 1e-3)
    with BuildPart() as tip:
        with BuildSketch(_sketch_plane(p).offset(x_end)):
            if r > 1e-4:
                RectangleRounded(p.square_size, p.square_size_min, radius=r)
            else:
                Rectangle(p.square_size, p.square_size_min)
        extrude(amount=p.square_tip_len)
    return shank + tip.part


def report(p: HandleParams, body) -> dict:
    """几何自检 + 尺寸校核，返回一份可打印的报告字典。"""
    bb = body.bounding_box()
    vol = body.volume
    size = (bb.size.X, bb.size.Y, bb.size.Z)

    # 参考体积（不含倒角与泪滴的解析近似）
    v_ref = p.volume_ref_mm3

    # 铰刀假体：柄身 + 方头，按装到位的姿态插进去
    probe = _probe(p)
    hit = body.intersect(probe)
    if hit is None:                       # 无交集
        clash = 0.0
    elif isinstance(hit, (list, tuple)):  # 过盈时返回 ShapeList
        clash = sum(s.volume for s in hit)
    else:
        clash = hit.volume

    def ok_dims() -> bool:
        return (
            abs(size[0] - p.length) < 1e-4
            and abs(size[1] - p.across_flats) < 1e-4
            and abs(size[2] - p.across_flats) < 1e-4
        )

    valid = body.is_valid() if callable(body.is_valid) else bool(body.is_valid)
    checks = [
        ("实体有效", valid),
        ("单一实体", len(body.solids()) == 1),
        (f"外形 {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f}", ok_dims()),
        ("底面贴床 z = 0", abs(bb.min.Z) < 1e-6),
        (f"柄身 + 方头 无干涉插入（{clash:.4f} mm³）", clash < 1e-3),
        (f"孔壁最小壁厚 {p.wall:.2f} ≥ {MIN_WALL}", p.wall >= MIN_WALL),
        (
            f"收口段壁厚 {p.neck_wall:.2f} ≥ {MIN_WALL}",
            (not p.neck_on) or p.neck_wall >= MIN_WALL,
        ),
        (
            f"孔系最高 z={p.bore_top_z:.2f} 未破顶面",
            p.bore_top_z <= p.across_flats - MIN_WALL,
        ),
        (
            f"顶丝孔螺纹 {p.screw_thread_depth:.2f} ≥ 2.5"
            if p.set_screw_hole else "顶丝孔（已关闭）",
            (not p.set_screw_hole) or p.screw_thread_depth >= 2.5,
        ),
        (
            f"方头对角 Ø{p.square_envelope_d:.2f}（实测）"
            f"能过 Ø{p.printed_bore_d:.2f} 孔（余量 {p.pass_margin:+.3f}）"
            if p.square_on else "方孔（未启用）",
            (not p.square_on) or p.pass_margin >= -0.04,
        ),
        (
            f"方孔 {p.square_hole:.2f} 见方 × {p.square_len:.2f} 深，"
            f"咬合 {p.square_engage:.2f}"
            if p.square_on else "方孔深度（未启用）",
            (not p.square_on) or p.square_engage >= 1.5,
        ),
        (
            f"柄身末端 x={p.shank_end_x:.2f} 未越过方孔入口 x={p.square_start:.2f}"
            if p.square_on else "柄身/方孔台阶（未启用）",
            (not p.square_on) or p.shank_end_x <= p.square_start + 1e-6,
        ),
        ("体积在合理范围", 0.90 * v_ref < vol < 1.06 * v_ref),
        (
            f"预估方孔旷量 {p.square_play:.3f} 在可用区间",
            p.square_on and -0.02 <= p.square_play <= 0.10,
        ) if p.square_on else (
            f"预估装配间隙 {p.printed_clearance:+.3f} 在可用区间",
            FIT_MIN <= p.printed_clearance <= FIT_MAX,
        ),
    ]

    print("\n几何自检")
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"  体积           {vol/1000:.3f} cm³")
    print(f"  估算质量       {vol/1000*p.mat.density:.2f} g  ({p.mat.name} 实心)")
    print(f"  表面积         {body.area/100:.2f} cm²")

    if p.square_on:
        print(f"\n孔系（Ø{p.bore_d:.2f} 圆孔 + {p.square_hole:.2f} 方孔）")
    else:
        print(f"\n插孔（Ø{p.bore_d:.2f} 盲孔）")
    print("-" * 64)
    print(f"  {'圆孔' if p.square_on else '插孔'}         Ø{p.bore_d:.2f} × "
          f"{p.pass_len:.2f} 深"
          + (f"（= 柄径 + 2×{p.bore_clearance:.2f}；方头对角 "
             f"{p.square_envelope_d:.2f} 不比它粗）" if p.square_on else ""))
    if p.square_on:
        print(f"  方孔           {p.square_hole:.2f} 见方 × {p.square_len:.2f} 深"
              f"，入口 x = {p.square_start:.2f}")
        print(f"  方头           {p.square_size:.2f} × {p.square_size_min:.2f}"
              f" × {p.square_tip_len:.2f} 长"
              f"（实测对角 {p.square_envelope_d:.2f}，四角倒圆 ≈R{p.square_corner_r:.2f}）")
        print(f"  预估打印方孔   {p.printed_square:.2f} 见方"
              f"   （扣收缩 {p.print_shrink_d:.2f}）"
              f" → 单边旷量 {p.square_play:.3f}")
        print(f"  柄身单边间隙   {p.shank_radial_gap:.3f}   [10 mm 配合长定心，摆动 "
              f"{math.degrees(math.atan(p.shank_radial_gap / p.shank_end_x)):.1f}°]")
    else:
        print(f"  预估打印孔径   Ø{p.printed_bore_d:.2f}"
              f"   （扣收缩 {p.print_shrink_d:.2f}）"
              f" → 单边 {p.printed_clearance:+.3f}")
    print(f"  插入深度       {p.insert_depth:.2f} mm"
          f"   柄部外露 {p.exposed_shank:.2f} mm")
    print("-" * 64)

    # 传扭校核 ------------------------------------------------------------
    t_need = (0.03, 0.10)                       # 手铰小孔的大致扭矩需求
    print("\n传扭校核")
    print("-" * 64)
    print(f"  手铰 Ø{p.shank_d:.2f} 需要     约 {t_need[0]:.2f} ~ {t_need[1]:.2f} N·m")
    if p.square_on:
        t_lo, t_hi = p.square_torque
        print(f"  方孔咬合 {p.square_engage:.2f} mm      {t_lo:.2f} ~ {t_hi:.2f} N·m"
              f"（形状约束，余量 {t_lo / t_need[1]:.1f} ~ {t_hi / t_need[1]:.1f} 倍）")
        if t_lo < t_need[1]:
            print(f"  {'':24s}下限按 15 MPa 长期蠕变算，偏保守；"
                  f"按 30 MPa 短时算是 {t_lo * 2:.2f} N·m，够手铰用")
        print(f"  旧版圆孔底（无方孔）       接触压力≈0 → 摩擦扭矩≈0  ← 空转的根因")
    else:
        r = p.shank_d / 2.0
        a = p.bond_area_mm2
        t_glue = (10.0 * a * r / 1000.0, 20.0 * a * r / 1000.0)
        print(f"  贴合推入（本版）           接触压力≈0 → 摩擦扭矩≈0")
        print(f"  胶接 638（{a:.0f} mm² 粘接面）   {t_glue[0]:.1f} ~ {t_glue[1]:.1f} N·m")
    print("-" * 64)

    if not all(ok for _, ok in checks):
        raise RuntimeError("自检未通过，请检查参数")

    return {"volume_mm3": vol, "bbox": size, "clash_mm3": clash}


# ----------------------------------------------------------------------
# 打印说明
# ----------------------------------------------------------------------
def write_print_notes(p: HandleParams, info: dict,
                      out_dir: Path | None = None, slug: str = "default") -> Path:
    out_dir = OUT_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "print_notes.txt"
    try:
        rel_out = out_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        rel_out = out_dir.name
    m = p.mat
    mass = info["volume_mm3"] / 1000.0 * m.density

    # 方孔实测对照表（左列纯 ASCII，等宽对齐）
    sq = p.square_size
    f_lo, f_ok, f_hi = sq + 0.02, sq + 0.10, sq + 0.18
    rows = [
        (f"< {f_lo:.2f}", "太紧。方锉轻修四角，或 square_clearance 加到 0.08 重打"),
        (f"{f_lo:.2f} ~ {f_ok:.2f}", "紧配，手压能进 ← 目标区间"),
        (f"{f_ok:.2f} ~ {f_hi:.2f}", "轻推入，起转有一点点空程，能用"),
        (f"> {f_hi:.2f}", "偏松，空程明显；重打时 square_clearance 减 0.03"),
    ]
    table = "\n".join(f"      {a:<14s}→ {b}" for a, b in rows)

    neck_note = (
        f"\n  收口那一小段比主体细，它下面有一块 {p.mouth_neck_len:.1f} mm 的悬挑（宽 {p.neck_side_len:.2f}）。\n"
        f"  切片器会按桥接打，不用加支撑；桥接速度设 25~30 mm/s 就不会塌。\n"
        f"  实在想要零悬挑，把 mouth_neck_len 设成 1.0。"
        if p.neck_on else ""
    )
    neck_line = (
        f"孔口收口 : 端面 Ø{p.mouth_neck_flats:.2f} A/F × {p.mouth_neck_len:.2f}，"
        f"45° 锥 {p.neck_taper:.2f} 过渡回 {p.across_flats:.2f}\n"
        if p.neck_on else ""
    )
    screw_line = (
        f"顶丝孔   : Ø{p.set_screw_d:.2f} 从顶面打到孔顶，可用螺纹 "
        f"{p.screw_thread_depth:.2f}，轴向 x = {p.set_screw_x:.2f}（M3 顶丝）\n"
        if p.set_screw_hole else ""
    )
    sq_line = (
        f"孔底方孔 : {p.square_hole:.2f} 见方 × {p.square_len:.2f} 深"
        f"（入口 x = {p.square_start:.2f}，45° 尖顶 z = {p.square_top_z:.2f}）\n"
        if p.square_on else ""
    )
    sq_fit_line = (
        f"配合     : 柄身 Ø{p.shank_d:.2f} / 方头对角 Ø{p.square_envelope_d:.2f}"
        f"（两者一样粗，实测）→ 单边 {p.shank_radial_gap:.3f}\n"
        f"           方头不比柄身粗，孔不用为它让位；Ø{p.bore_d:.2f} 就是"
        f" 柄径 + 2 × {p.bore_clearance:.2f}\n"
        f"           插入方孔 {p.square_hole:.2f}，预估打印 {p.printed_square:.2f}"
        f" → 单边旷量 {p.square_play:.3f}，{p.fit_word}\n"
        if p.square_on else
        f"配合     : Ø{p.shank_d:.2f} 刀柄直接压入，预估孔径 Ø{p.printed_bore_d:.2f}"
        f"\n           → 单边 {p.printed_clearance:+.3f}，{p.fit_word}\n"
    )

    t_lo = 10.0 * p.bond_area_mm2 * (p.shank_d / 2.0) / 1000.0   # 胶接扭矩下限 N·m
    t_hi = 2.0 * t_lo                                            # 上限（20 MPa）
    sq_t_lo, sq_t_hi = p.square_torque

    if p.square_on:
        fit_section = f"""三、装刀：推到孔底，最后 2 mm 转一下让它落进方孔
  ---------------------------------------------------------------
  1) 冷透再装。刚下机还热时材料是软的，孔形会跟着刀一起变形。
  2) 清孔。镊子掏掉孔口拉丝，再看一眼方孔：四个角有没有塌、尖顶有没有
     挂下来的料。有毛刺就用美工刀贴着侧面刮掉 —— 只刮毛刺，别扩孔。
  3) 插入。方头先进口，前 {p.square_start:.1f} mm 是 Ø{p.bore_d:.2f} 圆孔。
     方头对角 {p.square_envelope_d:.2f} = 柄径，所以它在圆孔里的松紧和柄身一样
     （单边 {p.shank_radial_gap:.3f}），插进去、转着找角都没有阻碍。
  4) 最后 {p.square_len:.0f} mm 会忽然顶住 —— 那是方头撞上了方孔的端面，等它对角。
     捏住刀轻转 10~20°，感觉「咔」地沉下去，再推到底。
     转不动时别用力拧：方头的角会啃方孔入口，把方孔入口磨圆。
  5) 到位的判断：推到底有硬限位 —— 方头端面顶到孔底，方孔入口的台阶挡住柄端面，
     两道限位同时到。径向只剩几度余量就撞死（单边旷量 {p.square_play:.3f}），
     不是自由转 360°。能 360° 自由转 = 方头没进方孔，退出来重插。
     孔底要清干净：那里若是有一层象脚或毛刺，方头顶不到底，
     柄身会停在方孔入口外侧，看着"到不了位"——用美工刀刮一下孔底就好。

四、为什么这样就能传扭
  ---------------------------------------------------------------
  方孔的四个侧面挡住方头的四个侧面，扭矩是「面顶面」，不是摩擦。
  咬合 {p.square_engage:.2f} mm，两个受力面各约 {p.square_size * p.square_engage:.1f} mm²，
  许用扭矩 {sq_t_lo:.2f} ~ {sq_t_hi:.2f} N·m（按 PETG 挤压 15 / 40 MPa）；
  手铰 Ø{p.shank_d:.2f} 只要 0.03 ~ 0.10 N·m。下限 15 MPa 是长期蠕变的保守值，
  短时按 30 MPa 算是 {sq_t_lo * 2:.2f} N·m，够用。
  而且这个余量不随时间衰减 —— 塑料蠕变只让旷量变大一点点，挡块还在。
  这正是胶接和顶丝做不到的：它们靠「持续压紧」。

  方头 {p.square_tip_len:.2f} 长、方孔 {p.square_len:.2f} 深，已经全部咬满，
  没有更多面可加了。想再多传扭只能换更长的方头（比如加个延长套），
  或者把孔底壁削到 2 mm、方孔做到 3 mm —— 但 3 mm 的孔底壁是给刀留的料，不建议动。

  上一版「手柄转、刀不转」的根因，就一句话：
      方头当时是在一个 Ø{p.bore_d:.2f} 的圆孔底上，只有四个角尖顶到孔壁，
      是点接触，一使劲就滑过去了。圆孔套方头，等于没有键槽。
      现在孔底是 {p.square_hole:.2f} 见方的方孔，四个面贴面，滑不动。

  两处分工 —— 就是你说的「把两个位置都固定住」：
      圆柱段 Ø{p.shank_d:.2f} 在 Ø{p.bore_d:.2f} 孔里，单边 {p.shank_radial_gap:.3f}，配合长 {p.square_start:.1f} mm
          → 最大摆动角 atan({p.shank_radial_gap:.3f} / {p.square_start:.1f}) ≈ {math.degrees(math.atan(p.shank_radial_gap / p.square_start)):.1f}°，刀杆被扶正，不会歪
          → 它不传扭，也不该传扭：圆孔套圆轴，只有摩擦
      方孔段 {p.square_hole:.2f}□ × {p.square_len:.2f} 深，方头整根埋进去
          → 四面贴面，扭矩全交给它

  孔径现在只看柄身，方头一点不参与：
      方头对角线实测 {p.square_envelope_d:.2f}，和柄径一样 —— 它整根内切在柄身的
      圆柱包络里，从过孔的角度看比柄身还「细」，不可能成为约束。
      所以 Ø{p.bore_d:.2f} 就是 {p.shank_d:.2f} + 2 × {p.bore_clearance:.2f}，改 bore_clearance 立刻生效：
          {p.shank_d + 0.02:.2f}（单边 0.01，几乎压入） / {p.shank_d + 0.06:.2f}（0.03，紧推） / {p.bore_d:.2f}（0.05，当前，轻推顺畅）
      早期版本以为「方头把孔撑到 Ø2.46」，那是拿两条边长 + 猜的倒圆推出来的
      假约束；跨角量一次对角线就没了。
"""
    else:
        fit_section = f"""三、装刀：Ø{p.shank_d:.2f} 柄压入 Ø{p.bore_d:.2f} 孔
  ---------------------------------------------------------------
  1) 冷透再装，等回室温。
  2) 清孔口 1~2 mm 的拉丝与毛刺；用 Ø{p.shank_d:.2f} 钻头手拧着过一遍，别上电钻。
  3) 试插：整根能进、手感发涩 → 正好；卡在一半 → 孔口有毛刺；整根进不去 → 砂纸修。
  4) 压入：抹一点洗洁精水或硅油，垂直匀速压到底，别用锤子敲。
  5) 验收：晃不动只说明装上了，能不能传扭是另一回事，见第四节。

四、让它真的能带动刀
  ---------------------------------------------------------------
  光滑圆柱套光滑圆柱，只有摩擦；贴合推入 = 接触压力≈0 = 摩擦扭矩≈0。
  两条路，A 或 B 单独做都行：

  【A】胶接 —— 乐泰 638 / 648，或环氧 AB 胶（别用 502）。
      粘接面 {p.bond_area_mm2:.0f} mm²，剪切 10~20 MPa → {t_lo:.1f} ~ {t_hi:.1f} N·m。

  【B】顶丝 + 平面 —— 磨一个深 {p.shank_flat_depth:.2f} 的小平面，M3 顶丝顶住。
      靠「平面挡住螺钉」，扭矩不随时间衰减，松顶丝就能抽刀。
"""

    text = f"""铰刀手柄 · 打印与装配说明
======================================================================
档案     : {slug}
模型文件 : {rel_out}/{STEP_NAME}   /   {rel_out}/{STL_NAME}
外形尺寸 : {info['bbox'][0]:.2f} × {info['bbox'][1]:.2f} × {info['bbox'][2]:.2f} mm
           （X 长 / Y 宽 / Z 高）
体积     : {info['volume_mm3'] / 1000:.2f} cm³      估算质量 : {mass:.1f} g
           （{m.name}，密度 {m.density} g/cm³）
插孔     : Ø{p.bore_d:.2f} × {p.pass_len:.2f} 深，孔口 {p.mouth_chamfer:.2f} × 45° 倒角
柄身     : Ø{p.shank_d:.2f}（游标卡尺实测）
{sq_line}{neck_line}{screw_line}{sq_fit_line}
  注：Ø{p.bore_d:.2f} 就是「柄径 + 2 × 单边间隙」，方头一点没参与。
      方头对角线实测 {p.square_envelope_d:.2f}，和柄径一样粗 —— 它整根内切在柄身的
      圆柱包络里，从过孔的角度看比柄身还细，不可能是约束。
      所以想让柄身更紧、更松，改 bore_clearance 一个数就行，不用动方头。

一、摆放：平放，不旋转，不加支撑
  STL 已经是打印姿态，拖进切片软件直接切，不要点“放平”也不要旋转。
  八边形躺倒后，其中一个平面整条贴床：{p.length:.2f} mm 长 × {p.side_len:.2f} mm 宽。
  两侧面相对竖直面是 45° 斜面 —— 45° 正好是免支撑临界角，
  所以整个件从头到尾不需要支撑，也不需要开树状支撑。
  打印高度只有 {p.across_flats:.2f} mm（{p.layers} 层）。

  贴床是一条窄带（{p.length:.1f} × {p.side_len:.1f}），长宽比 3.6:1，
  冷却收缩应力全压在这条窄带上，brim 一定要开：5 mm 宽，间距 0.1。
  {m.name} 比 PLA 更吃 brim，别省这一步。
{neck_note}

二、切片（{m.name}）
  喷嘴 {m.nozzle} / 热床 {m.bed} / 风扇 {m.fan}
  层高 {p.layer_height} / 墙 3 圈 / 顶底 4 层 / 填充 15% / 外壁 ≤ {m.wall_speed} mm/s

  1) 打之前先烘干：{m.dry}
     受潮的 {m.name} 会在孔壁拉出一片绒毛状的丝，孔径和方孔直接失真。
  2) 脱模：{m.release}
  3) 回抽别省：1.2 mm / 30 mm/s，最好开“只在跨区移动时回抽”。
     {m.name} 渗料厉害，孔口那圈拉丝会变成装配时的毛刺。
  4) 外壁速度压到 {m.wall_speed} mm/s 以内、关掉自适应层高，孔径重复性最好。
     同一个模型换三次速度打出来能差 0.05，所以定一个配置就别改。
  5) 别用“可变线宽 / 消除细小特征”这类优化，会把小孔算成椭圆或直接填掉。

  让位圆孔是水平孔，孔顶做了 45° 泪滴让位（不是纯圆孔）：顶部变成可打印的
  斜悬垂，而不是一段必然下垂的桥接。{m.name} 桥接能力比 PLA 弱，别关 teardrop。

  方孔不用另做处理：它的截面是「正方 + 45° 尖顶」，跟泪滴同一个道理 ——
  平顶方孔那 {p.square_hole:.1f} mm 宽的天花板在 {p.square_len:.1f} mm 的长度上纯靠桥接，
  {m.name} 会把它塌成腰形，方头就塞不进去了。改成尖顶后整段孔顶都是 45°。

{fit_section}
  插入 {p.insert_depth:.2f} mm（方头到底），柄身外露 {p.exposed_shank:.2f} mm；
  铰刀全长按 {p.reamer_total_len:.0f} mm 估，装好后刀杆水平外伸约 {p.reamer_protrusion:.0f} mm。

五、打完先量方孔（一分钟的事，决定下一件准不准）
  ---------------------------------------------------------------
  卡尺量方孔边长，量孔口那一侧的四条边，取最小的那个：
{table}

  然后把偏差写回 params.py：
      square_clearance = (实测方孔边长 - {p.square_size:.2f}) / 2
  例：实测 2.01，方头 1.90 → square_clearance = 0.055，取 0.06。
  （方头尺寸如果和 1.90 对不上，先改 square_size，改完其它都跟着走。）

  换打印机、换耗材、换颜色、换层高都会让这个数变，所以要重新量。
  {m.name} 还有个特点：方孔会随打印速度轻微变化（打得越快越小），
  想稳定就固定一个切片配置，别每次临时调。
  方孔最容易坏的是四个角 —— 塌了就用方锉轻轻过一下角，别碰四个平面。

六、改尺寸
  改 codes/params.py 后重跑 codes/handle_build.py，再跑 codes/render_preview.py。
  常用几个：
    length           总长（{p.length:.0f}），手捏段多长就填多少
    across_flats     对边距（{p.across_flats:.0f}），10 = 食指拇指自然捏合，12 = 偏粗
    square_size      方头边长（{p.square_size:.2f}），实测值（取大的那条边）
    square_size_min  方头另一条边（{p.square_size_min:.2f}），实测值
    square_diag_meas 方头对角线（{p.square_diag_meas:.2f}），卡尺跨角量，决定过孔
    square_tip_len   方头长度（{p.square_tip_len:.2f}），实测值，用来校核方孔深度
    square_clearance 方孔松紧（{p.square_clearance:.2f} → 方孔 {p.square_hole:.2f}）
    square_len       方孔深度（{p.square_len:.2f}），必须 ≤ 方头长 + 底隙
    bore_clearance   柄身松紧（{p.bore_clearance:.2f} → 孔 Ø{p.bore_d:.2f}）
    material         耗材（{p.material}），PETG / PLA，影响说明和估重
    floor_thickness  孔底壁厚（{p.floor_thickness:.1f}）
  （四角倒圆 R{p.square_corner_r:.2f} 是由对角线反推的，不用手改。）
  命令行临时改：python codes/handle_build.py --set across_flats=11
======================================================================
"""
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="铰刀手柄参数化建模（平放打印）")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="临时覆盖 params.py 里的参数，例如 --set length=18",
    )
    ap.add_argument(
        "--reamer",
        default=DEFAULT_PROFILE,
        metavar="NAME",
        help=f"用哪份档案（reamers/<NAME>.json），默认 {DEFAULT_PROFILE}；"
             "输出进 output/<NAME>/",
    )
    ap.add_argument("--no-export", action="store_true", help="只建模不导出，用于调试")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    p, slug = load_profile(args.reamer)
    out_dir = OUT_DIR / slug

    overrides: dict[str, object] = {}
    known = {f.name for f in fields(HandleParams)}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到 {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in known:
            raise SystemExit(f"未知参数 {k!r}，可选：{', '.join(sorted(known))}")
        cur = getattr(p, k)
        raw = v.strip()
        if isinstance(cur, bool):
            overrides[k] = raw.lower() in ("1", "true", "yes", "on")
        else:
            overrides[k] = float(raw)
    if overrides:
        p = p.with_overrides(**overrides)

    print(f"档案：{slug}   →   output/{slug}/")
    print(p.summary())

    t0 = time.time()
    try:
        body = build_handle(p)
    except ValueError as exc:
        sys.stdout.flush()          # 保证管道里错误信息排在摘要后面
        print(f"\n{exc}", file=sys.stderr)
        return 2
    print(f"\n建模完成，用时 {time.time() - t0:.2f} s")

    info = report(p, body)

    if not args.no_export:
        out_dir.mkdir(parents=True, exist_ok=True)
        step_path = out_dir / STEP_NAME
        stl_path = out_dir / STL_NAME
        export_step(body, str(step_path))
        export_stl(body, str(stl_path), tolerance=0.004, angular_tolerance=0.08)
        print(f"\n已导出：\n  {step_path}\n  {stl_path}"
              f"\n  {step_path.stat().st_size/1024:.1f} KB / "
              f"{stl_path.stat().st_size/1024:.1f} KB")
        notes = write_print_notes(p, info, out_dir, slug)
        print(f"  {notes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
