"""
铰刀手柄 —— 渲染预览图（与建模代码分开）
==========================================

读取 handle_build.build_handle() 的实体，输出 PNG 到 ../png/<档案名>/。

    python codes/render_preview.py                    # 默认档案 r01-2.36
    python codes/render_preview.py --reamer r02-xxx   # 换一把刀

为什么不用 matplotlib 的 3D / OpenGL：
    手柄带深盲孔 + 半剖，matplotlib 的画家算法（按三角形平均深度排序）
    会把孔内的面画到外壁前面，剖切面也会串色。
    所以这里自带一个极简正交 z-buffer 光栅化器：纯 numpy，无显卡依赖，
    Windows / Linux / macOS 结果完全一致，可复现。
    matplotlib 只负责在渲染好的位图上叠 2D 文字与尺寸线。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build123d import (  # noqa: E402
    Align,
    Box,
    BuildPart,
    BuildSketch,
    Cone,
    Cylinder,
    Plane,
    Pos,
    Rectangle,
    RectangleRounded,
    Vector,
    extrude,
)
from params import DEFAULT, DEFAULT_PROFILE, load_profile  # noqa: E402
from handle_build import _sketch_plane, build_handle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PNG_DIR = ROOT / "png"

# ---------------------------------------------------------------- 配色
BG_TOP = np.array([0.984, 0.988, 0.992])
BG_BOT = np.array([0.921, 0.933, 0.947])
INK = "#252a31"
MUTED = "#6b7280"
FAINT = "#9aa1ab"
BASE = np.array([0.640, 0.715, 0.790])   # 本体：淡蓝灰
CUT = np.array([0.945, 0.690, 0.450])    # 剖切面：暖橙
STEEL = np.array([0.455, 0.500, 0.555])  # 铰刀：钢灰
FACE_BG = np.array([0.958, 0.964, 0.972])  # 多视图平底色

LIGHT = np.array([-0.40, -0.55, 0.74])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
FILL = np.array([0.62, 0.68, 0.12])       # 补光，避免背光面死黑
FILL = FILL / np.linalg.norm(FILL)


# ======================================================================
# 1. 网格提取
# ======================================================================
def _norm(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def mesh(shape, tol=0.012, ang=0.20, cut_axis="y"):
    """返回 (V, F, N, is_cut)。逐面细分、统一法向朝外、面内平滑法向。

    cut_axis 指定哪个方向的平面剖切面要标成橙色：'y' = 沿轴线纵剖，
    'z' = 水平开盖剖（俯视看内部方孔用）。
    """
    Vs, Fs, Ns, cuts = [], [], [], []
    off = 0

    for f in shape.faces():
        try:
            v, t = f.tessellate(tol, ang)
        except Exception:  # noqa: BLE001
            continue
        if not t:
            continue

        V = np.array([[q.X, q.Y, q.Z] for q in v], dtype=float)
        F = np.array(t, dtype=int)
        tri = V[F]
        fn = _norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))

        fnormal = None
        try:
            c = tri[0].mean(axis=0)
            nv = f.normal_at(Vector(*c))
            fnormal = np.array([nv.X, nv.Y, nv.Z])
            if float(fn[0] @ fnormal) < 0.0:
                F = F[:, ::-1]
                tri = V[F]
                fn = -fn
        except Exception:  # noqa: BLE001
            pass

        N = np.zeros_like(V)
        for k in range(3):
            np.add.at(N, F[:, k], fn)
        ln = np.linalg.norm(N, axis=1)
        N = N / np.maximum(ln[:, None], 1e-12)
        if fnormal is not None:
            N[ln < 1e-9] = fnormal

        if cut_axis == "z":
            z0 = float(tri[0, 0, 2])
            is_cut = bool(
                fnormal is not None
                and abs(abs(fnormal[2]) - 1.0) < 1e-6
                and z0 > 1.0                      # 排除贴床的底面
                and np.all(np.abs(tri[:, :, 2] - z0) < 1e-6)
            )
        else:
            is_cut = bool(
                fnormal is not None
                and abs(abs(fnormal[1]) - 1.0) < 1e-6
                and np.all(np.abs(tri[:, :, 1]) < 1e-6)
            )

        Vs.append(V)
        Fs.append(F + off)
        Ns.append(N)
        cuts.append(np.full(len(F), is_cut, dtype=bool))
        off += len(V)

    return np.vstack(Vs), np.vstack(Fs), np.vstack(Ns), np.concatenate(cuts)


# ======================================================================
# 2. 正交相机
# ======================================================================
class Cam:
    def __init__(self, elev=0.0, azim=0.0, size=(1200, 1600), ss=2, margin=0.09):
        e, a = np.radians(elev), np.radians(azim)
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.f = d / np.linalg.norm(d)          # 由物体指向观察者
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(self.f @ up)) > 0.999:
            up = np.array([0.0, 1.0, 0.0])
        self.r = np.cross(up, self.f)
        self.r /= np.linalg.norm(self.r)        # 屏幕向右
        self.u = np.cross(self.f, self.r)       # 屏幕向上
        self.size = size
        self.ss = ss
        self.W, self.H = size[0] * ss, size[1] * ss   # 超采样画布
        self.margin = margin
        self.ppm = 1.0
        self.off = np.zeros(2)
        self.off3 = np.zeros(3)

    def _proj(self, V):
        d = np.asarray(V, float) - self.off3
        return np.stack([d @ self.r * self.ppm, -d @ self.u * self.ppm], axis=1) + self.off

    def depth(self, V):
        return (np.asarray(V, float) - self.off3) @ self.f

    def project(self, V):
        return self._proj(V) / self.ss

    def fit(self, pts, ppm=None, pad_bottom=0.0):
        """按一组点定比例并居中；pad_bottom 预留底部像素带放标题。"""
        pts = np.asarray(pts, float)
        self.off3 = np.array([0.0, 0.0, 0.5 * (pts[:, 2].min() + pts[:, 2].max())])
        self.ppm, self.off = 1.0, np.zeros(2)
        p = self._proj(pts)
        w = self.W * (1 - 2 * self.margin)
        h = (self.H - pad_bottom * self.ss) * (1 - 2 * self.margin)
        if ppm is None:
            self.ppm = min(w / max(np.ptp(p[:, 0]), 1e-6), h / max(np.ptp(p[:, 1]), 1e-6))
        else:
            self.ppm = ppm
        p = self._proj(pts)
        self.off = np.array([self.W / 2.0,
                             (self.H - pad_bottom * self.ss) / 2.0]) \
            - (p.min(axis=0) + p.max(axis=0)) / 2.0
        return self


# ======================================================================
# 3. z-buffer 光栅化
# ======================================================================
def render(items, cam: Cam, bg=True):
    """items: [(V, F, N, base_color, opts), ...]，共用一个深度缓冲。"""
    W, H = cam.W, cam.H

    if isinstance(bg, np.ndarray):
        img = np.repeat(np.repeat(bg.reshape(1, 1, 3), W, axis=1), H, axis=0).astype(np.float32)
    elif bg:
        t = np.linspace(0, 1, H)[:, None, None]
        img = (BG_TOP[None, None, :] * (1 - t) + BG_BOT[None, None, :] * t).repeat(W, axis=1)
        img = img.astype(np.float32)
    else:
        img = np.ones((H, W, 3), np.float32)

    zbuf = np.full((H, W), -1e18, np.float32)
    half = LIGHT + cam.f
    half /= np.linalg.norm(half)

    for V, F, N, color, opt in items:
        alpha = opt.get("alpha", 1.0)
        amb, dif = opt.get("amb", 0.40), opt.get("dif", 0.62)
        spec, shin = opt.get("spec", 0.18), opt.get("shin", 42)
        fill = opt.get("fill", 0.24)

        scr = cam._proj(V)
        dep = cam.depth(V)
        base = np.asarray(color, float).reshape(-1, 3)
        if base.shape[0] == 1:
            base = np.repeat(base, len(F), axis=0)

        for i in range(len(F)):
            ia, ib, ic = F[i]
            P = scr[[ia, ib, ic]]
            x0 = max(int(np.floor(P[:, 0].min())), 0)
            x1 = min(int(np.ceil(P[:, 0].max())), W - 1)
            y0 = max(int(np.floor(P[:, 1].min())), 0)
            y1 = min(int(np.ceil(P[:, 1].max())), H - 1)
            if x1 < x0 or y1 < y0:
                continue

            ax_, ay = P[0]
            bx, by = P[1]
            cx, cy = P[2]
            den = (by - cy) * (ax_ - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue

            xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
            ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
            X, Y = np.meshgrid(xs, ys)

            w0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
            w1 = ((cy - ay) * (X - cx) + (ax_ - cx) * (Y - cy)) / den
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue

            z = w0 * dep[ia] + w1 * dep[ib] + w2 * dep[ic]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (z > sub)
            if not upd.any():
                continue

            nx = w0 * N[ia, 0] + w1 * N[ib, 0] + w2 * N[ic, 0]
            ny = w0 * N[ia, 1] + w1 * N[ib, 1] + w2 * N[ic, 1]
            nz = w0 * N[ia, 2] + w1 * N[ib, 2] + w2 * N[ic, 2]
            nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
            nx, ny, nz = nx / nl, ny / nl, nz / nl

            lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
            lam2 = np.clip(nx * FILL[0] + ny * FILL[1] + nz * FILL[2], 0, 1)
            sp = np.clip(nx * half[0] + ny * half[1] + nz * half[2], 0, 1) ** shin * spec
            shade = (amb + dif * lam + fill * lam2 + sp)[upd]

            col = base[i][None, :] * shade[:, None]
            if alpha < 1.0:
                col = col * alpha + img[y0:y1 + 1, x0:x1 + 1][upd] * (1 - alpha)
            img[y0:y1 + 1, x0:x1 + 1][upd] = np.clip(col, 0, 1)
            sub[upd] = z[upd]

    if cam.ss > 1:
        s = cam.ss
        img = img.reshape(H // s, s, W // s, s, 3).mean(axis=(1, 3))
    return np.clip(img, 0, 1)


# ======================================================================
# 4. 出图工具
# ======================================================================
def figure_from(img, dpi=200):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=(0, w, h, 0), interpolation="lanczos")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    path = PNG_DIR / name
    fig.savefig(path, dpi=fig.dpi, facecolor="#f2f4f7")
    plt.close(fig)
    print(f"  {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return path


def caption(ax, lines, H, x=36, bottom=38):
    """左下角标题块（像素坐标，从下往上排，行距按字号换算）。"""
    dpi = ax.figure.dpi
    y = float(H - bottom)
    for txt, size, color, weight in reversed(lines):
        ax.text(x, y, txt, fontsize=size, color=color, ha="left", va="baseline",
                weight=weight, zorder=6)
        y -= size * dpi / 72.0 * 1.62


# ======================================================================
# 5. 铰刀示意件（沿 -X 方向伸出）
# ======================================================================
def build_reamer(p=DEFAULT):
    r = p.shank_d / 2.0
    # 柄身末端：方孔方案下停在方孔入口，剩下的深度由方头占掉
    x_back = p.shank_end_x if p.square_on else (p.socket_len - p.socket_floor_gap)
    x_front = x_back - p.shank_len                      # 柄前端
    flute_len = p.reamer_total_len - p.shank_len

    shank = Pos(x_front, 0, p.hole_z) * Cylinder(
        radius=r, height=p.shank_len,
        rotation=(0, 90, 0), align=(Align.CENTER, Align.CENTER, Align.MIN))
    flute = Pos(x_front, 0, p.hole_z) * Cone(
        bottom_radius=r, top_radius=r - 0.12,
        height=flute_len - 3.0,
        rotation=(0, -90, 0), align=(Align.CENTER, Align.CENTER, Align.MIN))
    tip = Pos(x_front - (flute_len - 3.0), 0, p.hole_z) * Cone(
        bottom_radius=r - 0.12, top_radius=0.38, height=3.0,
        rotation=(0, -90, 0), align=(Align.CENTER, Align.CENTER, Align.MIN))
    body = shank + flute + tip

    if p.square_on:
        # 尾部方头 —— 本案的扭矩全靠它。
        # 四角按实测对角反推的半径倒圆：尖角 1.9×1.8 的对角是 2.62，
        # 而实测只有 2.36，差的那截全在角上，画成尖角就假了。
        rr = min(p.square_corner_r, min(p.square_size, p.square_size_min) / 2 - 1e-3)
        with BuildPart() as head:
            with BuildSketch(_sketch_plane(p).offset(x_back)):
                if rr > 1e-4:
                    RectangleRounded(p.square_size, p.square_size_min, radius=rr)
                else:
                    Rectangle(p.square_size, p.square_size_min)
            extrude(amount=p.square_tip_len)
        body = body + head.part

    if p.set_screw_hole:
        # 顶丝顶住的那块小平面（实际要在铰刀柄上自己磨出来）
        zb = p.hole_z + r - p.shank_flat_depth
        body = body - Pos(p.set_screw_x, 0, zb + 4.0) * Box(4.0, 2 * r + 4.0, 8.0)
    return body


def bounds_pts(shape):
    bb = shape.bounding_box()
    return np.array([[x, y, z] for x in (bb.min.X, bb.max.X)
                     for y in (bb.min.Y, bb.max.Y)
                     for z in (bb.min.Z, bb.max.Z)])


# ======================================================================
# 6. 各视图
# ======================================================================
def view_iso(handle, p):
    V, F, N, _ = mesh(handle)
    pad = 118
    cam = Cam(elev=24, azim=-126, size=(1300, 1120)).fit(bounds_pts(handle), pad_bottom=pad)
    img = render([(V, F, N, BASE, {})], cam)
    fig, ax = figure_from(img)
    caption(ax, [
        ("REAMER HANDLE", 16, INK, "bold"),
        (f"{p.length:.0f} mm long  ·  {p.across_flats:.0f} mm across flats"
         f"  ·  Ø{p.bore_d:.2f} bore"
         + (f" + {p.square_hole:.2f} square drive" if p.square_on else ""),
         10, MUTED, "normal"),
        ("printed lying flat  —  no supports", 10, MUTED, "normal"),
    ], img.shape[0])
    return save(fig, "01_handle_iso.png")


def view_orthographic(handle, p):
    V, F, N, _ = mesh(handle)
    pts = bounds_pts(handle)
    panels = [(0, 180, "MOUTH (X)", (560, 500)),
              (0, -90, "SIDE (Y)", (560, 500)),
              (90, -90, "TOP (Z)", (560, 500))]
    cams = [Cam(elev=el, azim=az, size=sz, margin=0.13).fit(pts)
            for el, az, _, sz in panels]
    ppm = min(c.ppm for c in cams)      # 三视图同一比例
    subs = []
    for c in cams:
        c.fit(pts, ppm=ppm)
        subs.append(render([(V, F, N, BASE, {})], c, bg=FACE_BG))

    pw, ph = subs[0].shape[1], subs[0].shape[0]
    band, foot = 62, 46
    W, H = pw * 3, ph + band + foot
    img = np.repeat(np.repeat(FACE_BG.reshape(1, 1, 3), W, axis=1), H, axis=0)
    for i, s in enumerate(subs):
        img[band:band + ph, i * pw:(i + 1) * pw] = s

    fig, ax = figure_from(img, dpi=190)
    for i, (_, _, t, _) in enumerate(panels):
        ax.text(pw * (i + 0.5), 34, t, fontsize=10, color=MUTED,
                ha="center", va="center")
    ax.text(W / 2, H - 22, "ORTHOGRAPHIC  ·  three views at the same scale  ·  mm",
            fontsize=9, color=FAINT, ha="center", va="center")
    return save(fig, "02_orthographic.png")


def _cut_shapes(handle, reamer):
    cut = Pos(0, -80, 6.0) * Box(260, 160, 420)
    return handle - cut, reamer - cut


def view_section(handle, reamer, p):
    hs, rs = _cut_shapes(handle, reamer)
    Vh, Fh, Nh, ch = mesh(hs)
    Vr, Fr, Nr, _ = mesh(rs)

    pad = 118
    frame = bounds_pts(hs)
    cam = Cam(elev=12, azim=-104, size=(1250, 1000)).fit(frame, pad_bottom=pad)
    img = render([
        (Vh, Fh, Nh, np.where(ch[:, None], CUT, BASE), {}),
        (Vr, Fr, Nr, STEEL, {"alpha": 0.99, "amb": 0.44, "dif": 0.55,
                             "spec": 0.34, "shin": 60}),
    ], cam)
    fig, ax = figure_from(img)
    caption(ax, [
        ("SECTION  A–A", 15, INK, "bold"),
        (f"Ø{p.bore_d:.2f} pass bore × {p.pass_len:.2f}"
         + (f"   ·   {p.square_hole:.2f} square × {p.square_len:.2f} deep"
            if p.square_on else "")
         + f"   ·   floor {p.floor_thickness:.1f}", 9.5, MUTED, "normal"),
        (f"reamer {p.shank_len:.0f} mm shank"
         + (f" + {p.square_size:.1f} sq drive" if p.square_on else "")
         + f",  inserted {p.insert_depth:.1f} mm", 9.5, MUTED, "normal"),
    ], img.shape[0])
    return save(fig, "03_section.png")


def view_dimensioned(handle, reamer, p):
    """正视图半剖 + 尺寸标注（在渲染位图上按像素排版）。"""
    hs, rs = _cut_shapes(handle, reamer)
    Vh, Fh, Nh, ch = mesh(hs)
    Vr, Fr, Nr, _ = mesh(rs)

    W, H = 1640, 1000
    cam = Cam(elev=0, azim=-90, size=(W, H), margin=0.255).fit(bounds_pts(hs))
    img = render([
        (Vh, Fh, Nh, np.where(ch[:, None], CUT, BASE), {}),
        (Vr, Fr, Nr, STEEL, {"alpha": 0.99, "amb": 0.46, "dif": 0.54,
                             "spec": 0.26, "shin": 60}),
    ], cam)
    fig, ax = figure_from(img)
    P = lambda x, z: cam.project(np.array([[x, 0.0, z]], float))[0]

    def txt(x, y, t, size=9.5, color=INK, ha="left", va="center", rot=0, bbox=False):
        ax.text(x, y, t, fontsize=size, color=color, ha=ha, va=va, rotation=rot,
                zorder=7, family="DejaVu Sans",
                bbox=dict(fc="white", ec="none", alpha=0.88, pad=2.0) if bbox else None)

    def tick(x, y, s=6, color=INK):
        ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=0.9, zorder=6)

    x0, yb = P(0.0, 0.0)
    x1 = P(p.length, 0.0)[0]
    yt = P(0.0, p.across_flats)[1]
    cx = (x0 + x1) / 2
    w = x1 - x0

    # --- 1. 总长（下）
    yd = yb + 66
    for x in (x0, x1):
        ax.plot([x, x], [yb + 9, yd + 9], color=FAINT, lw=0.7, ls=(0, (5, 3)), zorder=5)
    ax.plot([x0, x1], [yd, yd], color=INK, lw=0.9, zorder=6)
    tick(x0, yd)
    tick(x1, yd)
    txt(cx, yd + 20, f"{p.length:.2f}", 11, ha="center", va="top", bbox=True)

    # --- 2. 对边距（左）
    xd = x0 - 78
    xref = P(p.neck_end + 2.5, 0.0)[0] if p.neck_on else x0
    for y in (yb, yt):
        ax.plot([xref - 9, xd - 9], [y, y], color=FAINT, lw=0.7, ls=(0, (5, 3)), zorder=5)
    ax.plot([xd, xd], [yb, yt], color=INK, lw=0.9, zorder=6)
    tick(xd, yb)
    tick(xd, yt)
    txt(xd - 21, (yb + yt) / 2, f"{p.across_flats:.2f} A/F", 10, ha="center", rot=90,
        bbox=True)

    # --- 3. 孔系长度 / 孔底壁厚（上，链式）
    ya = yt - 62
    xs = P(p.square_start if p.square_on else p.socket_len, 0.0)[0]
    xe = P(p.socket_len, 0.0)[0]
    for x in (x0, xs, xe, x1):
        ax.plot([x, x], [yt - 9, ya - 9], color=FAINT, lw=0.7, ls=(0, (5, 3)), zorder=5)
    ax.plot([x0, x1], [ya, ya], color=INK, lw=0.9, zorder=6)
    for x in (x0, xs, xe, x1):
        tick(x, ya)
    if p.square_on:
        txt((x0 + xs) / 2, ya - 18, f"\u00d8{p.bore_d:.2f} \u00d7 {p.pass_len:.2f}",
            9.2, ha="center", va="bottom", bbox=True)
        # 方孔那一段只有 2 mm 宽，标签放上面才不和「孔底壁厚」挤在一起
        txt((xs + xe) / 2, ya + 16, f"{p.square_hole:.2f}\u25a1 \u00d7 {p.square_len:.2f}",
            9.0, ha="center", va="bottom", bbox=True)
    else:
        txt((x0 + xe) / 2, ya - 18, f"\u00d8{p.bore_d:.2f} \u00d7 {p.socket_len:.2f}",
            10, ha="center", va="bottom", bbox=True)
    txt((xe + x1) / 2, ya - 18, f"{p.floor_thickness:.2f}", 10, ha="center",
        va="bottom", bbox=True)

    # --- 3b. 孔口收口（引线 + 文字）
    if p.neck_on:
        px, py = P(0.0, p.hole_z + p.mouth_neck_flats / 2)
        ly = yt - 26
        ax.annotate("", xy=(px + 24, py + 10), xytext=(px + 42, ly + 9),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.85,
                                    shrinkA=0, shrinkB=0), zorder=6)
        ax.plot([px + 24], [py + 10], marker="o", ms=2.2, color=INK, zorder=7)
        txt(px + 46, ly, f"neck \u00d8{p.mouth_neck_flats:.2f} A/F "
                        f"\u00d7 {p.mouth_neck_len:.2f}", 9.2, ha="left", va="center",
            bbox=True)

    # --- 4. 孔顶泪滴（在孔内画一条短虚线标出孔顶，不引出图外穿本体）
    ry = P(0.0, p.bore_top_z)[1]
    rx0 = P(p.pass_len * 0.22, p.bore_top_z)[0]
    rx1 = P(p.pass_len * 0.96, p.bore_top_z)[0]
    ax.plot([rx0, rx1], [ry, ry], color=INK, lw=0.8, ls=(0, (7, 3)), zorder=6)
    txt(rx0, ry - 13, "teardrop roof 45\u00b0", 8.8, ha="left", va="bottom", bbox=True)

    # --- 4b. 孔底方孔（引线，从孔底往下引，避开上方标注）
    if p.square_on:
        sy = P(0.0, p.hole_z - p.square_hole / 2)[1]
        sx = P(p.square_start + p.square_len * 0.55,
               p.hole_z - p.square_hole / 2)[0]
        ax.plot([sx, sx], [sy, sy + 54], color=INK, lw=0.8, zorder=6)
        ax.plot([sx], [sy], marker="o", ms=2.2, color=INK, zorder=7)
        txt(sx + 8, sy + 54,
            f"square drive {p.square_hole:.2f} \u00d7 {p.square_len:.2f}"
            f"   (for {p.square_size:.1f} sq head)", 8.8,
            ha="left", va="center", bbox=True)

    # --- 5. 顶丝孔（引线）
    if p.set_screw_hole:
        hx = P(p.set_screw_x, 0.0)[0]
        ax.plot([hx, hx], [yt - 4, yt - 32], color=INK, lw=0.85, zorder=6)
        ax.plot([hx], [yt - 4], marker="o", ms=2.2, color=INK, zorder=7)
        txt(hx + 10, yt - 32, "M3 set screw", 8.8, ha="left", va="center", bbox=True)

    caption(ax, [("REAMER HANDLE  \u00b7  SECTION A\u2013A  \u00b7  mm", 12, INK, "bold"),
                 (f"printed flat:  bed {p.length:.1f} \u00d7 {p.side_len:.2f}"
                  f"   \u00b7   mouth chamfer {p.mouth_chamfer:.2f} \u00d7 45\u00b0"
                  + (f"   \u00b7   neck 45\u00b0 shoulder" if p.neck_on else ""),
                  9, MUTED, "normal")], H)
    return save(fig, "04_dimensioned.png")


def view_assembled(handle, reamer, p):
    Vh, Fh, Nh, _ = mesh(handle)
    Vr, Fr, Nr, _ = mesh(reamer)
    cam = Cam(elev=11, azim=-100, size=(2000, 470), margin=0.05)
    cam.fit(np.vstack([bounds_pts(handle), bounds_pts(reamer)]), pad_bottom=92)
    img = render([
        (Vh, Fh, Nh, BASE, {}),
        (Vr, Fr, Nr, STEEL, {"alpha": 0.99, "amb": 0.46, "dif": 0.54,
                             "spec": 0.35, "shin": 70}),
    ], cam)
    fig, ax = figure_from(img, dpi=190)
    caption(ax, [("ASSEMBLED", 13, INK, "bold"),
                 (f"grip {p.length:.0f} mm  ·  reamer ≈{p.reamer_total_len:.0f} mm"
                  f"  ·  inserted {p.insert_depth:.1f} mm"
                  + (f"  ·  {p.square_size:.1f} sq drive engaged {p.square_engage:.1f} mm"
                     if p.square_on else ""), 9.5, MUTED, "normal")],
            img.shape[0])
    return save(fig, "05_assembled.png")


# ======================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="铰刀手柄预览图（与档案一一对应）")
    ap.add_argument(
        "--reamer",
        default=DEFAULT_PROFILE,
        metavar="NAME",
        help=f"用哪份档案（reamers/<NAME>.json），默认 {DEFAULT_PROFILE}；出图到 png/<NAME>/",
    )
    args = ap.parse_args(argv)

    global PNG_DIR
    p, slug = load_profile(args.reamer)
    PNG_DIR = ROOT / "png" / slug

    print(f"档案：{slug}   →   png/{slug}/")
    print(p.summary())
    print("\n渲染中…")
    t0 = time.time()
    handle = build_handle(p)
    reamer = build_reamer(p)

    view_iso(handle, p)
    view_orthographic(handle, p)
    view_section(handle, reamer, p)
    view_dimensioned(handle, reamer, p)
    view_assembled(handle, reamer, p)

    print(f"\n完成 5 张，用时 {time.time() - t0:.1f} s → {PNG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
