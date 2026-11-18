"""
铰刀手柄 —— 参数配置（唯一需要手动改的文件）
==============================================

所有长度单位均为 mm。改完这里重跑 handle_build.py，渲染脚本会跟着变。

多把铰刀怎么办（一把刀一份档案）
--------------------------------
    这里放的是「手柄设计本身」的默认值（外形、间隙、打印参数），所有刀共用。
    每把刀自己的实测数据不进这里，而是存成 reamers/<名字>.json，只写实测那几项：

        reamers/
        ├── r01-2.36.json     柄 2.36，方头 1.90×1.80，刀长 57
        └── r02-xxx.json      第二把

    用法：
        python codes/handle_build.py --reamer r02-xxx
        python codes/render_preview.py --reamer r02-xxx

    输出各进各的目录（output/<名字>/、png/<名字>/），互不覆盖。
    不带 --reamer 时用 DEFAULT_PROFILE（= r01-2.36）。
    这样一份代码、多份数据，第二把打完还能回头再打第一把，参数永远不会被顶掉。


坐标约定（= 打印姿态，STL 按这个方向导出，切片时不要再转）
----------------------------------------------------------
    X 轴 = 手柄轴线
        x = 0              插刀口端面
        x = length         封闭端面
    Z 轴 = 打印高度
        z = 0              贴床的八边形平面
        z = across_flats   顶面
    截面：YZ 平面内的正八边形，对边距 across_flats，旋转 22.5° 让一个平面朝下。


配合方案（v1.0：方头对角 = 柄径 = Ø2.36，孔径回到只看柄身）
--------------------------------------------------------------
    实测数据（全部来自你手上那几个打印件 + 卡尺）：
        · 柄身圆柱    Ø2.36
        · 方头边长    1.90 × 1.80 × 2.00 长
        · 方头对角线  Ø2.36   ← 跨着两个角量，和柄身一样粗
        · Ø2.46 孔：柄身 + 方头一路插到底，手感「刚好贴合」
        · Ø2.60 孔：太松

    对角线 2.36 这个数把问题一次解掉了：
        卡尺两爪是平面，跨角量出来的就是「包络圆直径」，也就是方头的
        最大外廓。它等于柄径，说明方头整根内切在柄身的圆柱包络里 ——
        从过孔的角度看，方头根本不存在，它比柄身还「细」。

        所以孔径不需要为方头让位，回到一个纯柄身配合问题：
            孔径 = 柄径 + 2 × 单边间隙 = 2.36 + 2 × 0.05 = Ø2.46

    v0.9 曾认为「方头外廓 2.452 逼着孔径到 2.46」，那个 2.452 是从
    两条边长 + 猜的倒圆 R0.20 反推的。现在有对角线的直接实测值，
    不用再猜：外廓 = 2.36。结论仍是 Ø2.46，但理由完全不同 ——
    不是「孔被方头撑大」，而是 0.05 单边本来就是你要的手感。
    想收紧随时能改，见 bore_clearance。

    由对角线反推的四角倒圆：r ≈ 0.31（尖角对角是 2.62，差得不是一点点）。
    这么大的角修量说明方头四角是专门倒过的（铣削件常见做法），
    不是磨损 —— 反正它现在由实测值直接决定，代码不再依赖这个猜测。

    两处怎么分工：
        圆柱段  Ø2.36 在 Ø2.46 里，单边 0.05，配合长度 10 mm
                → 最大摆动角 atan(0.05 / 10) ≈ 0.3°，足够扶正刀杆
                → 它负责「不让刀歪」，不负责传扭
        方孔    2.00□ × 2.00 深，方头 2.00 整根埋进去
                → 四个面贴四个面，负责传扭，挡块不会随蠕变消失

    上一版为什么「手柄转、刀不转」：
        方头在 Ø2.46 圆孔底上只靠四个角尖顶住孔壁 —— 点接触，
        一使劲就滑过去。方孔把点接触变成面接触，才是真的挡块。

    方孔跨平面的尺寸只跟方头的「边长」有关，与对角无关：
        方头 1.90 边 → 方孔 2.00（= 1.90 + 2 × 0.05）就能进。
        1.80 那条边对应旷量 0.10，更松一点，但挡的还是那四个平面。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class MaterialProfile:
    """一种耗材的打印参数与物性。仅用于生成打印说明和估重，不参与几何。"""

    name: str
    density: float          # g/cm³
    nozzle: str             # 喷嘴温度
    bed: str                # 热床温度
    fan: str                # 风扇
    wall_speed: int         # 外壁速度上限 mm/s
    dry: str                # 打前干燥
    release: str            # 脱模 / 附着处理


MATERIALS: dict[str, MaterialProfile] = {
    "PETG": MaterialProfile(
        name="PETG",
        density=1.27,
        nozzle="235~245 ℃",
        bed="80 ℃",
        fan="30~50 %",
        wall_speed=40,
        dry="65 ℃ / 4 h（PETG 吸湿，受潮会气泡、拉丝、孔壁发毛）",
        release="光滑 PEI 板粘得太死，涂一层胶棒当脱模剂",
    ),
    "PLA": MaterialProfile(
        name="PLA",
        density=1.24,
        nozzle="205~215 ℃",
        bed="60 ℃",
        fan="100 %",
        wall_speed=60,
        dry="一般不需要",
        release="不需要，冷却后自然松脱",
    ),
}


@dataclass(frozen=True)
class HandleParams:
    """手柄全部参数。默认值 = 15 mm 手捏段 / Ø2.36 铰刀 / PETG / 平放打印。"""

    # ==================================================================
    # 1. 铰刀（被测件，按实测改）
    # ==================================================================
    shank_d: float = 2.36
    """铰刀柄部圆柱的实际直径（mm）。游标卡尺实测 2.36，孔径就按这个数算。"""

    shank_len: float = 15.0
    """柄部可用长度（mm）。只用于算插入后露多少，不参与建模。"""

    # ==================================================================
    # 2. 插孔
    # ==================================================================
    bore_clearance: float = 0.05
    """柄身的单边间隙（mm）→ 2.36 + 2 × 0.05 = Ø2.46。

    这就是唯一决定孔径的数了 —— 方头对角也是 2.36，不比柄身粗，
    过孔不再是约束（见文件头）。0.05 是你打出来觉得「刚好贴合」的那个手感。

        还想更紧 → 0.03（Ø2.42，单边 0.03，推入会有明显阻尼）
        觉得晃   → 0.07（Ø2.50，10 mm 配合长下摆动 0.4°）
        别低于   → 0.02，PETG 打印件的手压配合做不了这么紧，
                   差 0.01 就是插不进，还不如留点量
    """

    bore_pass_margin: float = 0.005
    """方头过孔的最小单边余量（mm）—— 现在**不参与**孔径决策。

    保留它是当保险：万一哪天方头换大了（对角线 > 柄径），孔径会自动
    被这项抬起来，不至于做出一个「柄身能进、方头卡死」的孔。
    当前 2.36 + 2×0.005 = Ø2.37 < Ø2.46，永远不是约束。"""

    socket_floor_gap: float = 0.30
    """柄端面与孔底之间的空隙（mm）。**只在没有方孔（square_drive=False）时生效。**
    有方孔时，柄身停在方孔入口，端面余量由 方孔深 − 方头长 决定（见 tip_floor_gap）。"""

    mouth_chamfer: float = 0.50
    """孔口 45° 导入倒角（mm）。压刀时不啃边，也方便对中。0 = 关闭。"""

    mouth_neck_flats: float = 7.0
    """孔口收口：把插刀口那一端的外形收成一个更小的正八边形（对边距，mm）。
    0 = 不收口，整根等截面。7.0 = 从 10 收到 7（端面看起来是个小八边形）。"""

    mouth_neck_len: float = 2.0
    """收口直段长度（mm）。这段之后以 45° 锥面过渡回 across_flats。
    45° 是平放打印的免支撑临界角，所以锥长 = (across_flats - mouth_neck_flats) / 2，
    不能更短（更短就是悬垂），也没必要更长（会长得像喇叭口）。
    直段越长，底部那点悬挑越大，1.0 以内基本无感。"""

    teardrop: bool = True
    """孔顶做 45° 泪滴让位。水平圆孔的顶部是一段「天花板」，打印时是纯桥接，
    必然下垂，会把 Ø2.36 的刀杆卡在顶部。改成两条 45° 斜边收成尖顶后，
    顶部变成可打印的悬垂，孔形不塌。强烈建议保持 True。"""

    # ------------------------------------------------------------------
    # 2b. 尾部方头 → 孔底方孔（真正的传扭方式）
    # ------------------------------------------------------------------
    square_drive: bool = True
    """铰刀尾部有方头，手柄孔底开方孔与它咬合。
    False = 回退到「纯圆孔 + 胶接/顶丝」的老方案。"""

    square_size: float = 1.90
    """方头边长（mm），实测值 —— 取两条边里大的那条（另一条 1.80）。
    方孔跨平面尺寸按这个数算，所以 1.90 是对的基准。"""

    square_size_min: float = 1.80
    """方头的另一条边（mm）。方头不是正方（1.90 × 1.80），
    只用于算「过孔外廓」和提示旷量不均匀，不参与方孔尺寸。"""

    square_diag_meas: float = 2.36
    """方头对角线实测值（mm）—— **卡尺跨两个对角量出来的数**，不是算出来的。

    2.36 = 柄径。这不是巧合：卡尺两爪是平面，跨角量到的是「包络圆直径」，
    也就是方头的最大外廓。它等于柄径，意味方头整根内切在柄身的圆柱里，
    所以插孔不用为它让位。

    这一项现在是方头几何的**唯一上限依据**：
        · 过孔校核用它（不用再猜倒圆半径）
        · 四角倒圆半径由它反推（≈ 0.31，见 square_corner_r）
        · 尖角对角 2.62 只是理论参考，实物根本不到那个数

    改动场景：换一把刀、方头尺寸变了 → 重新量这条对角线填进来。
    """

    square_tip_len: float = 2.00
    """方头沿轴线的长度（mm），从方头端面量到它与 Ø2.36 圆柱交界的台阶。

    实测 2.00。这个数决定方头能咬合多深：方孔做 2.00、方头也 2.00，
    方头整段埋进方孔，四个面全接触。
    方头比方孔长 → 顶在方孔入口台阶上，柄身进不到底，程序会报错；
    方头比方孔短 → 只是端面留个空隙，不影响咬合。"""

    square_clearance: float = 0.05
    """方孔单边间隙（mm）→ 方孔边长 = 1.90 + 2 × 0.05 = 2.00（整数，好记）。

    方头实测 1.90 / 1.80，2.00 的方孔对 1.90 那条边是单边 0.05，
    对 1.80 那条边是 0.10 —— 两边旷量不一样，但都能进，挡的还是四个平面。
    0.05 打出来约 0.03：几乎不空程。塞不进 → 0.08（方孔 2.06）；
    空程明显 → 0.03（方孔 1.96）。"""

    square_len: float = 2.00
    """方孔轴向深度（mm）。紧跟孔底壁（3 mm）往孔口方向做 2 mm，其余是圆孔。

    几何链条（从孔底往孔口数）：
        3.00 孔底壁  →  2.00 方孔  →  10.00 圆孔  →  x = 0 孔口
    柄身末端正好停在方孔入口 x = 10.00：柄身 Ø2.46 比方孔 2.00 大，
    本来也进不去，停在入口是唯一合理的姿态。
    于是方头 2.00 整根落在方孔 2.00 里，咬合 = 2.00，不是 1.70。

    做深只多咬合，做浅会挡不住方头。想加余量就同步加 square_tip_len。"""

    # ------------------------------------------------------------------
    # 2c. 顶丝（可选：只做轴向防脱，不再负责传扭）
    # ------------------------------------------------------------------
    set_screw_hole: bool = False
    """顶面 M3 顶丝孔，顶住柄身，防止刀被轴向拉出来。

    有了方孔就不靠它传扭了（方孔是形状约束，扭矩不随时间衰减）。
    默认关闭：顶丝会把柄身压向孔壁一侧，给铰刀带来约 3° 的微倾斜，
    手铰没必要。想要「刀绝对掉不出来」再打开。"""

    set_screw_d: float = 2.70
    """顶丝孔直径（mm）。2.70 = M3 自攻顶丝直接拧；要先攻丝就改 2.50。"""

    set_screw_x: float = 4.00
    """顶丝孔位置（mm，从插刀口端面量起）。必须落在让位孔段（x < 方孔入口），
    压在方孔上会把方孔顶部的尖顶切穿。"""

    shank_flat_depth: float = 0.35
    """刀柄上要磨的小平面深度（mm）。只影响渲染示意和说明文字。"""

    # ==================================================================
    # 3. 外形
    # ==================================================================
    length: float = 15.0
    """手柄总长（mm）—— 就是手指捏住的那一段，沿 X 轴。"""

    across_flats: float = 10.0
    """八边形对边距（mm），两个平行平面之间的距离。
    10 是食指+拇指自然捏合的量级（圆珠笔、小螺丝刀把都在这附近）；
    第一版做的 12 偏粗，手指张太开反而捏不稳。想更细改 9，更粗改 12。"""

    floor_thickness: float = 3.0
    """封闭端壁厚（mm），孔底到端面的距离。这是刀柄顶到底时的承力面。"""

    profile_rotation: float = 22.5
    """八边形绕轴旋转角（度）。22.5° 让一个平面正对热床、另一个正对天。"""

    # ==================================================================
    # 4. 材料 / 打印 / 报告用
    # ==================================================================
    material: str = "PETG"
    """耗材名，取 MATERIALS 里的键（PETG / PLA）。只影响说明文字和估重。"""

    layer_height: float = 0.2
    """层高（mm）。只用于算层数和打印说明，不参与建模。"""

    print_shrink_d: float = 0.00
    """预估的孔径收缩量（mm，直径上）。仅用于打印说明里的「预估打印孔径」，
    不做几何补偿。

    两个实测点给出的区间：
        Ø2.60 打出来偏松                          → 缩量 < 0.10
        Ø2.46 打出来配 Ø2.36 柄，手感「刚好贴合」  → 缩量 ≈ 0.00 ~ 0.06
    取 0.00（不假设收缩）。这个数只影响报告里的「预估」，不影响刀路 ——
    所以宁可填 0 也别填个猜的数去骗自己。
    PETG 收得少，外壁降速后水平小孔几乎不缩。换料换机再量一次即可。"""

    reamer_total_len: float = 57.0
    """铰刀全长（mm）。实测 57。

    只用于预览图里刀的示意比例和「装好后外伸多少」，不参与手柄建模。
    之前的 95 是拍的，画出来刀长得离谱 —— 刀杆比手柄长一倍还多，比例不对。"""

    # ==================================================================
    # 派生尺寸（不要手改）
    # ==================================================================
    @property
    def mat(self) -> MaterialProfile:
        """当前耗材的参数表。名字写错时退回 PETG，validate() 会另行报错。"""
        return MATERIALS.get(self.material.upper(), MATERIALS["PETG"])

    # ---- 孔系：让位孔 + 方孔 ------------------------------------------
    @property
    def square_on(self) -> bool:
        """是否走方孔传扭方案。"""
        return self.square_drive and self.square_size > 0.0

    @property
    def square_diag(self) -> float:
        """「尖角」对角 = √(a² + b²) = √(1.90² + 1.80²) = Ø2.62。**只是理论上限。**

        实测对角线是 2.36，差的这 0.26 全在四角倒圆上。
        这个概念留着是为了在报告里说明「别拿尖角算过孔」，不参与任何决策。"""
        return round(math.hypot(self.square_size, self.square_size_min), 3)

    @property
    def square_envelope_d(self) -> float:
        """方头的实际最大外廓直径（mm）= 实测对角线 Ø2.36。

        这是过孔校核唯一该用的数。由卡尺跨角直接量得，不含任何假设 ——
        之前那版用「两条边长 + 猜的倒圆半径」去推，推歪了两次。
        """
        return round(self.square_diag_meas, 3)

    @property
    def square_corner_r(self) -> float:
        """四角倒圆半径（mm），由实测对角线**反推**出来的，不是输入参数。

        解 2 × ( √((a/2 − r)² + (b/2 − r)²) + r ) = 对角线，二分法：
            a=1.90, b=1.80, 对角 2.36 → r ≈ 0.311
        只用于把方头画/建成带倒圆的实物形状（探针、渲染），
        不参与手柄本体几何（本体只有那个 2.00 方孔）。
        """
        a, b = self.square_size / 2.0, self.square_size_min / 2.0
        target = self.square_diag_meas
        if target >= 2.0 * math.hypot(a, b):      # 对角大于尖角对角 → 就是尖角
            return 0.0
        if target <= max(a, b):                   # 退化成外接圆
            return round(min(a, b), 3)

        def env(r: float) -> float:
            x, y = a - r, b - r
            return 2.0 * (math.hypot(x, y) + r)

        lo, hi = 0.0, min(a, b) - 1e-6
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if env(mid) > target:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2.0, 3)

    @property
    def square_hole(self) -> float:
        """方孔边长 = 方头边长 + 2 × 单边间隙。"""
        return round(self.square_size + 2.0 * self.square_clearance, 3)

    @property
    def pass_margin(self) -> float:
        """方头穿过圆孔时的单边余量（mm，按打印后估）。

        现在 = (2.46 − 2.36) / 2 = 0.050，和柄身的间隙一模一样 ——
        因为方头对角线就等于柄径，两者在孔里是同一个松紧。
        它只在方头比柄粗时才有意义（那时会是负值 = 过不去）。
        """
        return round((self.printed_bore_d - self.square_envelope_d) / 2.0, 3)

    @property
    def bore_d(self) -> float:
        """插孔直径。整条孔一个尺寸，从孔口直通到方孔入口。

        现在是一个纯柄身配合：
            孔径 = 柄径 + 2 × 单边间隙 = 2.36 + 2 × 0.05 = Ø2.46

        代码里仍保留「方头下限」那一项取 max —— 但方头对角实测只有 2.36，
        算出来 2.37，永远小于 2.46，不起作用。留着是防止哪天换了把粗方头的刀，
        孔径能自动跟着抬，而不是做出一个柄身能进、方头卡死的孔。
        """
        body_need = self.shank_d + 2.0 * self.bore_clearance
        if not self.square_on:
            return round(body_need, 2)
        tip_need = self.square_envelope_d + 2.0 * self.bore_pass_margin
        return round(max(body_need, tip_need), 2)

    @property
    def bore_r(self) -> float:
        return self.bore_d / 2.0

    @property
    def square_start(self) -> float:
        """方孔入口的 x 坐标（从插刀口端面量起）。"""
        if not self.square_on:
            return self.socket_len
        return round(self.socket_len - self.square_len, 3)

    @property
    def pass_len(self) -> float:
        """让位孔长度 = 方孔入口位置。"""
        return self.square_start if self.square_on else self.socket_len

    @property
    def square_top_z(self) -> float:
        """方孔尖顶高度。正方 + 45° 收顶，同样是免支撑的。"""
        return round(self.hole_z + self.square_hole, 3)

    @property
    def square_engage(self) -> float:
        """方头与方孔的实际咬合长度（mm）。

        方头占 [shank_end_x, shank_end_x + square_tip_len]，
        方孔占 [square_start, square_start + square_len]，取重叠段。
        所以「牙长」不一定等于方孔深度 —— 方头短就是方头说了算。
        """
        if not self.square_on:
            return 0.0
        lo = max(self.square_start, self.shank_end_x)
        hi = min(self.square_start + self.square_len,
                 self.shank_end_x + self.square_tip_len)
        return round(max(0.0, hi - lo), 3)

    @property
    def shank_end_x(self) -> float:
        """装到位后柄身末端（靠方头那一端）的 x 坐标。

        有方孔时，柄身末端就停在方孔入口：柄身 Ø2.46 比方孔 2.00 大，
        本来也进不去方孔段，让它停在入口是唯一合理的装配姿态。
        这样方头整根落在方孔里，咬合长度 = min(方孔深, 方头长)。
        """
        if self.square_on:
            return self.square_start
        return round(self.socket_len - self.socket_floor_gap, 3)

    @property
    def tip_floor_gap(self) -> float:
        """方头端面到孔底的轴向余量（mm）。

        有方孔时它恒为 方孔深 − 方头长：方头比方孔短才有余量，
        方头比如方孔长就会顶在方孔入口的台阶上。
        """
        if self.square_on:
            return round(self.square_len - self.square_tip_len, 3)
        return self.socket_floor_gap

    @property
    def shank_radial_gap(self) -> float:
        """柄身与插孔的单边间隙（mm）。按缩量修正后的实际值，不是名义值。"""
        return round((self.printed_bore_d - self.shank_d) / 2.0, 3)

    @property
    def printed_bore_d(self) -> float:
        """预估打印出来的孔径（扣掉收缩量）。"""
        return round(self.bore_d - self.print_shrink_d, 3)

    @property
    def printed_clearance(self) -> float:
        """预估装配时柄身的单边间隙。"""
        return round((self.printed_bore_d - self.shank_d) / 2.0, 3)

    @property
    def printed_square(self) -> float:
        """预估打印出来的方孔边长。"""
        return round(self.square_hole - self.print_shrink_d, 3)

    @property
    def square_play(self) -> float:
        """方孔与方头的单边旷量（mm，按打印后估）。这才是真正的手感。"""
        return round((self.printed_square - self.square_size) / 2.0, 3)

    @property
    def square_torque(self) -> tuple[float, float]:
        """方孔能传的扭矩区间（N·m）。

        按四个侧面挤压算：T = σ · A · r，A 是两个受力面之和，r = 边长/2。
        PETG 的许用挤压应力取 15~40 MPa（长期/短期），已是保守值。
        """
        if not self.square_on:
            return (0.0, 0.0)
        arm = self.square_size / 2.0                          # 力臂 mm
        area = 2.0 * self.square_size * self.square_engage    # 两个受力面 mm²
        lo = 15.0 * area * arm / 1000.0
        hi = 40.0 * area * arm / 1000.0
        return (round(lo, 3), round(hi, 3))

    @property
    def fit_word(self) -> str:
        """按预估旷量给一句手感描述。"""
        if self.square_on:
            g = self.square_play
            if g < -0.02:
                return "方孔偏小，方头压不进"
            if g < 0.03:
                return "方孔紧配，几乎无旷量"
            if g < 0.08:
                return "方孔轻推入，旷量可接受"
            return "方孔偏松，起转会有空程"
        c = self.printed_clearance
        if c < -0.03:
            return "过盈，需明显加压或热装"
        if c < 0.0:
            return "微过盈，压得进但很紧"
        if c < 0.05:
            return "紧推入（推荐区间）"
        if c < 0.08:
            return "轻推入，略有手感"
        return "偏松，建议点胶"

    @property
    def mass_g(self) -> float:
        return round(self.volume_ref_mm3 / 1000.0 * self.mat.density, 2)

    @property
    def volume_ref_mm3(self) -> float:
        """解析参考体积（忽略倒角），用于估重和出图前预判。"""
        a_big = 2.0 * (math.sqrt(2.0) - 1.0) * self.across_flats**2
        if self.neck_on:
            a_small = 2.0 * (math.sqrt(2.0) - 1.0) * self.mouth_neck_flats**2
            neck_cut = (a_big - a_small) * (self.mouth_neck_len + self.neck_taper / 2.0)
        else:
            neck_cut = 0.0
        if self.square_on:
            # 让位孔 = 圆柱；方孔 = 正方形 + 45° 三角尖顶
            h = self.square_hole / 2.0
            a_pent = self.square_hole**2 + h**2
            hole = math.pi * self.bore_r**2 * self.pass_len + a_pent * self.square_len
        else:
            hole = math.pi * self.bore_r**2 * self.socket_len
        return a_big * self.length - neck_cut - hole

    @property
    def socket_len(self) -> float:
        """插孔深度（从插刀口端面量起）。"""
        return round(self.length - self.floor_thickness, 3)

    @property
    def insert_depth(self) -> float:
        """柄部实际插进去的长度 = 柄身末端位置。"""
        return self.shank_end_x

    @property
    def exposed_shank(self) -> float:
        """柄部露在孔外的长度。方孔方案下柄身只进到方孔入口，所以要按
        shank_end_x 算，而不是按整个插入深度算 —— 否则会少算一个方头长度。"""
        if self.square_on:
            return round(max(0.0, self.shank_len - self.shank_end_x), 2)
        return round(max(0.0, self.shank_len - self.insert_depth), 2)

    @property
    def circumradius(self) -> float:
        """八边形外接圆半径。"""
        return self.across_flats / (2.0 * math.cos(math.pi / 8.0))

    @property
    def side_len(self) -> float:
        """八边形边长，也就是贴床那条的宽度。"""
        return 2.0 * (self.across_flats / 2.0) * math.tan(math.pi / 8.0)

    @property
    def hole_z(self) -> float:
        """孔轴线高度 = 八边形中心高度。"""
        return self.across_flats / 2.0

    @property
    def neck_on(self) -> bool:
        """是否做孔口收口。"""
        return self.mouth_neck_flats > 0.0

    @property
    def neck_r(self) -> float:
        """收口处八边形的外接圆半径。"""
        return self.mouth_neck_flats / (2.0 * math.cos(math.pi / 8.0))

    @property
    def neck_taper(self) -> float:
        """45° 锥面过渡段的轴向长度。等于单边收进量，保证锥面正好 45°。"""
        if not self.neck_on:
            return 0.0
        return round(max(0.0, (self.across_flats - self.mouth_neck_flats) / 2.0), 3)

    @property
    def neck_end(self) -> float:
        """收口特征结束处的 x 坐标。"""
        if not self.neck_on:
            return 0.0
        return round(self.mouth_neck_len + self.neck_taper, 3)

    @property
    def neck_side_len(self) -> float:
        """收口段的边长，也就是它贴床那条的宽度。"""
        return round(2.0 * (self.mouth_neck_flats / 2.0) * math.tan(math.pi / 8.0), 2)

    @property
    def neck_wall(self) -> float:
        """收口段的孔壁厚度（mm）。"""
        return round(self.mouth_neck_flats / 2.0 - self.bore_r, 3)

    @property
    def neck_sections(self) -> list[tuple[float, float]]:
        """loft 用的 (x 位置, 八边形外接圆半径) 截面序列。"""
        if not self.neck_on:
            return [(0.0, self.circumradius), (self.length, self.circumradius)]
        return [
            (0.0, self.neck_r),
            (self.mouth_neck_len, self.neck_r),
            (self.neck_end, self.circumradius),
            (self.length, self.circumradius),
        ]


    @property
    def screw_thread_depth(self) -> float:
        """顶丝孔的可用螺纹深度 = 顶面到孔顶的距离（mm）。M3 至少要 2.5。"""
        return round(self.across_flats - self.bore_top_z, 3)

    @property
    def bond_area_mm2(self) -> float:
        """刀柄插入段的圆柱侧面积，也就是胶接面积（mm²）。"""
        return round(math.pi * self.shank_d * self.insert_depth, 1)

    @property
    def wall(self) -> float:
        """孔壁到平面的最小壁厚（mm）。"""
        return round(self.across_flats / 2.0 - self.bore_r, 3)

    @property
    def bore_top_z(self) -> float:
        """整条孔系的最高点。泪滴让位段和方孔尖顶取高的那个。"""
        if self.teardrop:
            pass_top = self.hole_z + self.bore_r * math.sqrt(2.0)
        else:
            pass_top = self.hole_z + self.bore_r
        if self.square_on:
            return round(max(pass_top, self.square_top_z), 3)
        return round(pass_top, 3)

    @property
    def layers(self) -> int:
        """总层数。"""
        return int(round(self.across_flats / self.layer_height))

    @property
    def reamer_protrusion(self) -> float:
        """插到底后刀杆水平伸出孔口的长度（mm）。"""
        return round(self.reamer_total_len - self.insert_depth, 1)

    def with_overrides(self, **kwargs) -> "HandleParams":
        return replace(self, **kwargs)

    def summary(self) -> str:
        sq = (
            f"  尾部方头       {self.square_size:.2f} × {self.square_size_min:.2f}"
            f" × {self.square_tip_len:.2f} 长（对角 Ø{self.square_envelope_d:.2f}"
            f"，四角倒圆 ≈R{self.square_corner_r:.2f}）"
            if self.square_on
            else "  尾部方头       未使用（纯圆孔方案）"
        )
        sq_hole = (
            f"  孔底方孔       {self.square_hole:.2f} 见方 × {self.square_len:.2f} 深"
            f"，入口 x = {self.square_start:.2f}，尖顶 z = {self.square_top_z:.2f}"
            if self.square_on
            else ""
        )
        sq_fit = (
            f"  预估方孔       {self.printed_square:.2f} 见方"
            f" → 单边旷量 {self.square_play:.3f}（{self.fit_word}）"
            f"，咬合 {self.square_engage:.2f}"
            if self.square_on
            else ""
        )
        lines = [
            "=" * 64,
            " 铰刀手柄参数摘要（平放打印姿态）",
            "=" * 64,
            f"  铰刀柄径       Ø{self.shank_d:.2f} × {self.shank_len:.0f} 长",
            sq,
            f"  插孔           Ø{self.bore_d:.2f} × {self.pass_len:.2f} 深"
            + (f"（= 柄径 + 2×{self.bore_clearance:.2f}；方头对角 "
               f"{self.square_envelope_d:.2f} 不比它粗，不起约束）"
               if self.square_on
               else f"（名义单边 {self.bore_clearance:+.2f}）"),
            sq_hole,
            f"  预估打印孔径   Ø{self.printed_bore_d:.2f}"
            f"（缩量 {self.print_shrink_d:.2f}）"
            f" → 柄身单边 {self.printed_clearance:+.3f}"
            + ("  [方头过孔余量 "
               f"{(self.printed_bore_d - self.square_envelope_d) / 2:.3f}]"
               if self.square_on else ""),
            sq_fit,
            f"  孔口倒角       {self.mouth_chamfer:.2f} × 45°"
            f"   孔系最高 z = {self.bore_top_z:.2f}"
            f"（{'泪滴' if self.teardrop else '圆孔'}）",
            f"  外形           长 {self.length:.2f} × 对边距 {self.across_flats:.2f}"
            f"（边长 {self.side_len:.2f}，最小壁厚 {self.wall:.2f}）",
            (
                f"  孔口收口       Ø{self.mouth_neck_flats:.2f} A/F × {self.mouth_neck_len:.2f}"
                f" + 45° 锥 {self.neck_taper:.2f}（壁厚 {self.neck_wall:.2f}）"
                if self.neck_on
                else "  孔口收口       关闭（等截面）"
            ),
            (
                f"  顶丝孔         M3 × {self.screw_thread_depth:.2f} 深"
                f"，x = {self.set_screw_x:.2f}（Ø{self.set_screw_d:.2f}）"
                if self.set_screw_hole
                else "  顶丝孔         关闭（有方孔，不需要）"
            ),
            f"  耗材           {self.mat.name}（{self.mat.density} g/cm³）"
            f"，估重 {self.mass_g:.2f} g",
            f"  打印姿态       平放，底面 {self.length:.2f} × {self.side_len:.2f} 贴床",
            f"  打印高度       只到 z = {self.across_flats:.2f}，共 {self.layers} 层",
            f"  插入深度       {self.insert_depth:.2f}，柄部外露 {self.exposed_shank:.2f}",
            "=" * 64,
        ]
        return "\n".join(lines)


DEFAULT = HandleParams()


# ======================================================================
# 档案：一把铰刀一份 JSON（只写实测那几项）
# ======================================================================
PROFILE_DIR = Path(__file__).resolve().parent.parent / "reamers"
DEFAULT_PROFILE = "r01-2.36"

_PROFILE_META = {"name", "note", "comment"}   # 说明性字段，不进参数


def profile_path(name_or_path: str) -> Path:
    """名字 → 路径。`r02` 等价于 `reamers/r02.json`；也可以直接给 .json 路径。"""
    p = Path(name_or_path)
    if p.suffix.lower() == ".json" or p.parent != Path("."):
        return p if p.is_absolute() else (Path.cwd() / p)
    return PROFILE_DIR / f"{p.name}.json"


def list_profiles() -> list[str]:
    """reamers/ 下所有档案名（不带 .json），按名字排序。"""
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(f.stem for f in PROFILE_DIR.glob("*.json"))


def load_profile(name_or_path: str = DEFAULT_PROFILE) -> tuple[HandleParams, str]:
    """读一份档案，返回 (参数, 档案名)。档案名同时用作输出子目录名。

    JSON 里只需写跟这把刀有关的项，没写的就用手柄默认值 ——
    所以换刀只写 shank_d / square_* / reamer_total_len 这几个数，外形和打印参数不用抄。
    """
    path = profile_path(name_or_path)
    if not path.is_file():
        avail = "、".join(list_profiles()) or "（reamers/ 下还没有档案）"
        raise SystemExit(f"找不到档案 {path}\n现有：{avail}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"档案 {path.name} 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"档案 {path.name} 顶层必须是对象 {{...}}")

    overrides = {k: v for k, v in data.items() if k not in _PROFILE_META}
    known = {f.name for f in fields(HandleParams)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise SystemExit(
            f"档案 {path.name} 里有不认识的键：{', '.join(unknown)}\n"
            f"可用：{', '.join(sorted(known))}"
        )
    return replace(DEFAULT, **overrides), path.stem


def profile_report(p: HandleParams, slug: str) -> str:
    """把「档案里写的实测值」和「由它算出来的尺寸」并排列出来，方便核对。

    档案里只有实测值，Ø2.46 这类都是算的 —— 这个函数就是让「算出来的东西」有地方看。
    """
    two = 2.0 * p.bore_clearance
    lines = [
        "=" * 64,
        f" 档案 {slug}：实测值 → 派生尺寸",
        "=" * 64,
        "  实测（写进 JSON 的，只这几项）",
        f"    柄身直径      Ø{p.shank_d:.2f}      圆柱段，孔径就看它",
        f"    方头边长      {p.square_size:.2f} × {p.square_size_min:.2f}",
        f"    方头长度      {p.square_tip_len:.2f}",
        f"    方头对角      Ø{p.square_diag_meas:.2f}      卡尺跨角量，过孔看它",
        f"    铰刀全长      {p.reamer_total_len:.1f}",
        "",
        "  派生（不用写，代码算）",
        f"    插孔直径      Ø{p.bore_d:.2f}   = 柄径 {p.shank_d:.2f} + 2 × 单边 {p.bore_clearance:.2f}",
        f"    方孔边长      {p.square_hole:.2f}   = 边长 {p.square_size:.2f} + 2 × 单边 {p.square_clearance:.2f}",
        f"    圆孔 / 方孔深 {p.pass_len:.2f} / {p.square_len:.2f}",
        f"    咬合长度      {p.square_engage:.2f}",
        f"    柄身单边间隙  {p.shank_radial_gap:.3f}",
        f"    方头过孔余量  {(p.bore_d - p.square_envelope_d) / 2:.3f}",
        "=" * 64,
        "",
        "  两个 2.36 别搞混：柄身是圆柱直径，方头对角是跨角量的包络圆直径。",
        "  方孔只按边长算（自带 45°，对角不用留），所以孔径只受柄身约束。",
    ]
    if abs(p.square_envelope_d - p.shank_d) > 1e-9:
        lines += ["",
                  f"  注意：这把刀的方头对角 Ø{p.square_envelope_d:.2f} ≠ 柄径 Ø{p.shank_d:.2f}，",
                  "        过孔由两者中粗的那个决定（见 bore_d 的 max 分支）。"]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="查看参数 / 档案")
    ap.add_argument("--reamer", metavar="NAME", help="看某份档案：实测值 → 派生尺寸")
    ns = ap.parse_args()

    if ns.reamer:
        profile, name = load_profile(ns.reamer)
        print(profile_report(profile, name))
    else:
        print(DEFAULT.summary())
        if list_profiles():
            print("\n可用档案：" + "、".join(list_profiles()))
            print("看某份的实测→派生：python codes/params.py --reamer <名字>")
