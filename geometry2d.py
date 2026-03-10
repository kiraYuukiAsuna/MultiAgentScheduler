"""
多针细胞染料灌注调度 - 2D碰撞检测
基于 MathModel8.tex v8.0 Section 4 (2D简化)
"""

import numpy as np
from config import Needle, SimConfig


def point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    点P到线段AB的最短距离 (2D)
    d_pt-seg(P, A, B) = min_{t in [0,1]} ||P - (A + t(B-A))||
    """
    ab = b - a
    ap = p - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:
        return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def segment_to_segment_dist(a1: np.ndarray, b1: np.ndarray,
                            a2: np.ndarray, b2: np.ndarray) -> float:
    """
    线段A1B1到线段A2B2的最短距离 (2D)
    先检查是否相交，再计算四个端点到对方线段的距离取最小值
    """
    d1 = b1 - a1
    d2 = b2 - a2

    # 检查线段是否相交
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) > 1e-12:
        diff = a2 - a1
        t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
        u = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0  # 线段相交

    # 四个端点到对方线段的距离
    dists = [
        point_to_segment_dist(a1, a2, b2),
        point_to_segment_dist(b1, a2, b2),
        point_to_segment_dist(a2, a1, b1),
        point_to_segment_dist(b2, a1, b1),
    ]
    return min(dists)


def conflict(needle_m: Needle, pos_m: np.ndarray,
             needle_q: Needle, pos_q: np.ndarray,
             cfg: SimConfig) -> bool:
    """
    实时几何冲突判定: Conflict(m, p, q, p')
    三重准则: 针尖-针尖, 针尖-针身, 针身-针身
    使用 cfg.body_check_length 作为碰撞检测的有效针身长度
    返回True表示有冲突
    """
    tm = needle_m.tip_pos(pos_m)
    bm = pos_m + cfg.body_check_length * needle_m.direction
    tq = needle_q.tip_pos(pos_q)
    bq = pos_q + cfg.body_check_length * needle_q.direction

    # 准则1: 针尖-针尖
    if np.linalg.norm(tm - tq) < cfg.d_tip:
        return True

    # 准则2: 针尖-针身 (双向)
    if point_to_segment_dist(tm, tq, bq) < cfg.d_tip:
        return True
    if point_to_segment_dist(tq, tm, bm) < cfg.d_tip:
        return True

    # 准则3: 针身-针身
    if segment_to_segment_dist(tm, bm, tq, bq) < cfg.d_body:
        return True

    return False


def is_safe(needle_m: Needle, pos_m: np.ndarray,
            all_needles: list[Needle], all_positions: list[np.ndarray],
            cfg: SimConfig) -> bool:
    """
    全局安全性判定: IsSafe(m, p, tau)
    检查needle_m在pos_m处与所有其他针的当前位置是否无冲突
    """
    m_id = needle_m.id
    for needle_q, pos_q in zip(all_needles, all_positions):
        if needle_q.id == m_id:
            continue
        if conflict(needle_m, pos_m, needle_q, pos_q, cfg):
            return False
    return True


def compute_min_retract(needle_m: Needle, pos_m: np.ndarray,
                        all_needles: list[Needle], all_positions: list[np.ndarray],
                        cfg: SimConfig) -> float:
    """
    最小安全退让距离计算 (Algorithm 1: ComputeMinRetract)
    使用二分搜索加速
    返回最小退让距离 delta*
    """
    # 当前已安全，无需退让
    if is_safe(needle_m, pos_m, all_needles, all_positions, cfg):
        return 0.0

    ret_dir = needle_m.retract_dir

    # 先检查最大退让距离是否能解决冲突
    pos_max = pos_m + cfg.delta_max * ret_dir
    if not is_safe(needle_m, pos_max, all_needles, all_positions, cfg):
        return cfg.delta_max  # 兜底

    # 二分搜索: 在[0, delta_max]上找冲突从True变False的临界点
    lo, hi = 0.0, cfg.delta_max
    while hi - lo > cfg.delta_step:
        mid = (lo + hi) / 2.0
        pos_mid = pos_m + mid * ret_dir
        if is_safe(needle_m, pos_mid, all_needles, all_positions, cfg):
            hi = mid
        else:
            lo = mid

    # 返回hi (保证安全)
    return hi


def compute_retract_to_clear(needle_q: Needle, pos_q: np.ndarray,
                              needle_m: Needle, dest_m: np.ndarray,
                              cfg: SimConfig) -> float:
    """
    计算针q需要退让多少距离，使得针m在dest_m处不与q冲突。
    用于协作退让: q为空闲针，m为被阻塞针。
    返回退让距离 (0表示不冲突, delta_max+1表示无法解决)
    """
    if not conflict(needle_m, dest_m, needle_q, pos_q, cfg):
        return 0.0

    ret_dir = needle_q.retract_dir

    # 检查最大退让能否解决
    pos_max = pos_q + cfg.delta_max * ret_dir
    if conflict(needle_m, dest_m, needle_q, pos_max, cfg):
        return cfg.delta_max + 1.0  # 无法解决

    # 二分搜索
    lo, hi = 0.0, cfg.delta_max
    while hi - lo > cfg.delta_step:
        mid = (lo + hi) / 2.0
        pos_mid = pos_q + mid * ret_dir
        if not conflict(needle_m, dest_m, needle_q, pos_mid, cfg):
            hi = mid
        else:
            lo = mid

    return hi
