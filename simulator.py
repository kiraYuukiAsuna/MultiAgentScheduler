"""
多针细胞染料灌注调度 - 离散事件仿真器
基于 MathModel8.tex v8.0 Section 6-8
"""

from enum import IntEnum
from dataclasses import dataclass, field
import numpy as np
from copy import copy

from config import Cell, Needle, SimConfig


class Phase(IntEnum):
    IDLE = 0
    MOVING = 1
    FINDING = 2
    WAITING = 3
    RETRACTING = 4
    BLOCKED = 5
    FINISHED = 6


@dataclass
class NeedleState:
    """单根针的仿真状态"""
    actual_pos: np.ndarray          # 实际针尖2D坐标
    phase: Phase = Phase.IDLE
    cell_ptr: int = 0               # 已完成的细胞数
    next_event_time: float = np.inf
    total_blocked: float = 0.0
    block_start: float = 0.0
    total_dist: float = 0.0
    retract_count: int = 0


@dataclass
class SimEvent:
    """仿真事件记录 (用于可视化)"""
    needle_id: int
    phase: Phase
    start_time: float
    end_time: float
    pos_start: np.ndarray = None    # 移动起点
    pos_end: np.ndarray = None      # 移动终点 / 停留位置
    cell_id: int = -1               # 关联的细胞ID (-1表示无)
    retract_delta: float = 0.0      # 退让距离


@dataclass
class SimResult:
    """仿真结果"""
    f1_makespan: float = 0.0
    f2_total_dist: float = 0.0
    f3_retract_count: int = 0
    f4_blocked_time: float = 0.0
    deadlock: bool = False
    events: list = field(default_factory=list)  # SimEvent列表
    # 每根针的完成时间
    finish_times: list = field(default_factory=list)


def _compute_effective_body_check_length(cells: list[Cell], cfg: SimConfig) -> float:
    """
    自动计算碰撞检测的有效针身长度。
    当用户设置了过大的 body_check_length (如200000 代表无限长)时，
    自动裁剪到工作区域对角线长度，避免远距离虚假交叉。
    """
    if len(cells) == 0:
        return cfg.body_check_length

    all_x = [c.x for c in cells]
    all_y = [c.y for c in cells]
    workspace_diag = np.sqrt((max(all_x) - min(all_x))**2 +
                             (max(all_y) - min(all_y))**2)

    # 如果用户设置的值在合理范围内 (≤ 工作区对角线×3)，直接使用
    reasonable_limit = max(workspace_diag * 3, 3000.0)
    if cfg.body_check_length <= reasonable_limit:
        return cfg.body_check_length

    # 否则裁剪到工作区对角线 (保证近场碰撞检测有效)
    return max(workspace_diag, 1000.0)


def simulate(sequences: list[list[int]],
             retractions: list[int],
             cells: list[Cell],
             needles: list[Needle],
             cfg: SimConfig,
             record_events: bool = False) -> SimResult:
    """
    离散事件仿真器 (Algorithm 3: EvaluateIndividual)

    参数:
        sequences: M个排列，每个是该针要处理的细胞ID列表（按访问顺序）
        retractions: N个退让意向 (0或1)，按cell.id索引
        cells: 所有细胞
        needles: 所有针
        cfg: 仿真配置
        record_events: 是否记录详细事件（用于可视化，评估时关闭以加速）

    返回:
        SimResult
    """
    from geometry2d import is_safe, compute_min_retract, conflict, compute_retract_to_clear

    M = len(needles)
    cell_map = {c.id: c for c in cells}
    result = SimResult()

    # 自动计算有效碰撞检测长度
    effective_bcl = _compute_effective_body_check_length(cells, cfg)
    sim_cfg = copy(cfg)
    sim_cfg.body_check_length = effective_bcl

    # --- 阶段1: 初始化 ---
    tau = 0.0
    camera_free = True

    states: list[NeedleState] = []
    for m in range(M):
        states.append(NeedleState(
            actual_pos=needles[m].init_pos.copy(),
        ))

    events_log: list[SimEvent] = []
    phase_start_time = [0.0] * M
    phase_start_pos = [needles[m].init_pos.copy() for m in range(M)]
    pending_retract_delta = [0.0] * M

    def get_all_positions():
        return [s.actual_pos for s in states]

    def record(needle_id, phase, start_t, end_t, pos_start=None, pos_end=None,
               cell_id=-1, retract_delta=0.0):
        if record_events:
            events_log.append(SimEvent(
                needle_id=needle_id, phase=phase,
                start_time=start_t, end_time=end_t,
                pos_start=pos_start.copy() if pos_start is not None else None,
                pos_end=pos_end.copy() if pos_end is not None else None,
                cell_id=cell_id, retract_delta=retract_delta
            ))

    def try_move_to_cell(m_idx: int):
        """尝试让针m前往下一个细胞 (不触发协作退让)"""
        nonlocal camera_free
        st = states[m_idx]
        needle = needles[m_idx]
        seq = sequences[m_idx]

        if st.cell_ptr >= len(seq):
            st.phase = Phase.FINISHED
            return

        next_cell_id = seq[st.cell_ptr]
        c = cell_map[next_cell_id]
        p_dest = np.array([c.x, c.y])

        # 准入条件: 相机空闲 + 目标安全
        if not camera_free:
            if st.phase != Phase.BLOCKED:
                if st.phase == Phase.IDLE:
                    record(m_idx, Phase.IDLE, phase_start_time[m_idx], tau,
                           pos_end=st.actual_pos.copy())
                st.phase = Phase.BLOCKED
                st.block_start = tau
                phase_start_time[m_idx] = tau
            return

        if not is_safe(needle, p_dest, needles, get_all_positions(), sim_cfg):
            if st.phase != Phase.BLOCKED:
                if st.phase == Phase.IDLE:
                    record(m_idx, Phase.IDLE, phase_start_time[m_idx], tau,
                           pos_end=st.actual_pos.copy())
                st.phase = Phase.BLOCKED
                st.block_start = tau
                phase_start_time[m_idx] = tau
            return

        # 准入通过
        if st.phase == Phase.BLOCKED:
            st.total_blocked += tau - st.block_start
            record(m_idx, Phase.BLOCKED, phase_start_time[m_idx], tau,
                   pos_end=st.actual_pos.copy())
        elif st.phase == Phase.IDLE:
            record(m_idx, Phase.IDLE, phase_start_time[m_idx], tau,
                   pos_end=st.actual_pos.copy())

        p_old = st.actual_pos.copy()
        move_dist = np.linalg.norm(p_dest - p_old)
        move_time = move_dist / cfg.v if cfg.v > 0 else 0.0

        # 保守锁定 + 锁定相机
        st.actual_pos = p_dest.copy()
        camera_free = False
        st.phase = Phase.MOVING
        st.next_event_time = tau + move_time
        st.total_dist += move_dist

        phase_start_time[m_idx] = tau
        phase_start_pos[m_idx] = p_old.copy()

    def start_retraction(q_idx: int, delta: float):
        """让针q执行退让 (从当前位置后退delta距离)"""
        st_q = states[q_idx]
        p_old = st_q.actual_pos.copy()
        p_ret = p_old + delta * needles[q_idx].retract_dir
        move_dist = np.linalg.norm(p_ret - p_old)
        move_time = move_dist / cfg.v if cfg.v > 0 else 0.0

        # 记录之前的状态
        if st_q.phase == Phase.IDLE:
            record(q_idx, Phase.IDLE, phase_start_time[q_idx], tau,
                   pos_end=st_q.actual_pos.copy())
        elif st_q.phase == Phase.BLOCKED:
            st_q.total_blocked += tau - st_q.block_start
            record(q_idx, Phase.BLOCKED, phase_start_time[q_idx], tau,
                   pos_end=st_q.actual_pos.copy())

        st_q.actual_pos = p_ret.copy()
        st_q.phase = Phase.RETRACTING
        st_q.next_event_time = tau + move_time
        st_q.total_dist += move_dist
        st_q.retract_count += 1
        pending_retract_delta[q_idx] = delta

        phase_start_time[q_idx] = tau
        phase_start_pos[q_idx] = p_old.copy()

    def do_post_wait_dispatch(m_idx: int):
        """
        Wait完成后的退让分支逻辑 (Algorithm 2: PostWaitDispatch)
        注意: 不再直接调用 try_move_to_cell，交由 section 2.3 统一调度
        """
        st = states[m_idx]
        needle = needles[m_idx]
        seq = sequences[m_idx]

        if st.cell_ptr >= len(seq):
            p_old = st.actual_pos.copy()
            st.actual_pos = needles[m_idx].init_pos.copy()
            st.total_dist += np.linalg.norm(st.actual_pos - p_old)
            st.phase = Phase.FINISHED
            record(m_idx, Phase.FINISHED, tau, tau,
                   pos_start=p_old, pos_end=st.actual_pos.copy())
            return

        current_cell_id = seq[st.cell_ptr - 1]  # 刚完成的细胞
        beta = retractions[current_cell_id]

        if beta == 0:
            # 不退让，设为IDLE等待section 2.3调度
            st.phase = Phase.IDLE
            phase_start_time[m_idx] = tau
        else:
            # 主动退让: 检查当前位置是否与他针冲突
            p_current = st.actual_pos.copy()
            delta_star = compute_min_retract(
                needle, p_current, needles, get_all_positions(), sim_cfg)

            if delta_star <= sim_cfg.delta_step:
                # 无实际冲突，设为IDLE等待调度
                st.phase = Phase.IDLE
                phase_start_time[m_idx] = tau
            else:
                # 执行主动退让
                start_retraction(m_idx, delta_star)

    def try_cooperative_retraction():
        """
        协作退让: 当针A被阻塞(BLOCKED)且相机空闲时，
        找到阻塞A的非活跃针(IDLE/BLOCKED)B，让B退让以解除冲突。
        可以连续解决多个冲突 (一次调度中退让多根针)。
        返回True如果成功安排了至少一次退让。
        """
        if not camera_free:
            return False

        any_retracted = False
        # 多轮尝试: 一根针退让后可能解锁另一根针的路径
        for _round in range(M):
            made_progress = False
            for m in range(M):
                st_m = states[m]
                if st_m.phase != Phase.BLOCKED:
                    continue
                if st_m.cell_ptr >= len(sequences[m]):
                    continue

                next_cell_id = sequences[m][st_m.cell_ptr]
                c = cell_map[next_cell_id]
                p_dest = np.array([c.x, c.y])

                # 找所有阻挡 m 目标的非活跃针
                for q in range(M):
                    if q == m:
                        continue
                    st_q = states[q]
                    # 可以退让的状态: IDLE 或 BLOCKED (不在主动工作中)
                    if st_q.phase not in (Phase.IDLE, Phase.BLOCKED):
                        continue

                    if not conflict(needles[m], p_dest, needles[q],
                                    st_q.actual_pos, sim_cfg):
                        continue

                    # q 阻挡了 m，计算退让距离
                    delta = compute_retract_to_clear(
                        needles[q], st_q.actual_pos, needles[m], p_dest, sim_cfg)

                    if delta <= sim_cfg.delta_step:
                        continue
                    if delta > sim_cfg.delta_max:
                        continue  # 超出退让上限，跳过

                    start_retraction(q, delta)
                    any_retracted = True
                    made_progress = True
                    break  # q 已退让，重新检查 m 的其他冲突

            if not made_progress:
                break

        return any_retracted

    # --- 阶段2: 事件驱动仿真主循环 ---
    event_count = 0

    # 初始调度: 所有针尝试前往第一个细胞
    for m in range(M):
        if len(sequences[m]) > 0:
            try_move_to_cell(m)
        else:
            states[m].phase = Phase.FINISHED

    while True:
        event_count += 1
        if event_count > cfg.max_events:
            result.deadlock = True
            break

        # 检查是否所有针都已完成
        if all(s.phase == Phase.FINISHED for s in states):
            break

        # 2.1 找最近的事件时间
        active_phases = {Phase.MOVING, Phase.FINDING, Phase.WAITING, Phase.RETRACTING}
        tau_next = np.inf
        for s in states:
            if s.phase in active_phases and s.next_event_time < tau_next:
                tau_next = s.next_event_time

        if tau_next == np.inf:
            # 没有活跃事件 — 所有未完成的针都在IDLE或BLOCKED
            unfinished = [s for s in states if s.phase != Phase.FINISHED]
            if not unfinished:
                break

            # 尝试常规调度
            old_phases = [s.phase for s in states]
            for m in range(M):
                if states[m].phase in (Phase.IDLE, Phase.BLOCKED):
                    try_move_to_cell(m)
            new_phases = [s.phase for s in states]
            if old_phases != new_phases:
                continue

            # 尝试协作退让
            if camera_free and try_cooperative_retraction():
                continue

            # 真正的死锁
            result.deadlock = True
            break

        tau = tau_next

        # 2.2 处理到期事件
        for m in range(M):
            st = states[m]
            if st.phase not in active_phases:
                continue
            if abs(st.next_event_time - tau) > 1e-9:
                continue

            if st.phase == Phase.MOVING:
                cell_id = sequences[m][st.cell_ptr]
                record(m, Phase.MOVING, phase_start_time[m], tau,
                       pos_start=phase_start_pos[m], pos_end=st.actual_pos.copy(),
                       cell_id=cell_id)
                st.phase = Phase.FINDING
                st.next_event_time = tau + cfg.t_find
                phase_start_time[m] = tau

            elif st.phase == Phase.FINDING:
                record(m, Phase.FINDING, phase_start_time[m], tau,
                       pos_end=st.actual_pos.copy(),
                       cell_id=sequences[m][st.cell_ptr])
                camera_free = True
                st.phase = Phase.WAITING
                st.next_event_time = tau + cfg.t_wait
                phase_start_time[m] = tau

            elif st.phase == Phase.WAITING:
                cell_id = sequences[m][st.cell_ptr]
                record(m, Phase.WAITING, phase_start_time[m], tau,
                       pos_end=st.actual_pos.copy(), cell_id=cell_id)
                st.cell_ptr += 1
                st.next_event_time = np.inf

                if st.cell_ptr >= len(sequences[m]):
                    p_old = st.actual_pos.copy()
                    st.actual_pos = needles[m].init_pos.copy()
                    st.total_dist += np.linalg.norm(st.actual_pos - p_old)
                    st.phase = Phase.FINISHED
                    record(m, Phase.FINISHED, tau, tau,
                           pos_start=p_old, pos_end=st.actual_pos.copy())
                else:
                    do_post_wait_dispatch(m)

            elif st.phase == Phase.RETRACTING:
                record(m, Phase.RETRACTING, phase_start_time[m], tau,
                       pos_start=phase_start_pos[m], pos_end=st.actual_pos.copy(),
                       retract_delta=pending_retract_delta[m])
                st.phase = Phase.IDLE
                st.next_event_time = np.inf
                phase_start_time[m] = tau
                # 不在这里调用 try_move_to_cell，交由 2.3 统一调度

        # 2.3 协作退让 + 统一调度
        # 先检查: 是否有IDLE针阻挡了BLOCKED针，如果有则先退让
        if camera_free:
            try_cooperative_retraction()

        # 再尝试调度所有IDLE和BLOCKED的针
        for m in range(M):
            if states[m].phase in (Phase.IDLE, Phase.BLOCKED):
                try_move_to_cell(m)

    # --- 阶段3: 汇总 ---
    finish_times = []
    for m in range(M):
        st = states[m]
        if st.phase == Phase.BLOCKED:
            st.total_blocked += tau - st.block_start
        finish_times.append(tau)

    if result.deadlock:
        result.f1_makespan = cfg.h_penalty
        result.f2_total_dist = sum(s.total_dist for s in states)
        result.f3_retract_count = sum(s.retract_count for s in states)
        result.f4_blocked_time = cfg.h_penalty
    else:
        result.f1_makespan = tau
        result.f2_total_dist = sum(s.total_dist for s in states)
        result.f3_retract_count = sum(s.retract_count for s in states)
        result.f4_blocked_time = sum(s.total_blocked for s in states)

    result.events = events_log
    result.finish_times = finish_times
    return result
