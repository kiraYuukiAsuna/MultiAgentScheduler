"""
多针细胞染料灌注调度 - 交互式时间轴查看器
用滑动条浏览每个时刻的针/细胞位置、针身线段、碰撞检测结果
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from config import Cell, Needle, SimConfig
from simulator import SimResult, Phase, SimEvent
from geometry2d import segment_to_segment_dist, point_to_segment_dist


# ------------------------------------------------------------------
# 中文字体
# ------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

NEEDLE_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
                 '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']

PHASE_NAMES = {
    Phase.IDLE:       'IDLE',
    Phase.MOVING:     'MOVING',
    Phase.FINDING:    'FINDING',
    Phase.WAITING:    'WAITING',
    Phase.RETRACTING: 'RETRACTING',
    Phase.BLOCKED:    'BLOCKED',
    Phase.FINISHED:   'FINISHED',
}

PHASE_COLORS_GANTT = {
    Phase.IDLE:       '#CCCCCC',
    Phase.MOVING:     '#4A90D9',
    Phase.FINDING:    '#E8553A',
    Phase.WAITING:    '#F5A623',
    Phase.RETRACTING: '#9B59B6',
    Phase.BLOCKED:    '#2C3E50',
    Phase.FINISHED:   '#27AE60',
}


# ==================================================================
# 从事件日志重建任意时刻的针状态
# ==================================================================

def build_timeline(result: SimResult, needles: list[Needle]):
    """
    将事件列表转换为每根针的分段时间轴,
    每段记录 (t_start, t_end, phase, pos_start, pos_end, cell_id).
    移动/退让阶段: 针尖在 pos_start -> pos_end 之间线性插值
    其余阶段: 针尖固定在 pos_end
    """
    M = len(needles)
    timelines = [[] for _ in range(M)]  # timelines[m] = list of segments

    for evt in result.events:
        m = evt.needle_id
        seg = {
            't0': evt.start_time,
            't1': evt.end_time,
            'phase': evt.phase,
            'cell_id': evt.cell_id,
        }
        if evt.phase in (Phase.MOVING, Phase.RETRACTING):
            seg['p0'] = evt.pos_start if evt.pos_start is not None else evt.pos_end
            seg['p1'] = evt.pos_end if evt.pos_end is not None else evt.pos_start
        else:
            pos = evt.pos_end if evt.pos_end is not None else (
                  evt.pos_start if evt.pos_start is not None else None)
            seg['p0'] = pos
            seg['p1'] = pos
        timelines[m].append(seg)

    # 按开始时间排序
    for m in range(M):
        timelines[m].sort(key=lambda s: s['t0'])

    return timelines


def query_state_at(t: float, timelines, needles: list[Needle]):
    """
    查询时刻 t 每根针的 (position, phase, cell_id).
    返回: positions[m], phases[m], cell_ids[m]
    """
    M = len(needles)
    positions = []
    phases = []
    cell_ids = []

    for m in range(M):
        tl = timelines[m]
        found = False
        for seg in reversed(tl):  # 从最新往回找
            if seg['t0'] <= t + 1e-9:
                phase = seg['phase']
                cid = seg['cell_id']

                if seg['p0'] is None:
                    positions.append(needles[m].init_pos.copy())
                elif phase in (Phase.MOVING, Phase.RETRACTING) and seg['t1'] > seg['t0']:
                    # 线性插值
                    alpha = min(1.0, max(0.0, (t - seg['t0']) / (seg['t1'] - seg['t0'])))
                    pos = seg['p0'] * (1 - alpha) + seg['p1'] * alpha
                    positions.append(pos)
                    # 如果已过终点时间, 更新phase
                    if t >= seg['t1'] - 1e-9:
                        positions[-1] = seg['p1'].copy()
                else:
                    positions.append(seg['p1'].copy() if seg['p1'] is not None
                                     else needles[m].init_pos.copy())

                # 如果t超过该段结尾但没有后续段, 针可能已finished
                if t > seg['t1'] + 1e-9:
                    # 检查是不是最后一段
                    idx = tl.index(seg)
                    if idx == len(tl) - 1:
                        if phase == Phase.FINISHED:
                            pass  # 保持
                        else:
                            phase = Phase.IDLE  # 间隙

                phases.append(phase)
                cell_ids.append(cid)
                found = True
                break

        if not found:
            # t 在所有事件之前
            positions.append(needles[m].init_pos.copy())
            phases.append(Phase.IDLE)
            cell_ids.append(-1)

    return positions, phases, cell_ids


# ==================================================================
# 碰撞对检测 (用于可视化高亮)
# ==================================================================

def find_collisions(needles: list[Needle], positions: list[np.ndarray],
                    cfg: SimConfig):
    """
    返回碰撞对列表: [(i, j, collision_type, distance), ...]
    collision_type: 'tip-tip', 'tip-body', 'body-body'
    """
    M = len(needles)
    collisions = []
    for i in range(M):
        ti = needles[i].tip_pos(positions[i])
        bi = positions[i] + cfg.body_check_length * needles[i].direction
        for j in range(i + 1, M):
            tj = needles[j].tip_pos(positions[j])
            bj = positions[j] + cfg.body_check_length * needles[j].direction

            # tip-tip
            d_tt = np.linalg.norm(ti - tj)
            if d_tt < cfg.d_tip:
                collisions.append((i, j, 'tip-tip', d_tt))

            # tip-body (双向)
            d_tb1 = point_to_segment_dist(ti, tj, bj)
            if d_tb1 < cfg.d_tip:
                collisions.append((i, j, 'tip-body', d_tb1))
            d_tb2 = point_to_segment_dist(tj, ti, bi)
            if d_tb2 < cfg.d_tip:
                collisions.append((j, i, 'tip-body', d_tb2))

            # body-body
            d_bb = segment_to_segment_dist(ti, bi, tj, bj)
            if d_bb < cfg.d_body:
                collisions.append((i, j, 'body-body', d_bb))

    return collisions


# ==================================================================
# 交互式查看器
# ==================================================================

def interactive_viewer(result: SimResult,
                       cells: list[Cell],
                       needles: list[Needle],
                       sim_cfg: SimConfig = None):
    """
    打开交互式窗口:
    - 左上: 2D 针体视图 (针身线段 + 针尖 + 细胞), 碰撞用红色虚线高亮
    - 右上: 甘特图 + 当前时间线
    - 下方: 时间滑动条 + 状态信息面板
    """
    if sim_cfg is None:
        sim_cfg = SimConfig()

    M = len(needles)
    timelines = build_timeline(result, needles)
    t_max = result.f1_makespan if result.f1_makespan < 1e5 else 2000.0

    # 绘图用针身长度 (只画一段)
    DRAW_LEN = 600.0

    # ---- 收集所有坐标用于确定绘图范围 (只用draw_base_pos) ----
    all_x, all_y = [], []
    for c in cells:
        all_x.append(c.x); all_y.append(c.y)
    for n in needles:
        all_x.append(n.init_pos[0]); all_y.append(n.init_pos[1])
        bp = n.draw_base_pos(n.init_pos, DRAW_LEN)
        all_x.append(bp[0]); all_y.append(bp[1])
    for evt in result.events:
        for p in (evt.pos_start, evt.pos_end):
            if p is not None:
                bp = needles[evt.needle_id].draw_base_pos(p, DRAW_LEN)
                all_x.extend([p[0], bp[0]])
                all_y.extend([p[1], bp[1]])

    margin = 200
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    # ---- 创建图形 ----
    fig = plt.figure(figsize=(18, 11))
    fig.canvas.manager.set_window_title('多针调度 - 交互式碰撞查看器')
    gs = GridSpec(3, 2, figure=fig, height_ratios=[5, 5, 1.2],
                  hspace=0.32, wspace=0.25)

    ax_2d = fig.add_subplot(gs[0, 0])        # 2D 针体
    ax_gantt = fig.add_subplot(gs[0, 1])      # 甘特图
    ax_info = fig.add_subplot(gs[1, :])       # 状态信息
    ax_slider = fig.add_subplot(gs[2, :])     # 滑动条

    # ---- 时间滑动条 ----
    slider = Slider(ax_slider, '时间 (s)', 0.0, t_max,
                    valinit=0.0, valstep=0.5,
                    color='#3498DB')

    # ---- 绘制静态甘特图 ----
    def draw_gantt():
        ax_gantt.clear()
        ax_gantt.set_title('甘特图', fontsize=12)
        for evt in result.events:
            m_id = evt.needle_id
            dur = evt.end_time - evt.start_time
            if dur < 0.1:
                continue
            c = PHASE_COLORS_GANTT.get(evt.phase, '#CCCCCC')
            ax_gantt.barh(m_id, dur, left=evt.start_time,
                          color=c, edgecolor='white', linewidth=0.3,
                          height=0.7, alpha=0.85)
            # 细胞标注
            if evt.phase == Phase.FINDING and evt.cell_id >= 0 and dur > 15:
                ax_gantt.text(evt.start_time + dur / 2, m_id,
                              f'C{evt.cell_id}', ha='center', va='center',
                              fontsize=6, color='white', fontweight='bold')

        ax_gantt.set_yticks(range(M))
        ax_gantt.set_yticklabels([f'针{m}' for m in range(M)])
        ax_gantt.set_xlabel('时间 (秒)')
        ax_gantt.invert_yaxis()
        ax_gantt.grid(True, axis='x', alpha=0.3)
        ax_gantt.set_xlim(-5, t_max + 5)

        # 图例
        legend_patches = [
            mpatches.Patch(color=PHASE_COLORS_GANTT[p], label=n)
            for p, n in [
                (Phase.MOVING, '移动'), (Phase.FINDING, 'Find'),
                (Phase.WAITING, 'Wait'), (Phase.RETRACTING, '退让'),
                (Phase.BLOCKED, '阻塞'),
            ]
        ]
        ax_gantt.legend(handles=legend_patches, loc='lower right',
                        fontsize=7, ncol=3)

    draw_gantt()

    # 甘特图上的时间线 (会被更新)
    gantt_line = ax_gantt.axvline(x=0, color='red', linewidth=2, linestyle='-')

    # ---- 更新函数 ----
    def update(t):
        positions, phases, cell_ids = query_state_at(t, timelines, needles)
        collisions = find_collisions(needles, positions, sim_cfg)

        # --- 更新 2D 针体视图 ---
        ax_2d.clear()
        ax_2d.set_aspect('equal')
        ax_2d.set_xlim(x_min, x_max)
        ax_2d.set_ylim(y_min, y_max)
        ax_2d.set_title(f'2D 针体视图  t = {t:.1f}s', fontsize=12, fontweight='bold')
        ax_2d.grid(True, alpha=0.15)
        ax_2d.set_xlabel('X (μm)')
        ax_2d.set_ylabel('Y (μm)')

        # 绘制细胞
        for c in cells:
            nc = NEEDLE_COLORS[c.color % len(NEEDLE_COLORS)]
            # 判断该细胞是否正在被处理
            being_processed = False
            for m_id in range(M):
                if cell_ids[m_id] == c.id and phases[m_id] in (Phase.FINDING, Phase.WAITING):
                    being_processed = True
                    break

            if being_processed:
                ax_2d.plot(c.x, c.y, 'o', color=nc, markersize=14, alpha=1.0,
                           markeredgecolor='black', markeredgewidth=2, zorder=3)
            else:
                ax_2d.plot(c.x, c.y, 'o', color=nc, markersize=9, alpha=0.4,
                           markeredgecolor='gray', markeredgewidth=0.5, zorder=2)
            ax_2d.annotate(f'C{c.id}', (c.x, c.y), fontsize=7,
                           ha='center', va='bottom', xytext=(0, 7),
                           textcoords='offset points', color='#555555')

        # 绘制每根针: 针身线段(只画DRAW_LEN长度) + 针尖标记
        for m_id in range(M):
            nc = NEEDLE_COLORS[m_id % len(NEEDLE_COLORS)]
            pos = positions[m_id]
            tip = needles[m_id].tip_pos(pos)
            draw_base = needles[m_id].draw_base_pos(pos, DRAW_LEN)
            phase = phases[m_id]

            # 针身 (只画可见部分)
            lw = 4 if phase in (Phase.FINDING, Phase.WAITING) else 2.5
            alpha = 1.0 if phase != Phase.FINISHED else 0.3
            ax_2d.plot([tip[0], draw_base[0]], [tip[1], draw_base[1]],
                       '-', color=nc, linewidth=lw, alpha=alpha,
                       solid_capstyle='round', zorder=4)
            # 尾端箭头 (表示针身继续延伸)
            d = needles[m_id].direction
            ax_2d.annotate('', xy=(draw_base[0] + d[0]*30, draw_base[1] + d[1]*30),
                           xytext=(draw_base[0], draw_base[1]),
                           arrowprops=dict(arrowstyle='->', color=nc,
                                           lw=1.5, alpha=alpha * 0.6))
            # 针尖 (实心圆, 大)
            ax_2d.plot(tip[0], tip[1], 'o', color=nc, markersize=9,
                       markeredgecolor='black', markeredgewidth=1.5,
                       zorder=6, alpha=alpha)

            # 标签
            ax_2d.annotate(f'针{m_id}\n{PHASE_NAMES[phase]}',
                           (tip[0], tip[1]), fontsize=7, fontweight='bold',
                           ha='left', va='top', xytext=(8, -5),
                           textcoords='offset points', color=nc,
                           bbox=dict(boxstyle='round,pad=0.2',
                                     facecolor='white', alpha=0.7,
                                     edgecolor=nc, linewidth=0.5))

        # 绘制碰撞高亮 (画图只用 draw_base_pos)
        collision_set = set()
        for (i, j, ctype, dist) in collisions:
            key = (min(i, j), max(i, j))
            collision_set.add(key)
            ti = needles[i].tip_pos(positions[i])
            dbi = needles[i].draw_base_pos(positions[i], DRAW_LEN)
            tj = needles[j].tip_pos(positions[j])
            dbj = needles[j].draw_base_pos(positions[j], DRAW_LEN)

            if ctype == 'tip-tip':
                ax_2d.plot([ti[0], tj[0]], [ti[1], tj[1]],
                           '--', color='red', linewidth=2, alpha=0.8, zorder=10)
                mid = (ti + tj) / 2
                ax_2d.plot(mid[0], mid[1], 'X', color='red', markersize=14,
                           markeredgewidth=2, zorder=11)
            elif ctype == 'body-body':
                # 两根针身都画红色粗边 (只画可见段)
                ax_2d.plot([ti[0], dbi[0]], [ti[1], dbi[1]],
                           '-', color='red', linewidth=6, alpha=0.35, zorder=3)
                ax_2d.plot([tj[0], dbj[0]], [tj[1], dbj[1]],
                           '-', color='red', linewidth=6, alpha=0.35, zorder=3)
            elif ctype == 'tip-body':
                ax_2d.plot(ti[0], ti[1], 'X', color='red', markersize=12,
                           markeredgewidth=2, zorder=11)

        # --- 更新甘特图时间线 ---
        gantt_line.set_xdata([t, t])

        # --- 更新状态信息面板 ---
        ax_info.clear()
        ax_info.axis('off')

        # 表头
        col_labels = ['针', '阶段', '针尖 X', '针尖 Y', '当前细胞', '碰撞']
        table_data = []

        for m_id in range(M):
            pos = positions[m_id]
            phase = phases[m_id]
            cid = cell_ids[m_id]
            cid_str = f'C{cid}' if cid >= 0 else '-'
            # 检查碰撞
            involved = any(m_id in (i, j) for (i, j) in collision_set)
            col_str = 'YES !!!' if involved else 'safe'

            table_data.append([
                f'针{m_id}',
                PHASE_NAMES[phase],
                f'{pos[0]:.1f}',
                f'{pos[1]:.1f}',
                cid_str,
                col_str,
            ])

        # 碰撞详情
        collision_detail_lines = []
        for (i, j, ctype, dist) in collisions:
            collision_detail_lines.append(
                f'  !!! 针{i} vs 针{j}: {ctype}  距离={dist:.1f}μm')

        # 绘制表格
        table = ax_info.table(cellText=table_data, colLabels=col_labels,
                              loc='upper left', cellLoc='center',
                              colWidths=[0.06, 0.12, 0.08, 0.08, 0.08, 0.08])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        # 表头颜色
        for j_col in range(len(col_labels)):
            table[0, j_col].set_facecolor('#34495E')
            table[0, j_col].set_text_props(color='white', fontweight='bold')

        # 行颜色
        for row in range(len(table_data)):
            nc = NEEDLE_COLORS[row % len(NEEDLE_COLORS)]
            table[row + 1, 0].set_text_props(color=nc, fontweight='bold')
            # 碰撞列红色
            if table_data[row][-1] != 'safe':
                table[row + 1, 5].set_facecolor('#FFE0E0')
                table[row + 1, 5].set_text_props(color='red', fontweight='bold')
            else:
                table[row + 1, 5].set_text_props(color='green')

        # 碰撞详情文本
        if collision_detail_lines:
            detail = '\n'.join(collision_detail_lines)
            ax_info.text(0.55, 0.85, f'碰撞警告:\n{detail}',
                         transform=ax_info.transAxes, fontsize=10,
                         color='red', fontweight='bold', verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='#FFF0F0',
                                   edgecolor='red', alpha=0.9))
        else:
            ax_info.text(0.55, 0.85, '无碰撞 - 所有针安全',
                         transform=ax_info.transAxes, fontsize=11,
                         color='green', fontweight='bold', verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='#F0FFF0',
                                   edgecolor='green', alpha=0.9))

        # 时间信息
        ax_info.text(0.55, 0.45,
                     f't = {t:.1f}s / {t_max:.1f}s   ({t/t_max*100:.1f}%)',
                     transform=ax_info.transAxes, fontsize=12,
                     color='#333333', verticalalignment='top')

        fig.canvas.draw_idle()

    # 连接滑动条
    slider.on_changed(update)

    # ---- 播放/暂停按钮 ----
    ax_play = fig.add_axes([0.42, 0.01, 0.06, 0.025])
    ax_step_fwd = fig.add_axes([0.49, 0.01, 0.04, 0.025])
    ax_step_bwd = fig.add_axes([0.37, 0.01, 0.04, 0.025])
    btn_play = Button(ax_play, '播放')
    btn_fwd = Button(ax_step_fwd, '>>>')
    btn_bwd = Button(ax_step_bwd, '<<<')

    play_state = {'playing': False, 'timer': None}

    def on_play(event):
        if play_state['playing']:
            play_state['playing'] = False
            btn_play.label.set_text('播放')
            if play_state['timer'] is not None:
                play_state['timer'].stop()
        else:
            play_state['playing'] = True
            btn_play.label.set_text('暂停')
            play_state['timer'] = fig.canvas.new_timer(interval=50)

            def advance():
                if not play_state['playing']:
                    return
                cur = slider.val
                step = t_max / 300  # ~300帧走完全程
                nxt = cur + step
                if nxt > t_max:
                    nxt = 0.0
                    play_state['playing'] = False
                    btn_play.label.set_text('播放')
                    play_state['timer'].stop()
                slider.set_val(nxt)

            play_state['timer'].add_callback(advance)
            play_state['timer'].start()

    def on_step_fwd(event):
        cur = slider.val
        # 找到当前时间之后最近的事件边界
        event_times = set()
        for evt in result.events:
            event_times.add(evt.start_time)
            event_times.add(evt.end_time)
        event_times = sorted(event_times)
        for et in event_times:
            if et > cur + 0.01:
                slider.set_val(et)
                return
        slider.set_val(t_max)

    def on_step_bwd(event):
        cur = slider.val
        event_times = set()
        for evt in result.events:
            event_times.add(evt.start_time)
            event_times.add(evt.end_time)
        event_times = sorted(event_times, reverse=True)
        for et in event_times:
            if et < cur - 0.01:
                slider.set_val(et)
                return
        slider.set_val(0.0)

    btn_play.on_clicked(on_play)
    btn_fwd.on_clicked(on_step_fwd)
    btn_bwd.on_clicked(on_step_bwd)

    # 初始绘制
    update(0.0)
    plt.show()


# ==================================================================
# 独立运行入口
# ==================================================================
if __name__ == '__main__':
    from main import generate_demo_scenario
    from problem import decode_individual
    from simulator import simulate

    print("生成演示场景...")
    cells, needles = generate_demo_scenario(seed=42)
    sim_cfg = SimConfig()

    # 使用默认顺序做一次仿真
    sequences = [list(n.cells) for n in needles]
    retractions = [0] * len(cells)

    print("运行仿真...")
    result = simulate(sequences, retractions, cells, needles, sim_cfg,
                      record_events=True)
    print(f"Makespan={result.f1_makespan:.1f}s  deadlock={result.deadlock}")

    print("启动交互式查看器...")
    interactive_viewer(result, cells, needles, sim_cfg)
