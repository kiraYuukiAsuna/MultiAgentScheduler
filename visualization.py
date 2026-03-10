"""
多针细胞染料灌注调度 - 可视化模块
包含: 收敛曲线, Pareto前沿, 甘特图, 2D轨迹与针体展示
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

from config import Cell, Needle, SimConfig
from simulator import SimResult, Phase, SimEvent


# 中文字体 & 全局样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 阶段颜色映射
PHASE_COLORS = {
    Phase.IDLE:       '#CCCCCC',
    Phase.MOVING:     '#4A90D9',
    Phase.FINDING:    '#E8553A',
    Phase.WAITING:    '#F5A623',
    Phase.RETRACTING: '#9B59B6',
    Phase.BLOCKED:    '#2C3E50',
    Phase.FINISHED:   '#27AE60',
}

PHASE_NAMES = {
    Phase.IDLE:       '空闲',
    Phase.MOVING:     '移动',
    Phase.FINDING:    'Find(占相机)',
    Phase.WAITING:    'Wait(等扩散)',
    Phase.RETRACTING: '退让',
    Phase.BLOCKED:    '阻塞',
    Phase.FINISHED:   '完成',
}

NEEDLE_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
                 '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']


def plot_convergence(callback, save_path=None):
    """绘制收敛曲线 (4个目标的每代最优值)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NSGA-II 收敛曲线', fontsize=16, fontweight='bold')

    gens = range(1, len(callback.history_f1) + 1)

    titles = ['f1: Makespan (秒)', 'f2: 总移动距离 (μm)',
              'f3: 退让次数', 'f4: 总阻塞时间 (秒)']
    data = [callback.history_f1, callback.history_f2,
            callback.history_f3, callback.history_f4]
    colors = ['#E74C3C', '#3498DB', '#9B59B6', '#2C3E50']

    for ax, title, d, color in zip(axes.flat, titles, data, colors):
        ax.plot(list(gens), d, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('代数')
        ax.set_ylabel('最优值')
        ax.grid(True, alpha=0.3)
        # 标注最终值
        if d:
            ax.annotate(f'{d[-1]:.1f}', xy=(len(d), d[-1]),
                        fontsize=9, color=color,
                        xytext=(-40, 10), textcoords='offset points')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if save_path:
        plt.close(fig)
    else:
        plt.show()


def plot_pareto_front(callback, save_path=None):
    """绘制最终代的Pareto前沿 (f1 vs f2, f1 vs f3, f1 vs f4 等多组合)"""
    if not callback.all_F:
        return

    F = callback.all_F[-1]
    # 过滤死锁个体
    mask = F[:, 0] < 1e5
    F_valid = F[mask]

    if len(F_valid) == 0:
        print("警告: 无可行解，无法绘制Pareto前沿")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Pareto 前沿 (最终代)', fontsize=16, fontweight='bold')

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    labels = ['Makespan(s)', '总距离(μm)', '退让次数', '阻塞时间(s)']

    for ax, (i, j) in zip(axes.flat, pairs):
        ax.scatter(F_valid[:, i], F_valid[:, j], c='#3498DB', alpha=0.6, s=20)
        ax.set_xlabel(labels[i], fontsize=10)
        ax.set_ylabel(labels[j], fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if save_path:
        plt.close(fig)
    else:
        plt.show()


def plot_gantt(result: SimResult, needles: list[Needle], title="调度甘特图",
               save_path=None):
    """
    绘制甘特图: 每根针一行，按时间展示各阶段
    标注退让事件及退让距离
    """
    M = len(needles)
    fig, ax = plt.subplots(figsize=(18, max(4, M * 1.5)))
    ax.set_title(title, fontsize=14, fontweight='bold')

    y_labels = [f'针 {m} (θ={np.degrees(needles[m].theta_xy):.0f}°)' for m in range(M)]

    for evt in result.events:
        m = evt.needle_id
        duration = evt.end_time - evt.start_time
        if duration < 0.1:
            continue

        color = PHASE_COLORS.get(evt.phase, '#CCCCCC')
        bar = ax.barh(m, duration, left=evt.start_time,
                      color=color, edgecolor='white', linewidth=0.5,
                      height=0.7, alpha=0.85)

        # 标注退让距离
        if evt.phase == Phase.RETRACTING and evt.retract_delta > 0:
            mid_x = evt.start_time + duration / 2
            ax.text(mid_x, m, f'δ={evt.retract_delta:.0f}',
                    ha='center', va='center', fontsize=7, color='white',
                    fontweight='bold')

        # 标注细胞ID
        if evt.phase in (Phase.FINDING, Phase.WAITING) and evt.cell_id >= 0:
            if duration > 10:
                mid_x = evt.start_time + duration / 2
                label = f'C{evt.cell_id}' if evt.phase == Phase.FINDING else ''
                if label:
                    ax.text(mid_x, m, label, ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')

    ax.set_yticks(range(M))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('时间 (秒)')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    # 图例
    legend_patches = [mpatches.Patch(color=PHASE_COLORS[p], label=PHASE_NAMES[p])
                      for p in [Phase.MOVING, Phase.FINDING, Phase.WAITING,
                                Phase.RETRACTING, Phase.BLOCKED, Phase.IDLE]]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9, ncol=3)

    # 标注makespan
    ax.axvline(x=result.f1_makespan, color='red', linestyle='--', linewidth=1.5)
    ax.text(result.f1_makespan, -0.5, f'Makespan={result.f1_makespan:.1f}s',
            color='red', fontsize=10, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if save_path:
        plt.close(fig)
    else:
        plt.show()


def plot_2d_trajectory(result: SimResult, cells: list[Cell],
                       needles: list[Needle], title="2D 轨迹与针体",
                       save_path=None):
    """
    绘制2D俯视图:
    - 细胞位置 (按颜色标记)
    - 每根针的移动轨迹
    - 退让路径 (虚线)
    - 针的初始位置
    """
    M = len(needles)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    # 绘制细胞
    for c in cells:
        color = NEEDLE_COLORS[c.color % len(NEEDLE_COLORS)]
        ax.plot(c.x, c.y, 'o', color=color, markersize=10, alpha=0.8,
                markeredgecolor='black', markeredgewidth=0.5)
        ax.annotate(f'C{c.id}', (c.x, c.y), fontsize=7,
                    ha='center', va='bottom', xytext=(0, 6),
                    textcoords='offset points')

    # 绘制针初始位置和方向
    for m, needle in enumerate(needles):
        color = NEEDLE_COLORS[m % len(NEEDLE_COLORS)]
        ip = needle.init_pos
        ax.plot(ip[0], ip[1], 's', color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.annotate(f'针{m}起点', (ip[0], ip[1]), fontsize=8,
                    ha='center', va='top', xytext=(0, -10),
                    textcoords='offset points', color=color,
                    fontweight='bold')

        # 画针的方向箭头
        d = needle.direction * needle.length * 0.3
        ax.annotate('', xy=(ip[0] + d[0], ip[1] + d[1]),
                    xytext=(ip[0], ip[1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # 绘制移动轨迹
    for evt in result.events:
        if evt.pos_start is None or evt.pos_end is None:
            continue
        m = evt.needle_id
        color = NEEDLE_COLORS[m % len(NEEDLE_COLORS)]

        if evt.phase == Phase.MOVING:
            ax.plot([evt.pos_start[0], evt.pos_end[0]],
                    [evt.pos_start[1], evt.pos_end[1]],
                    '-', color=color, linewidth=1.5, alpha=0.6)
            # 箭头
            mid_x = (evt.pos_start[0] + evt.pos_end[0]) / 2
            mid_y = (evt.pos_start[1] + evt.pos_end[1]) / 2
            dx = evt.pos_end[0] - evt.pos_start[0]
            dy = evt.pos_end[1] - evt.pos_start[1]
            if abs(dx) + abs(dy) > 1:
                ax.annotate('', xy=(mid_x + dx * 0.01, mid_y + dy * 0.01),
                            xytext=(mid_x, mid_y),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1))

        elif evt.phase == Phase.RETRACTING:
            ax.plot([evt.pos_start[0], evt.pos_end[0]],
                    [evt.pos_start[1], evt.pos_end[1]],
                    '--', color=color, linewidth=2, alpha=0.8)
            # 标记退让终点
            ax.plot(evt.pos_end[0], evt.pos_end[1], 'x', color=color,
                    markersize=8, markeredgewidth=2)

    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.grid(True, alpha=0.2)

    # 图例
    legend_items = []
    for m in range(M):
        color = NEEDLE_COLORS[m % len(NEEDLE_COLORS)]
        legend_items.append(mpatches.Patch(
            color=color,
            label=f'针{m} (θ={np.degrees(needles[m].theta_xy):.0f}°)'))
    ax.legend(handles=legend_items, loc='upper left', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if save_path:
        plt.close(fig)
    else:
        plt.show()


def plot_needle_snapshot(cells: list[Cell], needles: list[Needle],
                         positions: list[np.ndarray],
                         title="针体快照", ax=None, save_path=None):
    """
    绘制某一时刻所有针的2D位置和针身线段
    用于交互式动画或单帧展示
    """
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')

    ax.set_title(title, fontsize=12)

    # 细胞
    for c in cells:
        color = NEEDLE_COLORS[c.color % len(NEEDLE_COLORS)]
        ax.plot(c.x, c.y, 'o', color=color, markersize=8, alpha=0.5)

    # 针身 (只画可见段)
    for m, (needle, pos) in enumerate(zip(needles, positions)):
        color = NEEDLE_COLORS[m % len(NEEDLE_COLORS)]
        tip = needle.tip_pos(pos)
        draw_base = needle.draw_base_pos(pos, 600.0)
        ax.plot([tip[0], draw_base[0]], [tip[1], draw_base[1]],
                '-', color=color, linewidth=3, alpha=0.8, solid_capstyle='round')
        ax.plot(tip[0], tip[1], 'o', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=1, zorder=5)

    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')

    if show:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_summary(result: SimResult, needles: list[Needle],
                 cells: list[Cell], title="调度方案总览"):
    """
    综合展示: 甘特图 + 2D轨迹 + 统计指标
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # --- 统计指标 (左上) ---
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis('off')
    info_text = (
        f"{'='*30}\n"
        f"  调度方案执行摘要\n"
        f"{'='*30}\n\n"
        f"  Makespan:    {result.f1_makespan:>10.1f} 秒\n"
        f"  总移动距离:  {result.f2_total_dist:>10.1f} μm\n"
        f"  退让次数:    {result.f3_retract_count:>10d}\n"
        f"  总阻塞时间:  {result.f4_blocked_time:>10.1f} 秒\n"
        f"  死锁:        {'是' if result.deadlock else '否':>10s}\n\n"
        f"  针数: {len(needles)}  细胞数: {len(cells)}\n"
    )
    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- 甘特图 (右上, 跨2列) ---
    ax_gantt = fig.add_subplot(gs[0, 1:])
    M = len(needles)
    y_labels = [f'针{m}' for m in range(M)]

    for evt in result.events:
        m = evt.needle_id
        duration = evt.end_time - evt.start_time
        if duration < 0.1:
            continue
        color = PHASE_COLORS.get(evt.phase, '#CCCCCC')
        ax_gantt.barh(m, duration, left=evt.start_time,
                      color=color, edgecolor='white', linewidth=0.5,
                      height=0.7, alpha=0.85)
        if evt.phase in (Phase.FINDING,) and evt.cell_id >= 0 and duration > 10:
            mid_x = evt.start_time + duration / 2
            ax_gantt.text(mid_x, m, f'C{evt.cell_id}', ha='center', va='center',
                          fontsize=6, color='white', fontweight='bold')

    ax_gantt.set_yticks(range(M))
    ax_gantt.set_yticklabels(y_labels)
    ax_gantt.set_xlabel('时间 (秒)')
    ax_gantt.set_title('甘特图')
    ax_gantt.invert_yaxis()
    ax_gantt.grid(True, axis='x', alpha=0.3)
    ax_gantt.axvline(x=result.f1_makespan, color='red', linestyle='--', linewidth=1)

    legend_patches = [mpatches.Patch(color=PHASE_COLORS[p], label=PHASE_NAMES[p])
                      for p in [Phase.MOVING, Phase.FINDING, Phase.WAITING,
                                Phase.RETRACTING, Phase.BLOCKED]]
    ax_gantt.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=3)

    # --- 2D轨迹 (左下, 跨2列) ---
    ax_traj = fig.add_subplot(gs[1, :2])
    ax_traj.set_aspect('equal')
    ax_traj.set_title('2D 轨迹')

    for c in cells:
        color = NEEDLE_COLORS[c.color % len(NEEDLE_COLORS)]
        ax_traj.plot(c.x, c.y, 'o', color=color, markersize=8, alpha=0.7,
                     markeredgecolor='black', markeredgewidth=0.5)
        ax_traj.annotate(f'C{c.id}', (c.x, c.y), fontsize=6,
                         ha='center', va='bottom', xytext=(0, 5),
                         textcoords='offset points')

    for m, needle in enumerate(needles):
        color = NEEDLE_COLORS[m % len(NEEDLE_COLORS)]
        ip = needle.init_pos
        ax_traj.plot(ip[0], ip[1], 's', color=color, markersize=10,
                     markeredgecolor='black', markeredgewidth=1, zorder=5)

    for evt in result.events:
        if evt.pos_start is None or evt.pos_end is None:
            continue
        m = evt.needle_id
        color = NEEDLE_COLORS[m % len(NEEDLE_COLORS)]
        if evt.phase == Phase.MOVING:
            ax_traj.plot([evt.pos_start[0], evt.pos_end[0]],
                         [evt.pos_start[1], evt.pos_end[1]],
                         '-', color=color, linewidth=1.2, alpha=0.5)
        elif evt.phase == Phase.RETRACTING:
            ax_traj.plot([evt.pos_start[0], evt.pos_end[0]],
                         [evt.pos_start[1], evt.pos_end[1]],
                         '--', color=color, linewidth=2, alpha=0.8)

    ax_traj.grid(True, alpha=0.2)
    ax_traj.set_xlabel('X (μm)')
    ax_traj.set_ylabel('Y (μm)')

    # --- 阶段时间饼图 (右下) ---
    ax_pie = fig.add_subplot(gs[1, 2])
    phase_totals = {}
    for evt in result.events:
        dur = evt.end_time - evt.start_time
        if dur > 0:
            phase_totals[evt.phase] = phase_totals.get(evt.phase, 0) + dur

    if phase_totals:
        labels = [PHASE_NAMES[p] for p in phase_totals.keys()]
        sizes = list(phase_totals.values())
        colors = [PHASE_COLORS[p] for p in phase_totals.keys()]
        ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 9})
        ax_pie.set_title('时间分布')

    if not result.deadlock:
        plt.savefig('summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_generation_snapshot(callback, gen_idx: int = -1, save_path=None):
    """绘制某一代的种群目标值分布 (f1 vs f2)"""
    if not callback.all_F:
        return
    F = callback.all_F[gen_idx]
    mask = F[:, 0] < 1e5
    F_valid = F[mask]
    if len(F_valid) == 0:
        print(f"Gen {gen_idx}: 无可行解")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(F_valid[:, 0], F_valid[:, 1], c='#3498DB', alpha=0.6, s=20)
    ax.set_xlabel('Makespan (秒)')
    ax.set_ylabel('总距离 (μm)')
    gen_label = gen_idx if gen_idx >= 0 else len(callback.all_F) + gen_idx
    ax.set_title(f'第 {gen_label + 1} 代 种群分布 (f1 vs f2)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
