"""
多针细胞染料灌注调度 - 主入口
基于 MathModel8.tex v8.0
使用 pymoo NSGA-II 求解，带交互式可视化
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.selection.tournament import compare as tournament_compare

from config import Cell, Needle, SimConfig, GAConfig
from problem import (MultiNeedleProblem, MultiNeedleSampling,
                     MultiNeedleCrossover, MultiNeedleMutation,
                     ProgressCallback, decode_individual, encode_individual)
from simulator import simulate
from visualization import (plot_convergence, plot_pareto_front,
                           plot_gantt, plot_2d_trajectory,
                           plot_summary, plot_generation_snapshot)


def generate_demo_scenario(seed: int = 42):
    """
    生成演示场景:
    - 3根针, 各分配若干细胞
    - 细胞随机分布在 2000x2000 μm 区域
    - 针从不同角度安装, 针身朝工作区域外侧延伸
    """
    rng = np.random.default_rng(seed)
    M = 3   # 3根针
    N = 12  # 12个细胞

    # 细胞坐标: 在 2000x2000 区域随机分布
    cells = []
    for i in range(N):
        x = rng.uniform(400, 1600)
        y = rng.uniform(400, 1600)
        color = i % M  # 轮流分配颜色
        cells.append(Cell(id=i, color=color, x=x, y=y))

    # 针参数: 3根针，120度间隔安装
    # 物理模型: 针从工作区域外侧伸入, 针尖靠近center, 针座(尾部)远离center
    # theta_xy = 针尖→针座方向 = 从center指向init_pos的方向
    needles = []
    center = np.array([1000.0, 1000.0])
    for m in range(M):
        # away_angle: 用于计算init_pos的辅助角度
        away_angle = np.radians(90 + m * 120)  # 90°, 210°, 330°

        # init_pos: 针尖在center的对侧 (外侧)
        init_pos = center - 1500.0 * np.array([np.cos(away_angle),
                                                np.sin(away_angle)])

        # theta_xy: 针尖→针座 = 从center朝向init_pos的方向
        # center→init_pos 方向 = away_angle + π
        theta_xy = away_angle + np.pi

        needle = Needle(
            id=m,
            theta_xy=theta_xy,
            length=200000.0,          # 近似无限长
            init_pos=init_pos,
            cells=[c.id for c in cells if c.color == m]
        )
        needles.append(needle)

    return cells, needles


def binary_tournament(pop, P, **kwargs):
    """二元锦标赛选择"""
    n_tournaments, n_competitors = P.shape
    S = np.full(n_tournaments, -1, dtype=int)
    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        # 比较非支配等级和拥挤度
        S[i] = tournament_compare(a, b,
                                  pop[a].get("rank"), pop[b].get("rank"),
                                  pop[a].get("crowding"), pop[b].get("crowding"))
        if S[i] == -1:
            S[i] = P[i, np.random.randint(2)]
    return S


def run_optimization(cells, needles, sim_cfg=None, ga_cfg=None):
    """运行 NSGA-II 优化"""
    if sim_cfg is None:
        sim_cfg = SimConfig()
    if ga_cfg is None:
        ga_cfg = GAConfig()

    n_cells = len(cells)
    M = len(needles)

    print("=" * 60)
    print("  多针细胞染料灌注调度 - NSGA-II 优化")
    print("=" * 60)
    print(f"  针数: {M}  细胞数: {n_cells}")
    print(f"  各针细胞数: {[len(n.cells) for n in needles]}")
    print(f"  种群: {ga_cfg.pop_size}  代数: {ga_cfg.n_gen}")
    print(f"  t_find={sim_cfg.t_find}s  t_wait={sim_cfg.t_wait}s  v={sim_cfg.v}μm/s")
    print(f"  d_tip={sim_cfg.d_tip}μm  d_body={sim_cfg.d_body}μm")
    print("=" * 60)

    # 构建问题
    problem = MultiNeedleProblem(cells, needles, sim_cfg)

    # 自定义算子
    sampling = MultiNeedleSampling(needles, n_cells)
    crossover = MultiNeedleCrossover(needles, n_cells, prob=ga_cfg.crossover_prob)
    mutation = MultiNeedleMutation(needles, n_cells, prob=ga_cfg.mutation_prob)

    # 回调
    callback = ProgressCallback(display_interval=ga_cfg.display_interval)

    # NSGA-II 算法
    algorithm = NSGA2(
        pop_size=ga_cfg.pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=False,
    )

    termination = get_termination("n_gen", ga_cfg.n_gen)

    print("\n开始优化...\n")
    res = minimize(
        problem,
        algorithm,
        termination,
        callback=callback,
        seed=ga_cfg.seed,
        verbose=False,
    )

    print(f"\n优化完成! 共 {callback.gen_count} 代")
    print(f"Pareto前沿解数量: {len(res.F)}")

    return res, callback


def analyze_results(res, callback, cells, needles, sim_cfg=None):
    """分析和展示结果"""
    if sim_cfg is None:
        sim_cfg = SimConfig()

    F = res.F
    X = res.X

    # 过滤死锁解
    feasible_mask = F[:, 0] < 1e5
    n_feasible = feasible_mask.sum()
    print(f"\n可行解: {n_feasible}/{len(F)}")

    if n_feasible == 0:
        print("警告: 无可行解! 所有个体都发生了死锁。")
        print("建议: 增加种群大小、代数，或调整碰撞检测参数。")
        # 仍然展示收敛曲线
        plot_convergence(callback)
        return

    # === 1. 收敛曲线 ===
    print("\n[1/5] 绘制收敛曲线...")
    plot_convergence(callback, save_path='convergence.png')

    # === 2. Pareto前沿 ===
    print("[2/5] 绘制Pareto前沿...")
    plot_pareto_front(callback, save_path='pareto_front.png')

    # === 3. 选择最优Makespan解做详细分析 ===
    feasible_F = F[feasible_mask]
    feasible_X = X[feasible_mask]

    # 按makespan排序，选最优
    best_idx = np.argmin(feasible_F[:, 0])
    best_x = feasible_X[best_idx]
    best_f = feasible_F[best_idx]

    print(f"\n最优Makespan解:")
    print(f"  f1 (Makespan):   {best_f[0]:.1f} 秒")
    print(f"  f2 (总距离):     {best_f[1]:.1f} μm")
    print(f"  f3 (退让次数):   {best_f[2]:.0f}")
    print(f"  f4 (阻塞时间):   {best_f[3]:.1f} 秒")

    # 解码并重新仿真 (开启事件记录)
    sequences, retractions = decode_individual(best_x, needles, len(cells))
    print(f"\n  各针访问序列:")
    for m, seq in enumerate(sequences):
        print(f"    针{m}: {seq}")
    print(f"  退让意向: {retractions}")

    result = simulate(sequences, retractions, cells, needles, sim_cfg,
                      record_events=True)

    # === 4. 甘特图 ===
    print("\n[3/5] 绘制甘特图...")
    plot_gantt(result, needles, title="最优Makespan方案 - 甘特图",
              save_path='gantt_best_makespan.png')

    # === 5. 2D轨迹 ===
    print("[4/5] 绘制2D轨迹...")
    plot_2d_trajectory(result, cells, needles,
                       title="最优Makespan方案 - 2D轨迹",
                       save_path='trajectory_best_makespan.png')

    # === 6. 综合总览 ===
    print("[5/5] 绘制综合总览...")
    plot_summary(result, needles, cells,
                 title="最优Makespan方案 - 综合总览")

    # === 额外: 如果有多个Pareto解，也看看最优距离的方案 ===
    if n_feasible > 1:
        best_dist_idx = np.argmin(feasible_F[:, 1])
        if best_dist_idx != best_idx:
            best_dist_x = feasible_X[best_dist_idx]
            best_dist_f = feasible_F[best_dist_idx]
            print(f"\n最优距离解:")
            print(f"  f1={best_dist_f[0]:.1f}s  f2={best_dist_f[1]:.1f}μm  "
                  f"f3={best_dist_f[2]:.0f}  f4={best_dist_f[3]:.1f}s")

            seq2, ret2 = decode_individual(best_dist_x, needles, len(cells))
            result2 = simulate(seq2, ret2, cells, needles, sim_cfg,
                               record_events=True)
            plot_gantt(result2, needles, title="最优距离方案 - 甘特图",
                      save_path='gantt_best_dist.png')

    print("\n所有图表已保存到当前目录。")

    # === 交互式查看器: 滑动时间轴查看针身碰撞 ===
    print("\n启动交互式碰撞查看器 (关闭窗口继续)...")
    # 延迟导入, 因为它会切换 matplotlib 后端到 TkAgg
    from interactive_viewer import interactive_viewer
    interactive_viewer(result, cells, needles, sim_cfg)

    return result


def main():
    """主函数"""
    # 生成演示场景
    cells, needles = generate_demo_scenario(seed=42)

    print("\n细胞分布:")
    for c in cells:
        print(f"  C{c.id}: ({c.x:.0f}, {c.y:.0f}) -> 针{c.color}")

    print("\n针参数:")
    for n in needles:
        print(f"  针{n.id}: θ={np.degrees(n.theta_xy):.0f}°  "
              f"L={n.length:.0f}μm  "
              f"init=({n.init_pos[0]:.0f}, {n.init_pos[1]:.0f})  "
              f"cells={n.cells}")

    # 配置
    sim_cfg = SimConfig(
        t_find=60.0,
        t_wait=240.0,
        v=500.0,
        d_tip=80.0,
        d_body=150.0,
        delta_max=5000.0,
        delta_step=20.0,
    )

    ga_cfg = GAConfig(
        pop_size=80,
        n_gen=100,
        crossover_prob=0.9,
        mutation_prob=0.15,
        display_interval=10,
        seed=42,
    )

    # 运行优化
    res, callback = run_optimization(cells, needles, sim_cfg, ga_cfg)

    # 分析结果
    analyze_results(res, callback, cells, needles, sim_cfg)


if __name__ == "__main__":
    main()
