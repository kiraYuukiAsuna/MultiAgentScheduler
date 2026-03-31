"""
批量运行Test/下的测试用例, 结果保存到Result/
每个测试用例的结果保存在 Result/test_XX/ 子目录下
"""

import json
import os
import sys
import time
import numpy as np

# 使用非交互式后端, 避免弹出窗口
import matplotlib
matplotlib.use('Agg')

from config import Cell, Needle, SimConfig, GAConfig
from main import run_optimization
from problem import decode_individual
from simulator import simulate
from visualization import (plot_convergence, plot_pareto_front,
                           plot_gantt, plot_2d_trajectory)

BASE_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(BASE_DIR, "Test")
RESULT_DIR = os.path.join(BASE_DIR, "Result")


def load_test_case(json_path):
    """从JSON文件加载测试用例, 返回 (cells, needles, sim_cfg, ga_cfg, meta)"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cells = [Cell(id=c["id"], color=c["color"], x=c["x"], y=c["y"])
             for c in data["cells"]]

    needles = []
    for nd in data["needles"]:
        needles.append(Needle(
            id=nd["id"],
            theta_xy=nd["theta_xy"],
            length=nd["length"],
            init_pos=np.array(nd["init_pos"]),
            cells=nd["cells"],
        ))

    sc = data["sim_config"]
    sim_cfg = SimConfig(
        t_find=sc["t_find"],
        t_wait=sc["t_wait"],
        v=sc["v"],
        d_tip=sc["d_tip"],
        d_body=sc["d_body"],
        delta_max=sc["delta_max"],
        delta_step=sc["delta_step"],
    )

    gc = data["ga_config"]
    ga_cfg = GAConfig(
        pop_size=gc["pop_size"],
        n_gen=gc["n_gen"],
        crossover_prob=gc["crossover_prob"],
        mutation_prob=gc["mutation_prob"],
        display_interval=gc["display_interval"],
        seed=gc["seed"],
    )

    meta = {
        "name": data["name"],
        "description": data["description"],
        "n_cells": data["n_cells"],
        "n_needles": data["n_needles"],
        "area_size": data["area_size"],
    }

    return cells, needles, sim_cfg, ga_cfg, meta


def run_single_test(json_path, result_dir):
    """运行单个测试用例并保存结果"""
    os.makedirs(result_dir, exist_ok=True)

    cells, needles, sim_cfg, ga_cfg, meta = load_test_case(json_path)

    print(f"\n{'='*60}")
    print(f"  测试: {meta['name']} - {meta['description']}")
    print(f"  细胞={meta['n_cells']}, 针={meta['n_needles']}, 区域={meta['area_size']}μm")
    print(f"  GA: pop={ga_cfg.pop_size}, gen={ga_cfg.n_gen}")
    print(f"{'='*60}")

    t_start = time.time()

    # 运行优化
    res, callback = run_optimization(cells, needles, sim_cfg, ga_cfg)

    t_opt = time.time() - t_start

    F = res.F
    X = res.X

    # 过滤死锁解
    feasible_mask = F[:, 0] < 1e5
    n_feasible = feasible_mask.sum()

    # 保存结果摘要
    summary = {
        "test_name": meta["name"],
        "description": meta["description"],
        "n_cells": meta["n_cells"],
        "n_needles": meta["n_needles"],
        "area_size": meta["area_size"],
        "ga_pop_size": ga_cfg.pop_size,
        "ga_n_gen": ga_cfg.n_gen,
        "optimization_time_sec": round(t_opt, 2),
        "pareto_solutions": int(len(F)),
        "feasible_solutions": int(n_feasible),
    }

    if n_feasible == 0:
        summary["status"] = "NO_FEASIBLE_SOLUTION"
        summary["best_makespan"] = None
        summary["best_total_dist"] = None
        summary["best_retract_count"] = None
        summary["best_blocked_time"] = None
        print(f"\n  [警告] 无可行解!")
    else:
        feasible_F = F[feasible_mask]
        feasible_X = X[feasible_mask]

        best_idx = np.argmin(feasible_F[:, 0])
        best_f = feasible_F[best_idx]
        best_x = feasible_X[best_idx]

        summary["status"] = "OK"
        summary["best_makespan"] = round(float(best_f[0]), 2)
        summary["best_total_dist"] = round(float(best_f[1]), 2)
        summary["best_retract_count"] = int(best_f[2])
        summary["best_blocked_time"] = round(float(best_f[3]), 2)

        print(f"\n  最优Makespan: {best_f[0]:.1f}s")
        print(f"  总距离: {best_f[1]:.1f}μm")
        print(f"  退让次数: {best_f[2]:.0f}")
        print(f"  阻塞时间: {best_f[3]:.1f}s")
        print(f"  优化耗时: {t_opt:.1f}s")

        # 解码最优解并仿真 (带事件记录)
        sequences, retractions = decode_individual(best_x, needles, len(cells))
        result = simulate(sequences, retractions, cells, needles, sim_cfg,
                          record_events=True)

        # 保存各针序列
        summary["best_sequences"] = [seq.tolist() if hasattr(seq, 'tolist') else list(seq)
                                      for seq in sequences]

        # 生成图表
        print(f"  生成图表...")

        try:
            plot_convergence(callback,
                             save_path=os.path.join(result_dir, 'convergence.png'))
        except Exception as e:
            print(f"    [跳过] 收敛曲线: {e}")

        try:
            plot_pareto_front(callback,
                              save_path=os.path.join(result_dir, 'pareto_front.png'))
        except Exception as e:
            print(f"    [跳过] Pareto前沿: {e}")

        try:
            plot_gantt(result, needles, title=f"{meta['name']} - 甘特图",
                       save_path=os.path.join(result_dir, 'gantt.png'))
        except Exception as e:
            print(f"    [跳过] 甘特图: {e}")

        try:
            plot_2d_trajectory(result, cells, needles,
                               title=f"{meta['name']} - 2D轨迹",
                               save_path=os.path.join(result_dir, 'trajectory.png'))
        except Exception as e:
            print(f"    [跳过] 2D轨迹: {e}")

    # 保存JSON摘要
    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  结果已保存到: {result_dir}/")
    return summary


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 查找所有测试用例
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.json')])

    if not test_files:
        print(f"未找到测试用例! 请先运行 generate_tests.py")
        return

    print(f"找到 {len(test_files)} 个测试用例")
    print(f"结果将保存到: {RESULT_DIR}/\n")

    all_summaries = []
    total_start = time.time()

    for test_file in test_files:
        json_path = os.path.join(TEST_DIR, test_file)
        test_name = test_file.replace('.json', '')
        result_dir = os.path.join(RESULT_DIR, test_name)

        # 跳过已完成的用例
        summary_path = os.path.join(result_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"\n  [跳过] {test_name} 已完成 (status={existing.get('status')})")
            all_summaries.append(existing)
            continue

        try:
            summary = run_single_test(json_path, result_dir)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n  [错误] {test_name}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({
                "test_name": test_name,
                "status": "ERROR",
                "error": str(e),
            })

    total_time = time.time() - total_start

    # 打印汇总表
    print(f"\n{'='*80}")
    print(f"  所有测试完成! 总耗时: {total_time:.1f}s")
    print(f"{'='*80}")
    print(f"{'测试':15s} {'细胞':>5s} {'针':>3s} {'状态':>8s} {'Makespan':>12s} "
          f"{'总距离':>12s} {'退让':>6s} {'耗时':>8s}")
    print("-" * 80)

    for s in all_summaries:
        name = s.get("test_name", "?")
        n_cells = s.get("n_cells", "?")
        n_needles = s.get("n_needles", "?")
        status = s.get("status", "?")
        makespan = f"{s['best_makespan']:.1f}s" if s.get("best_makespan") else "N/A"
        dist = f"{s['best_total_dist']:.0f}" if s.get("best_total_dist") else "N/A"
        retract = str(s.get("best_retract_count", "N/A"))
        opt_time = f"{s.get('optimization_time_sec', 0):.1f}s"

        print(f"{name:15s} {str(n_cells):>5s} {str(n_needles):>3s} {status:>8s} "
              f"{makespan:>12s} {dist:>12s} {retract:>6s} {opt_time:>8s}")

    # 保存总汇总
    overall_path = os.path.join(RESULT_DIR, "all_results.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存到: {overall_path}")


if __name__ == "__main__":
    main()
