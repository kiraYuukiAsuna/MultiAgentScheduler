"""
生成10个测试用例，细胞数从50到1000，保存为JSON到Test/文件夹
细胞分布模拟真实生物样本: 使用高斯聚类 + 均匀散布混合
"""

import json
import os
import numpy as np

TEST_DIR = os.path.join(os.path.dirname(__file__), "Test")
os.makedirs(TEST_DIR, exist_ok=True)

# 10个测试用例配置: (细胞数, 针数, 区域大小μm, 描述)
TEST_CONFIGS = [
    (50,   3,  2000,  "小规模-3针-稀疏分布"),
    (80,   3,  2500,  "小规模-3针-中等密度"),
    (120,  3,  3000,  "中小规模-3针-聚类分布"),
    (200,  4,  4000,  "中等规模-4针-均匀分布"),
    (300,  4,  5000,  "中等规模-4针-高密度聚类"),
    (400,  5,  6000,  "中大规模-5针-混合分布"),
    (500,  5,  7000,  "大规模-5针-多聚类"),
    (650,  5,  8000,  "大规模-5针-密集区域"),
    (800,  5,  9000,  "超大规模-5针-宽区域"),
    (1000, 5, 10000,  "最大规模-5针-全区域"),
]


def generate_clustered_cells(n_cells, area_size, rng, n_clusters=None):
    """
    生成聚类分布的细胞坐标
    70%的细胞在聚类中心附近 (高斯), 30%均匀散布
    细胞间最小距离约20μm (典型细胞直径~10-20μm)
    """
    if n_clusters is None:
        n_clusters = max(3, n_cells // 30)

    margin = area_size * 0.15
    coords = []

    # 聚类中心
    cluster_centers = rng.uniform(margin, area_size - margin, size=(n_clusters, 2))
    cluster_std = area_size * 0.08  # 聚类标准差

    # 70% 聚类细胞
    n_clustered = int(n_cells * 0.7)
    for _ in range(n_clustered):
        center = cluster_centers[rng.integers(n_clusters)]
        pos = rng.normal(center, cluster_std)
        pos = np.clip(pos, margin * 0.5, area_size - margin * 0.5)
        coords.append(pos)

    # 30% 均匀散布
    n_uniform = n_cells - n_clustered
    uniform_pos = rng.uniform(margin, area_size - margin, size=(n_uniform, 2))
    coords.extend(uniform_pos)

    coords = np.array(coords)

    # 确保最小间距 (简单推开过近的细胞)
    min_dist = 20.0  # μm
    for i in range(len(coords)):
        for j in range(i + 1, min(i + 20, len(coords))):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < min_dist and d > 0:
                direction = (coords[j] - coords[i]) / d
                push = (min_dist - d) / 2 + 1
                coords[j] += direction * push
                coords[i] -= direction * push

    return coords


def generate_needles(M, area_size):
    """
    生成M根针的配置
    针均匀分布在工作区域周围，角度间隔 360/M 度
    """
    center = np.array([area_size / 2, area_size / 2])
    radius = area_size * 0.75  # 针初始位置到中心距离
    needles = []

    for m in range(M):
        away_angle = np.radians(90 + m * (360 / M))
        init_pos = center - radius * np.array([np.cos(away_angle), np.sin(away_angle)])
        theta_xy = away_angle + np.pi

        needles.append({
            "id": m,
            "theta_xy": float(theta_xy),
            "length": 200000.0,
            "init_pos": [float(init_pos[0]), float(init_pos[1])],
        })

    return needles


def assign_cells_to_needles(coords, M, area_size):
    """
    将细胞分配给最近的针 (基于角度扇区)
    确保每根针至少分配一个细胞
    """
    center = np.array([area_size / 2, area_size / 2])
    N = len(coords)
    assignments = np.zeros(N, dtype=int)

    for i, pos in enumerate(coords):
        diff = pos - center
        angle = np.arctan2(diff[1], diff[0])
        # 映射到 [0, 2π)
        angle = angle % (2 * np.pi)
        sector = int(angle / (2 * np.pi / M)) % M
        assignments[i] = sector

    # 确保每根针至少有1个细胞
    for m in range(M):
        if np.sum(assignments == m) == 0:
            # 找最近的细胞重新分配
            away_angle = np.radians(90 + m * (360 / M))
            needle_dir = np.array([np.cos(away_angle), np.sin(away_angle)])
            best_i = 0
            best_dot = -np.inf
            for i, pos in enumerate(coords):
                diff = pos - center
                dot = np.dot(diff / (np.linalg.norm(diff) + 1e-10), needle_dir)
                if dot > best_dot:
                    best_dot = dot
                    best_i = i
            assignments[best_i] = m

    return assignments.tolist()


def generate_test_case(idx, n_cells, M, area_size, description, seed):
    """生成单个测试用例并保存为JSON"""
    rng = np.random.default_rng(seed)

    # 生成细胞坐标
    coords = generate_clustered_cells(n_cells, area_size, rng)

    # 分配细胞到针
    assignments = assign_cells_to_needles(coords, M, area_size)

    # 生成针配置
    needles = generate_needles(M, area_size)

    # 为每根针记录其细胞列表
    for m in range(M):
        needles[m]["cells"] = [i for i, a in enumerate(assignments) if a == m]

    # 构建细胞列表
    cells = []
    for i in range(n_cells):
        cells.append({
            "id": i,
            "color": assignments[i],
            "x": round(float(coords[i][0]), 2),
            "y": round(float(coords[i][1]), 2),
        })

    test_case = {
        "name": f"test_{idx+1:02d}",
        "description": description,
        "n_cells": n_cells,
        "n_needles": M,
        "area_size": area_size,
        "seed": seed,
        "cells": cells,
        "needles": needles,
        "sim_config": {
            "t_find": 60.0,
            "t_wait": 240.0,
            "v": 500.0,
            "d_tip": 80.0,
            "d_body": 150.0,
            "delta_max": 5000.0,
            "delta_step": 20.0,
        },
        "ga_config": {
            "pop_size": min(60, max(30, n_cells // 10)),
            "n_gen": min(100, max(30, 150 - n_cells // 10)),
            "crossover_prob": 0.9,
            "mutation_prob": 0.15,
            "display_interval": 10,
            "seed": 42,
        },
    }

    # 保存
    filename = os.path.join(TEST_DIR, f"test_{idx+1:02d}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(test_case, f, ensure_ascii=False, indent=2)

    # 打印统计
    needle_counts = [len(needles[m]["cells"]) for m in range(M)]
    print(f"  [{idx+1:2d}] {description}")
    print(f"      细胞={n_cells}, 针={M}, 区域={area_size}μm")
    print(f"      各针细胞数: {needle_counts}")
    print(f"      GA: pop={test_case['ga_config']['pop_size']}, gen={test_case['ga_config']['n_gen']}")
    print(f"      -> {filename}")

    return filename


def main():
    print("=" * 60)
    print("  生成测试用例")
    print("=" * 60)

    for idx, (n_cells, M, area_size, desc) in enumerate(TEST_CONFIGS):
        seed = 1000 + idx * 137  # 不同的随机种子
        generate_test_case(idx, n_cells, M, area_size, desc, seed)

    print(f"\n完成! 共生成 {len(TEST_CONFIGS)} 个测试用例到 {TEST_DIR}/")


if __name__ == "__main__":
    main()
