"""
多针细胞染料灌注调度 - pymoo NSGA-II 问题定义与自定义算子
基于 MathModel8.tex v8.0 Section 5, 9
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback

from config import Cell, Needle, SimConfig, GAConfig
from simulator import simulate


# ============================================================================
# 编码/解码工具
# ============================================================================

def encode_individual(sequences: list[list[int]], retractions: list[int],
                      needles: list[Needle], n_cells: int) -> np.ndarray:
    """
    将 (sequences, retractions) 编码为一维浮点数组
    格式: [seq_needle_0..., seq_needle_1..., ..., beta_0, beta_1, ..., beta_{N-1}]
    序列部分使用cell id (整数)，退让部分使用0/1
    """
    parts = []
    for m, needle in enumerate(needles):
        parts.extend(sequences[m])
    parts.extend(retractions)
    return np.array(parts, dtype=float)


def decode_individual(x: np.ndarray, needles: list[Needle],
                      n_cells: int) -> tuple[list[list[int]], list[int]]:
    """
    从一维数组解码出 (sequences, retractions)
    """
    idx = 0
    sequences = []
    for needle in needles:
        n_m = len(needle.cells)
        seq = [int(round(x[idx + k])) for k in range(n_m)]
        sequences.append(seq)
        idx += n_m

    retractions_raw = x[idx:idx + n_cells]
    retractions = [int(round(r)) for r in retractions_raw]
    return sequences, retractions


# ============================================================================
# pymoo Problem
# ============================================================================

class MultiNeedleProblem(Problem):
    def __init__(self, cells: list[Cell], needles: list[Needle],
                 sim_cfg: SimConfig):
        self.cells = cells
        self.needles = needles
        self.sim_cfg = sim_cfg
        self.n_cells = len(cells)
        self.M = len(needles)

        # 计算编码长度: sum(n_m) + N = N + N = 2N
        n_var = self.n_cells + self.n_cells  # 序列部分 + 退让部分
        super().__init__(
            n_var=n_var,
            n_obj=4,
            n_constr=0,
            xl=0.0,
            xu=1.0,  # 不实际使用xl/xu，由自定义算子处理
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """评估种群中每个个体"""
        F = np.zeros((X.shape[0], 4))
        for i in range(X.shape[0]):
            sequences, retractions = decode_individual(
                X[i], self.needles, self.n_cells)
            result = simulate(sequences, retractions,
                              self.cells, self.needles, self.sim_cfg,
                              record_events=False)
            F[i, 0] = result.f1_makespan
            F[i, 1] = result.f2_total_dist
            F[i, 2] = result.f3_retract_count
            F[i, 3] = result.f4_blocked_time
        out["F"] = F


# ============================================================================
# 自定义采样 (初始化)
# ============================================================================

class MultiNeedleSampling(Sampling):
    def __init__(self, needles: list[Needle], n_cells: int):
        super().__init__()
        self.needles = needles
        self.n_cells = n_cells

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        rng = np.random.default_rng()

        for i in range(n_samples):
            sequences = []
            for needle in self.needles:
                seq = list(needle.cells)
                rng.shuffle(seq)
                sequences.append(seq)

            retractions = rng.integers(0, 2, size=self.n_cells).tolist()
            X[i] = encode_individual(sequences, retractions,
                                     self.needles, self.n_cells)
        return X


# ============================================================================
# 自定义交叉 (PMX for 序列 + 均匀交叉 for 退让)
# ============================================================================

class MultiNeedleCrossover(Crossover):
    def __init__(self, needles: list[Needle], n_cells: int, prob: float = 0.9):
        super().__init__(n_parents=2, n_offsprings=2)
        self.needles = needles
        self.n_cells = n_cells
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        n_matings = X.shape[1]
        Y = np.full((2, n_matings, problem.n_var), np.nan)

        for k in range(n_matings):
            p1, p2 = X[0, k].copy(), X[1, k].copy()
            c1, c2 = p1.copy(), p2.copy()

            if np.random.random() < self.prob:
                seq1, beta1 = decode_individual(p1, self.needles, self.n_cells)
                seq2, beta2 = decode_individual(p2, self.needles, self.n_cells)

                # PMX交叉 for 每根针的序列
                cseq1, cseq2 = [], []
                for m in range(len(self.needles)):
                    s1, s2 = _pmx(seq1[m], seq2[m])
                    cseq1.append(s1)
                    cseq2.append(s2)

                # 均匀交叉 for 退让意向
                cbeta1, cbeta2 = list(beta1), list(beta2)
                for j in range(self.n_cells):
                    if np.random.random() < 0.5:
                        cbeta1[j], cbeta2[j] = cbeta2[j], cbeta1[j]

                c1 = encode_individual(cseq1, cbeta1, self.needles, self.n_cells)
                c2 = encode_individual(cseq2, cbeta2, self.needles, self.n_cells)

            Y[0, k] = c1
            Y[1, k] = c2

        return Y


def _pmx(p1: list[int], p2: list[int]) -> tuple[list[int], list[int]]:
    """Partially Mapped Crossover (PMX) for permutations"""
    n = len(p1)
    if n <= 1:
        return list(p1), list(p2)

    cx1, cx2 = list(p1), list(p2)
    # 选择两个切点
    pt1, pt2 = sorted(np.random.choice(n, 2, replace=False))

    # 映射表
    map1to2 = {}
    map2to1 = {}
    for i in range(pt1, pt2 + 1):
        cx1[i], cx2[i] = p2[i], p1[i]
        map1to2[p1[i]] = p2[i]
        map2to1[p2[i]] = p1[i]

    # 修复冲突
    def fix(child, mapping, other_parent):
        for i in list(range(0, pt1)) + list(range(pt2 + 1, n)):
            while child[i] in [child[j] for j in range(pt1, pt2 + 1)]:
                child[i] = mapping.get(child[i], child[i])
        # 最终验证/修复
        seen = set()
        missing = []
        all_vals = set(p1)
        for i in range(n):
            if child[i] in seen:
                child[i] = -1
            else:
                seen.add(child[i])
        missing = list(all_vals - seen)
        mi = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = missing[mi]
                mi += 1

    fix(cx1, map2to1, p1)
    fix(cx2, map1to2, p2)
    return cx1, cx2


# ============================================================================
# 自定义变异 (交换变异 for 序列 + 位翻转 for 退让)
# ============================================================================

class MultiNeedleMutation(Mutation):
    def __init__(self, needles: list[Needle], n_cells: int, prob: float = 0.15):
        super().__init__()
        self.needles = needles
        self.n_cells = n_cells
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i in range(X.shape[0]):
            if np.random.random() < self.prob:
                seq, beta = decode_individual(X[i], self.needles, self.n_cells)

                # 序列变异: 随机选一根针做交换变异
                m = np.random.randint(len(self.needles))
                if len(seq[m]) >= 2:
                    a, b = np.random.choice(len(seq[m]), 2, replace=False)
                    seq[m][a], seq[m][b] = seq[m][b], seq[m][a]

                # 退让变异: 随机翻转一个bit
                j = np.random.randint(self.n_cells)
                beta[j] = 1 - beta[j]

                Y[i] = encode_individual(seq, beta, self.needles, self.n_cells)
        return Y


# ============================================================================
# 自定义回调 (每代记录 + 交互展示)
# ============================================================================

class ProgressCallback(Callback):
    """记录每代的优化进展，用于交互式展示"""

    def __init__(self, display_interval: int = 10):
        super().__init__()
        self.display_interval = display_interval
        self.history_f1 = []  # 每代最优makespan
        self.history_f2 = []  # 每代最优total_dist
        self.history_f3 = []  # 每代最优retract_count
        self.history_f4 = []  # 每代最优blocked_time
        self.history_hv = []  # 可选: hypervolume
        self.all_F = []       # 每代的完整目标值矩阵
        self.gen_count = 0

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        self.all_F.append(F.copy())

        # 记录每个目标的最优值
        self.history_f1.append(F[:, 0].min())
        self.history_f2.append(F[:, 1].min())
        self.history_f3.append(F[:, 2].min())
        self.history_f4.append(F[:, 3].min())

        self.gen_count += 1

        # 打印进度
        if self.gen_count % self.display_interval == 0 or self.gen_count == 1:
            n_feasible = np.sum(F[:, 0] < 1e5)  # 非死锁个体
            best_f1 = F[:, 0].min()
            best_f2 = F[:, 1].min()
            print(f"  Gen {self.gen_count:4d} | "
                  f"可行: {n_feasible:3d}/{F.shape[0]} | "
                  f"最优Makespan: {best_f1:10.1f}s | "
                  f"最优距离: {best_f2:10.1f}μm | "
                  f"最优退让: {F[:, 2].min():.0f} | "
                  f"最优阻塞: {F[:, 3].min():.1f}s")
