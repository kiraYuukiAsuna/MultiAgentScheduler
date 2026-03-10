"""
多针细胞染料灌注调度 - 配置与数据结构
基于 MathModel8.tex v8.0
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Cell:
    """细胞信息"""
    id: int           # 细胞编号 (0-indexed)
    color: int        # 所属颜色/针 (0-indexed)
    x: float          # X坐标
    y: float          # Y坐标


@dataclass
class Needle:
    """针信息 (2D简化)"""
    id: int                     # 针编号 (0-indexed)
    theta_xy: float             # 水平安装角度 (rad)
    length: float               # 有效长度 (从针尖到针座)
    init_pos: np.ndarray        # 初始位置 [x, y]
    cells: list = field(default_factory=list)  # 分配的细胞ID列表

    @property
    def direction(self) -> np.ndarray:
        """单位方向向量 (针尖 -> 针座)"""
        return np.array([np.cos(self.theta_xy), np.sin(self.theta_xy)])

    @property
    def retract_dir(self) -> np.ndarray:
        """退让方向 (默认 = direction)"""
        return self.direction

    def tip_pos(self, p: np.ndarray) -> np.ndarray:
        """针尖位置"""
        return p.copy()

    def base_pos(self, p: np.ndarray) -> np.ndarray:
        """针座位置 (用于碰撞检测, 包含完整长度)"""
        return p + self.length * self.direction

    def draw_base_pos(self, p: np.ndarray, draw_length: float = 600.0) -> np.ndarray:
        """绘图用的针座位置 (只画一段可见长度)"""
        return p + draw_length * self.direction


@dataclass
class SimConfig:
    """仿真参数配置"""
    # 时间参数 (秒)
    t_find: float = 60.0        # Find阶段耗时
    t_wait: float = 240.0       # Wait阶段耗时
    v: float = 500.0            # 针移动速度 (微米/秒)

    # 碰撞检测参数 (微米)
    d_tip: float = 80.0         # 针尖安全距离
    d_body: float = 150.0       # 针身安全距离
    body_check_length: float = 200000.0   # 碰撞检测用的针身有效长度 (非全长)
    delta_max: float = 5000.0   # 退让距离上界
    delta_step: float = 20.0    # 退让搜索步长

    # 死锁惩罚
    h_penalty: float = 1e6

    # 仿真安全阀: 最大事件数
    max_events: int = 100000


@dataclass
class GAConfig:
    """NSGA-II 参数配置"""
    pop_size: int = 100         # 种群大小
    n_gen: int = 200            # 最大代数
    crossover_prob: float = 0.9 # 交叉率
    mutation_prob: float = 0.15 # 变异率
    seed: int = 42              # 随机种子
    display_interval: int = 10  # 每隔多少代展示一次中间结果
