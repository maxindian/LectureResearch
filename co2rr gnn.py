"""
================================================================================
CO2RR GNN模型 - 纯PyTorch简化版
================================================================================
用于CO2电化学还原反应（CO2RR）中间体分析的图神经网络模型。
无需任何外部依赖，仅使用PyTorch实现。

架构说明：
- 基于SchNet（J. Chem. Phys. 2018）的连续滤波卷积
- 融合PaiNN（ICML 2021）的极化交互思想
- 适用于吸附体系的原子图表示学习

核心功能：
1. 从原子坐标和元素类型自动构建图结构
2. 通过消息传递网络学习原子环境的向量表示
3. 预测吸附能和产物选择性（CO/HCOOH/CH4/C2H4）
4. 分析反应路径，识别速控步骤
================================================================================
"""

# ================================================================================
# 第一部分：导入依赖
# ================================================================================
import torch                    # PyTorch深度学习框架核心库
import torch.nn as nn           # 神经网络模块（包含Layer、Module等基类）
import torch.nn.functional as F  # 神经网络函数（如softmax、relu等）
from torch.nn import Linear, Sequential, ReLU, LayerNorm, Dropout  # 常用层
import numpy as np              # 数值计算库，用于处理原子坐标等数组
import matplotlib
matplotlib.use('Agg')           # 使用非交互式后端（适合服务器/无显示器环境）
import matplotlib.pyplot as plt  # 绘图库，用于生成反应路径图

# ================================================================================
# 第二部分：原子数据字典
# ================================================================================
# ATOMIC_NUMBERS: 元素符号到原子序数的映射字典
# 包含电催化常见元素：H, C, N, O及过渡金属（Cu, Ag, Au, Pt, Pd等）
# 用于将元素符号（如'Cu'）转换为模型可处理的数字编码
ATOMIC_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84
}


# ================================================================================
# 第三部分：自定义Scatter函数（替代torch_geometric）
# ================================================================================
# 在图神经网络中，scatter操作用于将边上的消息聚合到节点上。
# torch_geometric提供了高效的scatter实现，但这里用纯PyTorch手动实现，
# 以便不依赖任何外部库。

def scatter_add(src, index, dim=0, dim_size=None):
    """
    自定义scatter_add: 按索引将源张量累加到目标张量
    
    原理：对于图中的消息传递，每个目标节点需要接收来自邻居节点的消息并求和。
         例如：agg[dst[i]] += messages[i]，其中dst[i]是第i条边的目标节点
    
    参数:
        src: 源张量，形状为[n_edges, feature_dim]，表示每条边携带的消息
        index: 索引张量，形状为[n_edges]，表示每条消息应该发送到哪个节点
        dim: 聚合的维度，默认为0（沿第0维聚合）
        dim_size: 输出张量在该维度的大小，即节点数量
    
    返回:
        聚合后的张量，形状为[n_nodes, feature_dim]
    
    示例:
        messages = [[1,2], [3,4], [5,6]]  # 3条边，每条边2维特征
        dst = [0, 1, 0]  # 第0条边到节点0，第1条边到节点1，第2条边到节点0
        result = scatter_add(messages, dst)  # 节点0收到1+5=6, 节点1收到3
        # result = [[6, 8], [3, 4]]
    """
    # 如果未指定输出大小，自动计算为最大索引+1
    if dim_size is None:
        dim_size = int(index.max()) + 1 if len(index) > 0 else 1
    
    # 创建输出张量，初始化为0
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # 使用index_add_将src按index累加到out中
    # index_add_是PyTorch的原地操作，效率较高
    out.index_add_(dim, index, src)
    return out


def scatter_mean(src, index, dim=0, dim_size=None):
    """
    自定义scatter_mean: 按索引将源张量求平均聚合到目标张量
    
    原理：与scatter_add相同，但最后除以每个节点接收到的消息数量，得到平均值。
         这可以防止节点度数（邻居数量）不同导致的偏差。
    
    参数:
        src: 源张量，形状为[n_edges]或[n_edges, feature_dim]
        index: 索引张量，形状为[n_edges]
        dim: 聚合维度，默认0
        dim_size: 输出大小
    
    返回:
        平均聚合后的张量，形状为[n_nodes]或[n_nodes, feature_dim]
    
    注意：与scatter_add的区别在于最后除以计数，实现均值聚合
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if len(index) > 0 else 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    
    # 第一步：求和聚合（与scatter_add相同）
    out_sum = torch.zeros(shape, dtype=src.dtype, device=src.device)
    out_sum.index_add_(dim, index, src)
    
    # 第二步：计算每个节点接收到的消息数量
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    ones = torch.ones(src.size(0), dtype=src.dtype, device=src.device)
    count.index_add_(0, index, ones)
    
    # 防止除以0（孤立节点至少计数为1）
    count = torch.clamp(count, min=1)
    
    # 扩展count的维度以便与out_sum进行广播除法
    # 例如 out_sum形状为[n_nodes, 4]，count形状为[n_nodes, 1]
    for _ in range(len(shape) - 1):
        count = count.unsqueeze(-1)
    
    # 返回均值 = 总和 / 计数
    return out_sum / count


# ================================================================================
# 第四部分：图构建器（GraphBuilder）
# ================================================================================
# 在GNN中，首先需要将原子结构（坐标+元素）转换为图表示。
# 图由节点（原子）和边（原子间距离<cutoff的连接）组成。

class GraphBuilder:
    """
    图构建器：从原子坐标和元素类型构建图结构
    
    核心思想：
    - 每个原子 = 图中的一个节点（node）
    - 距离小于cutoff的两个原子 = 图中的一条边（edge）
    - 节点特征 = 原子的物理化学属性（原子序数、电负性、半径等）
    - 边特征 = 原子间距离（用于后续的高斯径向基函数展开）
    
    这种表示方法的优势：
    - 保持了原子间的空间几何关系
    - 不依赖于特定的坐标系（平移/旋转不变性）
    - 天然的适用于不同大小的体系（不同数量的原子）
    """
    
    def __init__(self, cutoff=6.0):
        """
        初始化图构建器
        
        参数:
            cutoff: 截断半径（单位：埃Angstrom）
                   只有距离小于cutoff的原子对才会建立边连接
                   6.0埃是电催化计算中常用的经验值，平衡了精度和效率
        """
        self.cutoff = cutoff
    
    def build_graph(self, positions, elements, cell=None):
        """
        从原子坐标构建图
        
        参数:
            positions: [n_atoms, 3] numpy数组，原子三维坐标（单位：埃）
            elements: 元素符号列表，如['C', 'O', 'H', 'Cu']
            cell: [3, 3] 晶胞向量（可选，用于周期性边界条件PBC）
                  例如：surface slab计算中需要考虑周期性
        
        返回:
            dict: 包含以下键的字典
                - node_features: [n_atoms, 64] 节点特征矩阵
                - edge_index: [2, n_edges] 边索引（COO格式：[src, dst]）
                - edge_dist: [n_edges, 1] 边距离
                - edge_vec: [n_edges, 3] 边方向向量（单位向量）
                - positions: [n_atoms, 3] 原子坐标
                - atomic_numbers: [n_atoms] 原子序数
                - batch: [n_atoms] 批次索引（单结构全为0）
        
        图构建流程:
        1. 将numpy坐标转为PyTorch张量
        2. 将元素符号转为原子序数
        3. 双重循环遍历所有原子对，筛选距离<cutoff的作为边
        4. 提取每条边的距离和方向向量
        5. 构建节点特征（编码原子物理化学属性）
        """
        # 将numpy坐标转换为PyTorch张量，float32精度
        positions = torch.tensor(positions, dtype=torch.float32)
        n_atoms = len(positions)
        
        # 将元素符号（如'Cu'）转换为原子序数（如29）
        # 使用列表推导式遍历elements列表，查ATOMIC_NUMBERS字典
        # get(e, 1)表示如果元素不在字典中，默认使用H（原子序数1）
        atomic_numbers = torch.tensor([ATOMIC_NUMBERS.get(e, 1) for e in elements], dtype=torch.long)
        
        # ===================================================================
        # 构建邻居列表（Neighbor List）
        # ===================================================================
        # 这是图构建的核心步骤：找出哪些原子对应该连接成边
        # 采用双重循环遍历所有原子对，计算距离，筛选<cutoff的
        
        src_list, dst_list, dist_list, vec_list = [], [], [], []
        # src_list: 边的起点（源节点）索引
        # dst_list: 边的终点（目标节点）索引
        # dist_list: 边对应的原子间距离
        # vec_list: 边对应的单位方向向量
        
        for i in range(n_atoms):      # 遍历所有原子作为起点
            for j in range(n_atoms):  # 遍历所有原子作为终点
                if i == j:
                    continue          # 跳过自身（原子不与自己连接）
                
                # 计算从原子i指向原子j的向量
                vec = positions[j] - positions[i]
                # 计算欧几里得距离（向量范数）
                dist = torch.norm(vec)
                
                # 只保留距离在截断半径内的原子对
                if dist < self.cutoff:
                    src_list.append(i)           # 记录起点
                    dst_list.append(j)           # 记录终点
                    dist_list.append(dist.item()) # 记录距离值
                    # 记录方向向量（归一化为单位向量，避免距离信息重复）
                    # 加1e-8是为了防止除以0（理论上dist>0因为i!=j）
                    vec_list.append((vec / (dist + 1e-8)).tolist())
        
        # 将列表转为PyTorch张量
        # edge_index采用COO格式：[2, n_edges]，第0行是源节点，第1行是目标节点
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        # edge_dist: [n_edges, 1]，每条边对应一个距离值，unsqueeze添加维度
        edge_dist = torch.tensor(dist_list, dtype=torch.float32).unsqueeze(1)
        # edge_vec: [n_edges, 3]，每条边对应一个3维方向向量
        edge_vec = torch.tensor(vec_list, dtype=torch.float32)
        
        # ===================================================================
        # 构建节点特征（Node Features）
        # ===================================================================
        # 节点特征编码了每个原子的物理化学属性，帮助GNN区分不同元素
        # 这里使用4个物理化学描述符，归一化到[0, 1]范围：
        #   dim 0: 原子序数 / 100  - 标识元素种类
        #   dim 1: 电负性 / 4      - 反映原子吸引电子的能力
        #   dim 2: 共价半径 / 2    - 反映原子大小
        #   dim 3: 价电子数 / 8    - 反映成键能力
        # 其余60维预留为0，可通过projection层映射到hidden_dim
        
        node_features = torch.zeros(n_atoms, 64)  # 初始化全0特征
        for i, Z in enumerate(atomic_numbers):
            node_features[i, 0] = float(Z) / 100.0              # 归一化原子序数
            node_features[i, 1] = self._get_electronegativity(Z.item()) / 4.0  # 电负性
            node_features[i, 2] = self._get_radius(Z.item()) / 2.0            # 共价半径
            node_features[i, 3] = self._get_valence(Z.item()) / 8.0           # 价电子数
        
        # 返回图的字典表示
        return {
            'node_features': node_features,       # [n_atoms, 64] 节点特征
            'edge_index': edge_index,             # [2, n_edges] 边索引
            'edge_dist': edge_dist,               # [n_edges, 1] 边距离
            'edge_vec': edge_vec,                 # [n_edges, 3] 边方向向量
            'positions': positions,               # [n_atoms, 3] 原子坐标
            'atomic_numbers': atomic_numbers,     # [n_atoms] 原子序数
            'batch': torch.zeros(n_atoms, dtype=torch.long)  # 批次索引
        }
    
    # ===================================================================
    # 静态方法：查询原子的物理化学属性
    # ===================================================================
    # 以下方法均为静态方法（@staticmethod），不需要self参数
    # 它们提供了一套简化的原子属性查找表，用于构建节点特征
    
    @staticmethod
    def _get_electronegativity(Z):
        """
        获取原子的Pauling电负性
        
        电负性反映原子在化学键中吸引电子的能力，是区分不同元素的关键特征。
        例如：O(3.44) > C(2.55) > Cu(1.90) > H(2.20)
        
        参数:
            Z: 原子序数（如6表示碳）
        返回:
            Pauling电负性值（无量纲），未知元素默认返回1.5
        """
        pauling = {
            1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
            13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
            22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83,
            27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
            42: 2.16, 44: 2.20, 45: 2.28, 46: 2.20,
            47: 1.93, 78: 2.28, 79: 2.54
        }
        return pauling.get(Z, 1.5)  # 未知元素返回默认值1.5
    
    @staticmethod
    def _get_radius(Z):
        """
        获取原子的共价半径（单位：埃）
        
        共价半径反映原子的大小，影响原子间的空间排斥和轨道重叠。
        例如：Cu(1.12) > C(0.76) > O(0.66)
        
        参数:
            Z: 原子序数
        返回:
            共价半径（埃），未知元素默认返回1.0
        """
        radii = {
            1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
            13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
            22: 1.36, 23: 1.34, 24: 1.22, 25: 1.19, 26: 1.16,
            27: 1.11, 28: 1.10, 29: 1.12, 30: 1.18,
            42: 1.29, 44: 1.25, 45: 1.21, 46: 1.20,
            47: 1.28, 78: 1.21, 79: 1.21
        }
        return radii.get(Z, 1.0)
    
    @staticmethod
    def _get_valence(Z):
        """
        获取原子的价电子数
        
        价电子数决定原子的成键能力，是化学反应性的关键指标。
        按照元素周期表的周期规则计算：
        - 第1周期（H, He）: Z
        - 第2周期（Li-Ne）: Z - 2
        - 第3周期（Na-Ar）: Z - 10
        - 第4周期（K-Kr）: Z - 18
        - 第5周期及以后: Z - 36
        
        例如：C(4), O(6), Cu(11), H(1)
        
        参数:
            Z: 原子序数
        返回:
            价电子数
        """
        if Z <= 2:
            return Z
        elif Z <= 10:
            return Z - 2
        elif Z <= 18:
            return Z - 10
        elif Z <= 36:
            return Z - 18
        else:
            return Z - 36


# ================================================================================
# 第五部分：高斯基函数（GaussianBasis）
# ================================================================================
# 高斯基函数是SchNet架构的核心组件，用于将连续的距离值编码为向量表示。
# 
# 为什么需要这个？
# 在GNN中，边的信息（原子间距离）需要被转换为神经网络可以处理的特征向量。
# 直接使用原始距离（标量）信息不足，通过RBF展开可以将距离映射到高维空间，
# 使得模型可以更精细地区分不同距离的原子相互作用。

class GaussianBasis(nn.Module):
    """
    高斯径向基函数（Radial Basis Function, RBF）
    
    原理：
    将标量距离d展开为num_rbf维的向量，每个维度对应一个高斯函数：
        rbf_i(d) = exp(-gamma * (d - center_i)^2) * cutoff(d)
    
    其中：
    - center_i: 第i个高斯函数的中心位置（在[0, cutoff]区间均匀分布）
    - gamma: 高斯函数的宽度参数，控制衰减速度
    - cutoff(d): 余弦截断函数，保证距离接近cutoff时平滑衰减到0
    
    这种展开的优势：
    1. 平滑性：高斯函数是无穷阶可导的，保证力预测的平滑性
    2. 局部性：每个基函数只覆盖一小段距离范围
    3. 物理合理性：截断函数保证远距离相互作用平滑消失
    
    参考文献：SchNet (J. Chem. Phys. 2018)
    """
    
    def __init__(self, num_rbf=64, cutoff=6.0, gamma=10.0):
        """
        初始化RBF层
        
        参数:
            num_rbf: 高斯基函数的数量（RBF维度），默认64
                    越多则距离分辨率越高，但计算量也越大
            cutoff: 截断半径（单位：埃），默认6.0
            gamma: 高斯函数的宽度参数，默认10.0
                  越大则高斯峰越窄，距离分辨率越高
        """
        super().__init__()
        self.cutoff = cutoff
        self.gamma = gamma
        
        # 在[0, cutoff]区间内均匀生成num_rbf个中心点
        # 例如：num_rbf=64, cutoff=6.0 → centers = [0, 0.095, 0.19, ..., 6.0]
        centers = torch.linspace(0, cutoff, num_rbf)
        # register_buffer将centers注册为模型的持久化缓冲区（非可训练参数）
        # 这意味着centers会被保存到state_dict中，但不会参与梯度更新
        self.register_buffer('centers', centers)
    
    def forward(self, distances):
        """
        前向传播：将距离转换为RBF特征
        
        参数:
            distances: [n_edges, 1] 原子间距离张量
        
        返回:
            rbf: [n_edges, num_rbf] RBF展开后的特征矩阵
                 每一行对应一条边，每一列对应一个高斯基函数的响应
        
        计算步骤：
        1. 将距离展开为[n_edges, 1]以便广播
        2. 计算每个距离与每个中心点的高斯函数值
        3. 应用余弦截断函数
        4. 返回加权后的RBF特征
        """
        # d: [n_edges, 1] - 保持最后维度的unsqueeze以便广播
        d = distances.squeeze(-1).unsqueeze(-1)
        # c: [1, num_rbf] - 将centers扩展为可广播的形状
        c = self.centers.unsqueeze(0)
        
        # 计算高斯函数值: exp(-gamma * (d - c)^2)
        # 形状: [n_edges, num_rbf]
        rbf = torch.exp(-self.gamma * (d - c) ** 2)
        
        # ===================================================================
        # 余弦截断函数（Cosine Cutoff）
        # ===================================================================
        # 作用：保证距离接近cutoff时，RBF值平滑衰减到0
        # 公式：cf(d) = 0.5 * (cos(pi * d / cutoff) + 1)  for d < cutoff
        #              = 0                                   for d >= cutoff
        # 
        # 物理意义：避免截断半径处的非物理不连续性
        # 这种平滑截断对于能量守恒和力预测至关重要
        
        cf = 0.5 * (torch.cos(np.pi * d / self.cutoff) + 1.0)
        # 将d >= cutoff的位置置为0（硬截断）
        cf = cf * (d.squeeze(-1) < self.cutoff).float().unsqueeze(-1)
        
        # 返回RBF特征乘以截断权重
        # 距离接近cutoff的边，其RBF特征会被抑制
        return rbf * cf


# ================================================================================
# 第六部分：交互块（InteractionBlock）- SchNet核心
# ================================================================================
# 交互块是SchNet架构的消息传递层，实现了图神经网络的核心功能：
# 1. 消息生成（Message Generation）：基于边的特征生成消息
# 2. 消息聚合（Message Aggregation）：将消息汇总到目标节点
# 3. 节点更新（Node Update）：基于聚合消息更新节点表示

class InteractionBlock(nn.Module):
    """
    SchNet风格的交互块（消息传递层）
    
    这是SchNet架构的核心创新：连续滤波卷积（Continuous-Filter Convolution）。
    与传统GNN不同，SchNet的边权重不是标量，而是通过RBF->MLP生成的向量，
    即"连续滤波器"，可以捕获距离相关的复杂相互作用。
    
    消息传递流程：
    1. 对每条边的距离应用RBF展开 → [n_edges, num_rbf]
    2. 通过filter_net将RBF映射为滤波器 → [n_edges, hidden_dim]
    3. 滤波器与源节点特征逐元素相乘生成消息 → [n_edges, hidden_dim]
    4. 将消息聚合到目标节点（求和+平均归一化）→ [n_nodes, hidden_dim]
    5. 拼接原始特征和聚合特征，通过MLP更新 → [n_nodes, hidden_dim]
    6. 残差连接：新特征 = 旧特征 + 更新量
    
    参考文献：SchNet (J. Chem. Phys. 148, 241722, 2018)
    """
    
    def __init__(self, hidden_dim=128, num_rbf=64, num_filters=128):
        """
        初始化交互块
        
        参数:
            hidden_dim: 节点隐藏特征维度，默认128
            num_rbf: RBF展开维度，默认64
            num_filters: 滤波器中间层维度，默认128
        """
        super().__init__()
        
        # RBF层：将距离转换为高斯特征
        self.rbf = GaussianBasis(num_rbf=num_rbf)
        
        # ===================================================================
        # 滤波器网络（Filter Network）
        # ===================================================================
        # 这是SchNet的关键创新：一个基于距离的连续滤波器
        # 输入：RBF展开的距离特征 [n_edges, num_rbf]
        # 输出：边特定的滤波器 [n_edges, hidden_dim]
        # 
        # 作用：为每条边生成一个与距离相关的权重向量，
        #       不同的距离对应不同的滤波器，从而实现距离敏感的图卷积
        
        self.filter_net = Sequential(
            Linear(num_rbf, num_filters),   # [n_edges, num_rbf] → [n_edges, num_filters]
            ReLU(),                          # 非线性激活
            Linear(num_filters, num_filters), # 中间层
            ReLU(),
            Linear(num_filters, hidden_dim)   # 输出层 → [n_edges, hidden_dim]
        )
        
        # ===================================================================
        # 更新网络（Update Network）
        # ===================================================================
        # 输入：拼接的[原始特征, 聚合消息] [n_nodes, hidden_dim * 2]
        # 输出：节点更新量 [n_nodes, hidden_dim]
        # 
        # 作用：基于聚合的邻居信息，计算节点特征的更新量
        
        self.update_mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # 拼接特征 → 隐藏层
            ReLU(),
            Linear(hidden_dim, hidden_dim)        # 输出更新量
        )
        
        # 层归一化：稳定训练过程
        # 对每个节点的特征进行归一化，防止梯度爆炸/消失
        self.norm = LayerNorm(hidden_dim)
    
    def forward(self, h, edge_index, edge_dist):
        """
        前向传播：执行一次消息传递
        
        参数:
            h: [n_nodes, hidden_dim] 当前节点特征
            edge_index: [2, n_edges] 边索引 [src; dst]
            edge_dist: [n_edges, 1] 边距离
        
        返回:
            h_new: [n_nodes, hidden_dim] 更新后的节点特征（含残差连接）
        
        详细流程：
        """
        # 安全检查：如果没有边（空图），直接返回原特征
        if edge_dist.numel() == 0:
            return h
        
        # 解包边索引
        src, dst = edge_index  # src: [n_edges], dst: [n_edges]
        
        # ===================================================================
        # Step 1: RBF展开距离 → [n_edges, num_rbf]
        # ===================================================================
        rbf = self.rbf(edge_dist)
        
        # ===================================================================
        # Step 2: 滤波器网络生成边权重 → [n_edges, hidden_dim]
        # ===================================================================
        W = self.filter_net(rbf)
        
        # 安全检查：确保滤波器维度与节点特征维度匹配
        if W.size(-1) != h.size(-1):
            W = W[:, :h.size(-1)]
        
        # ===================================================================
        # Step 3: 消息生成（连续滤波卷积）
        # ===================================================================
        # 核心操作：messages = filter * node_features
        # 每条边的消息 = 该边的滤波器向量 * 源节点的特征向量（逐元素相乘）
        # 
        # h[src]: [n_edges, hidden_dim] - 通过索引获取每条边的源节点特征
        # W: [n_edges, hidden_dim] - 每条边的滤波器
        # messages: [n_edges, hidden_dim] - 逐元素乘法（Hadamard积）
        
        messages = W * h[src]
        
        # ===================================================================
        # Step 4: 消息聚合（Aggregation）
        # ===================================================================
        # 将所有指向同一目标节点的消息求和
        
        # 初始化聚合结果为全0 [n_nodes, hidden_dim]
        agg = torch.zeros_like(h)
        # index_add_: 将messages按dst索引累加到agg中
        # 例如：agg[dst[i]] += messages[i]
        agg.index_add_(0, dst, messages)
        
        # ===================================================================
        # Step 5: 平均归一化（Mean Normalization）
        # ===================================================================
        # 计算每个节点接收到的消息数量，防止度数高的节点有过大的聚合值
        
        cnt = torch.zeros(h.size(0), 1, device=h.device)  # [n_nodes, 1]
        cnt.index_add_(0, dst, torch.ones(messages.size(0), 1, device=h.device))
        cnt = torch.clamp(cnt, min=1)  # 防止除以0
        agg = agg / cnt  # 均值归一化
        
        # ===================================================================
        # Step 6: 节点更新（Update）
        # ===================================================================
        # 拼接原始特征和聚合消息 → [n_nodes, hidden_dim * 2]
        combined = torch.cat([h, agg], dim=-1)
        
        # 通过MLP计算更新量
        h_new = self.update_mlp(combined)
        
        # 层归一化
        h_new = self.norm(h_new)
        
        # ===================================================================
        # Step 7: 残差连接（Residual Connection）
        # ===================================================================
        # 残差连接是深度神经网络的重要技巧：
        # 新特征 = 旧特征 + 更新量
        # 这样可以保留原始信息，同时叠加新的学习信息，
        # 防止梯度消失，允许训练更深的网络
        
        return h + h_new


# ================================================================================
# 第七部分：CO2RR GNN主模型
# ================================================================================
# 这是整个模型的顶层封装，整合了图嵌入、多层交互、预测头。

class CO2RRGNN(nn.Module):
    """
    CO2RR图神经网络主模型
    
    整体架构：
    1. 原子嵌入层（Atom Embedding）：将原子序数映射为稠密向量
    2. 多层交互块（Interaction Blocks）：迭代更新节点表示
    3. 能量预测头（Energy Head）：从节点特征预测吸附能
    4. 选择性预测头（Selectivity Head）：预测4种产物的选择性概率
    
    输入：图表示（由GraphBuilder构建）
    输出：
        - energy: 体系总能量（吸附能）
        - selectivity: 4类产物选择性（CO, HCOOH, CH4, C2H4）
        - atom_features: 每个原子的学习特征（用于可解释性分析）
    
    参考文献：SchNet (2018), PaiNN (2021)
    """
    
    def __init__(self, hidden_dim=128, num_interactions=3, num_rbf=64, num_filters=128, node_feat_dim=64):
        """
        初始化CO2RR GNN模型
        
        参数:
            hidden_dim: 隐藏层维度，默认128
            num_interactions: 交互块层数，默认3
                             越多则感受野越大（捕获更远距离的相互作用）
            num_rbf: RBF基函数数量，默认64
            num_filters: 滤波器中间维度，默认128
            node_feat_dim: 输入节点特征维度，默认64（与GraphBuilder一致）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # ===================================================================
        # 原子嵌入层（Atom Embedding）
        # ===================================================================
        # 将离散的元素类型（原子序数）映射为连续的稠密向量
        # 类似NLP中的词嵌入：每个元素类型学习一个固定维度的向量表示
        # 
        # nn.Embedding(100, hidden_dim):
        #   - 100: 支持的原子序数范围（1-99，覆盖常见元素）
        #   - hidden_dim: 嵌入向量维度（默认128）
        
        self.atom_embed = nn.Embedding(100, hidden_dim)
        # Xavier均匀初始化：保持输入输出方差一致，有助于训练稳定性
        nn.init.xavier_uniform_(self.atom_embed.weight)
        
        # ===================================================================
        # 节点特征投影层（Node Feature Projection）
        # ===================================================================
        # GraphBuilder生成的节点特征维度（64）可能与hidden_dim不同
        # 这里使用线性层将64维节点特征投影到hidden_dim维
        # 以便与atom_embed的输出相加
        
        if node_feat_dim != hidden_dim:
            self.node_proj = Linear(node_feat_dim, hidden_dim)
        else:
            self.node_proj = None
        
        # ===================================================================
        # 多层交互块（Stacked Interaction Blocks）
        # ===================================================================
        # 堆叠多个交互块，实现多层消息传递
        # 每层交互块扩展一次感受野（即每个节点能感知更远处的原子）
        # 
        # 例如：3层交互块
        #   第1层：每个节点聚合直接邻居（1-hop）信息
        #   第2层：每个节点聚合邻居的邻居（2-hop）信息
        #   第3层：每个节点聚合3-hop内的信息
        # 
        # 这类似于CNN中的多层卷积，深层节点具有更大的感受野
        
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, num_rbf, num_filters)
            for _ in range(num_interactions)
        ])
        
        # ===================================================================
        # 能量预测头（Energy Prediction Head）
        # ===================================================================
        # 从每个原子的学习特征预测该原子对总能量的贡献
        # 体系总能量 = 所有原子能量贡献之和
        # 
        # 这种"原子分解"的方式具有物理意义：
        # - 可以分析哪些原子对能量贡献最大（可解释性）
        # - 满足能量可分解的物理直觉
        
        self.energy_head = Sequential(
            Linear(hidden_dim, hidden_dim // 2),  # 降维：128 → 64
            ReLU(),
            Linear(hidden_dim // 2, 1)             # 输出：每原子1个标量能量
        )
        
        # ===================================================================
        # 选择性预测头（Selectivity Prediction Head）
        # ===================================================================
        # 预测4种CO2RR产物的选择性：
        #   dim 0: CO     （2e-还原，C1路径）
        #   dim 1: HCOOH  （2e-还原，C1路径）
        #   dim 2: CH4    （8e-深度还原）
        #   dim 3: C2H4   （C-C偶联，C2+产物）
        # 
        # 选择性可以帮助理解催化剂倾向于生成哪种产物
        
        self.selectivity_head = Sequential(
            Linear(hidden_dim, hidden_dim // 2),  # 降维：128 → 64
            ReLU(),
            Linear(hidden_dim // 2, 4)             # 输出：每原子4维选择性logits
        )
    
    def forward(self, graph):
        """
        前向传播：从图表示预测能量和选择性
        
        参数:
            graph: 图字典（由GraphBuilder.build_graph生成），包含：
                - node_features: [n_atoms, 64] 节点物理化学特征
                - edge_index: [2, n_edges] 边索引
                - edge_dist: [n_edges, 1] 边距离
                - batch: [n_atoms] 批次索引
                - atomic_numbers: [n_atoms] 原子序数
        
        返回:
            dict: 包含以下键
                - energy: [batch_size, 1] 预测的总吸附能
                - selectivity: [batch_size, 4] 预测的选择性logits
                - atom_features: [n_atoms, hidden_dim] 学习到的原子特征
        
        计算流程：
        """
        # ===================================================================
        # Step 1: 节点特征初始化
        # ===================================================================
        # 通过嵌入层将原子序数映射为稠密向量
        # h: [n_atoms, hidden_dim]
        h = self.atom_embed(graph['atomic_numbers'])
        
        # 如果图中包含物理化学节点特征，则将其投影后加到嵌入向量上
        # 这样可以融合学习到的嵌入和先验的物理化学知识
        if 'node_features' in graph:
            node_feats = graph['node_features']  # [n_atoms, 64]
            if self.node_proj is not None:
                node_feats = self.node_proj(node_feats)  # [n_atoms, 64] → [n_atoms, hidden_dim]
            h = h + node_feats  # 残差式融合
        
        # ===================================================================
        # Step 2: 多层消息传递
        # ===================================================================
        # 依次通过每个交互块，迭代更新节点表示
        # 每次交互块都会聚合邻居信息，扩展感受野
        
        for interaction in self.interactions:
            h = interaction(h, graph['edge_index'], graph['edge_dist'])
        
        # ===================================================================
        # Step 3: 能量预测
        # ===================================================================
        # 每个原子预测一个标量能量贡献
        atom_E = self.energy_head(h).squeeze(-1)  # [n_atoms, 1] → [n_atoms]
        
        # 使用scatter_add将所有原子的能量贡献求和
        # 得到整个体系的预测能量
        energy = scatter_add(atom_E, graph['batch'])  # [batch_size]
        
        # ===================================================================
        # Step 4: 选择性预测
        # ===================================================================
        # 每个原子预测4维选择性logits
        atom_S = self.selectivity_head(h)  # [n_atoms, 4]
        
        # 对所有原子的选择性logits求平均
        # 得到整个体系的选择性预测
        batch = graph['batch']
        selectivity = torch.stack([
            scatter_mean(atom_S[:, i], batch)  # 对第i个输出维度分别聚合
            for i in range(atom_S.size(1))       # 遍历4个选择性类别
        ], dim=-1)  # 最后stack得到 [batch_size, 4]
        
        # ===================================================================
        # Step 5: 返回结果
        # ===================================================================
        return {
            'energy': energy.unsqueeze(-1),   # [batch_size, 1] 总能量
            'selectivity': selectivity,        # [batch_size, 4] 选择性
            'atom_features': h                 # [n_atoms, hidden_dim] 原子特征（可解释性）
        }


# ================================================================================
# 第八部分：演示数据 - CO2RR中间体结构
# ================================================================================
# 以下是CO2RR（CO路径）的5个关键中间体的演示结构
# 吸附在Cu(111)表面上的反应路径：CO2 → *COOH → *CO → *CHO → *CH2O → *CH3

def create_demo_structure(intermediate_name):
    """
    创建演示用的CO2RR中间体结构
    
    提供5个CO2RR关键中间体的原子坐标和元素类型：
    
    1. *COOH: 羧基中间体（CO2质子化后的第一个中间体）
       结构：HOOC-吸附在Cu表面
       这是CO2活化的关键步骤，*COOH的形成能垒决定了CO2还原的起始电位
    
    2. *CO: 一氧化碳吸附态
       结构：CO-吸附在Cu表面
       *CO是CO2RR的重要分支点：可以脱附生成CO产物，或继续还原
    
    3. *CHO: 甲酰基中间体
       结构：OHC-吸附在Cu表面
       *CO → *CHO是CO路径的速控步骤，需要较高的活化能
    
    4. *CH2O: 甲醛中间体
       结构：H2CO-吸附在Cu表面
       继续加氢还原的中间体
    
    5. *CH3: 甲基中间体
       结构：CH3-吸附在Cu表面
       进一步还原可生成CH4（甲烷）
    
    坐标说明：
    - Cu原子位于z≈7.5埃（模拟Cu(111)表面层）
    - 吸附物位于z>9埃（在表面上方）
    - x,y坐标模拟表面晶格位置
    
    参数:
        intermediate_name: 中间体名称（'*COOH', '*CO', '*CHO', '*CH2O', '*CH3'）
    
    返回:
        dict: {'elements': 元素列表, 'positions': [n_atoms, 3] 坐标数组}
    """
    
    structures = {
        # ===================================================================
        # *COOH: 羧基中间体
        # ===================================================================
        # CO2 + H+ + e- → *COOH（CO2活化的第一步，质子耦合电子转移）
        # 结构：C连接两个O（一个双键O，一个OH），整体通过C吸附在Cu上
        '*COOH': {
            'elements': ['H', 'O', 'C', 'O'] + ['Cu'] * 6,  # 10个原子
            'positions': np.array([
                [2.5, 2.5, 11.5],  # H: 远离表面的OH氢
                [2.5, 2.5, 10.5],  # O: OH中的氧
                [2.5, 3.2, 9.5],   # C: 中心碳，连接到表面
                [2.5, 2.5, 8.5],   # O: 双键氧
                # Cu(111)表面层（6个Cu原子）
                [1.5, 1.5, 7.5], [3.5, 1.5, 7.5],
                [1.5, 3.5, 7.5], [3.5, 3.5, 7.5],
                [2.5, 1.0, 7.5], [2.5, 4.0, 7.5]
            ])
        },
        
        # ===================================================================
        # *CO: 一氧化碳吸附态
        # ===================================================================
        # *COOH → *CO + H2O（脱水步骤，释放水分子）
        # CO可以脱附生成气相CO产物，或继续还原
        '*CO': {
            'elements': ['C', 'O'] + ['Cu'] * 6,  # 8个原子
            'positions': np.array([
                [2.5, 2.5, 9.5],   # C: 碳吸附在表面
                [2.5, 2.5, 10.5],  # O: 氧在碳上方（C≡O三键）
                # Cu(111)表面层
                [1.5, 1.5, 7.5], [3.5, 1.5, 7.5],
                [1.5, 3.5, 7.5], [3.5, 3.5, 7.5],
                [2.5, 1.0, 7.5], [2.5, 4.0, 7.5]
            ])
        },
        
        # ===================================================================
        # *CHO: 甲酰基中间体
        # ===================================================================
        # *CO + H+ + e- → *CHO（CO加氢步骤，速控步骤）
        # 这是CO路径的能垒最高步骤，决定了整体反应速率
        '*CHO': {
            'elements': ['C', 'O', 'H'] + ['Cu'] * 6,  # 9个原子
            'positions': np.array([
                [2.5, 2.5, 10.0],  # C: 中心碳
                [2.5, 2.5, 11.0],  # O: 双键氧
                [2.0, 2.0, 10.5],  # H: 加上的氢
                # Cu(111)表面层
                [1.5, 1.5, 7.5], [3.5, 1.5, 7.5],
                [1.5, 3.5, 7.5], [3.5, 3.5, 7.5],
                [2.5, 1.0, 7.5], [2.5, 4.0, 7.5]
            ])
        },
        
        # ===================================================================
        # *CH2O: 甲醛中间体
        # ===================================================================
        # *CHO + H+ + e- → *CH2O（继续加氢）
        '*CH2O': {
            'elements': ['C', 'O', 'H', 'H'] + ['Cu'] * 6,  # 10个原子
            'positions': np.array([
                [2.5, 2.5, 10.0],  # C: 中心碳
                [2.5, 2.5, 11.2],  # O: 双键氧
                [2.0, 2.0, 9.5],   # H: 第一个加上的氢
                [3.0, 2.5, 9.8],   # H: 第二个加上的氢
                # Cu(111)表面层
                [1.5, 1.5, 7.5], [3.5, 1.5, 7.5],
                [1.5, 3.5, 7.5], [3.5, 3.5, 7.5],
                [2.5, 1.0, 7.5], [2.5, 4.0, 7.5]
            ])
        },
        
        # ===================================================================
        # *CH3: 甲基中间体
        # ===================================================================
        # *CH2O + H+ + e- → *CH3 + *O（C-O键断裂）
        # *CH3进一步加氢生成CH4（甲烷）
        '*CH3': {
            'elements': ['C', 'H', 'H', 'H'] + ['Cu'] * 6,  # 10个原子
            'positions': np.array([
                [2.5, 2.5, 10.0],  # C: 中心碳
                [2.0, 2.0, 10.5],  # H: 第一个氢
                [3.0, 2.0, 9.8],   # H: 第二个氢
                [2.5, 3.2, 9.8],   # H: 第三个氢
                # Cu(111)表面层
                [1.5, 1.5, 7.5], [3.5, 1.5, 7.5],
                [1.5, 3.5, 7.5], [3.5, 3.5, 7.5],
                [2.5, 1.0, 7.5], [2.5, 4.0, 7.5]
            ])
        }
    }
    
    # 如果指定的中间体不存在，默认返回*COOH
    return structures.get(intermediate_name, structures['*COOH'])


# ================================================================================
# 第九部分：反应路径绘图函数
# ================================================================================

def plot_pathway(energies_dict, save_path='/mnt/agents/output/co2rr_pathway.png'):
    """
    绘制CO2RR反应路径图
    
    生成类似DFT计算文献中的能量剖面图：
    - 横轴：反应步骤
    - 纵轴：吸附能（eV）
    - 蓝色横线：每个中间体的能量水平
    - 红色箭头：反应步骤间的能量变化
    - 红色标注：每步的能垒值
    
    参数:
        energies_dict: dict {中间体名称: 吸附能(eV)}
        save_path: 图片保存路径
    
    返回:
        None（图片保存到文件）
    """
    names = list(energies_dict.keys())
    energies = [energies_dict[n] for n in names]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制每个中间体的能量水平（蓝色粗横线）
    for i, (name, E) in enumerate(zip(names, energies)):
        plt.plot([i-0.3, i+0.3], [E, E], 'b-', linewidth=3)
        # 标注中间体名称和能量值
        plt.text(i, E+0.05, f"{name}\n{E:.2f} eV", ha='center', fontsize=9)
    
    # 绘制反应步骤间的能量变化（红色箭头）
    for i in range(len(names)-1):
        plt.annotate(
            '',  # 无文本（文本单独标注）
            xy=(i+1, energies[i+1]),     # 箭头终点
            xytext=(i, energies[i]),     # 箭头起点
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )
        # 计算并标注能垒值
        dE = energies[i+1] - energies[i]
        mid_x = (i + i+1) / 2  # 中点x坐标
        mid_y = (energies[i] + energies[i+1]) / 2  # 中点y坐标
        plt.text(mid_x, mid_y+0.05, f"{dE:+.2f} eV", fontsize=8, color='red')
    
    # 图形格式设置
    plt.xlabel('Reaction Step', fontsize=12)       # 横轴标签
    plt.ylabel('Adsorption Energy (eV)', fontsize=12)  # 纵轴标签
    plt.title('CO2RR Reaction Pathway: CO Pathway', fontsize=14)  # 标题
    plt.grid(True, alpha=0.3)                       # 半透明网格
    plt.xlim(-0.5, len(names)-0.5)                  # x轴范围
    plt.tight_layout()                              # 自动调整布局
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 保存图片
    plt.close()
    print(f"  Pathway plot saved to: {save_path}")


# ================================================================================
# 第十部分：主程序（演示流程）
# ================================================================================

def main():
    """
    主函数：演示完整的CO2RR GNN分析流程
    
    流程：
    1. 创建GNN模型
    2. 创建图构建器
    3. 加载演示中间体结构
    4. 构建图并预测能量和选择性
    5. 分析反应路径（能垒、速控步）
    6. 分析选择性趋势
    7. 绘制反应路径图
    """
    
    # ===================================================================
    # 步骤1：创建GNN模型
    # ===================================================================
    print("=" * 70)
    print("   CO2RR GNN Model - Graph Neural Network for CO2 Reduction Analysis")
    print("=" * 70)
    
    print("\n[Step 1] Creating GNN Model...")
    # 实例化模型：hidden_dim=128, 3层交互块, 64个RBF基函数
    model = CO2RRGNN(hidden_dim=128, num_interactions=3, num_rbf=64, num_filters=128)
    # 计算模型参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created with {n_params:,} parameters")
    print(f"  Architecture: SchNet-style continuous-filter convolution")
    print(f"  Hidden dim: 128, Interactions: 3, RBF: 64")
    
    # ===================================================================
    # 步骤2：创建图构建器
    # ===================================================================
    print("\n[Step 2] Creating Graph Builder...")
    graph_builder = GraphBuilder(cutoff=6.0)
    # cutoff=6.0埃是电催化体系的常用值，平衡精度和计算效率
    
    # ===================================================================
    # 步骤3：定义分析的中间体序列
    # ===================================================================
    print("\n[Step 3] Creating Demo CO2RR Intermediates...")
    # CO路径的5个关键中间体，按反应顺序排列
    intermediates = ['*COOH', '*CO', '*CHO', '*CH2O', '*CH3']
    
    # ===================================================================
    # 步骤4：构建图并预测
    # ===================================================================
    print("\n[Step 4] Building Graphs and Predicting...")
    model.eval()  # 设置为评估模式（关闭Dropout等）
    results = {}
    
    # 关闭梯度计算（推理时不需要反向传播，节省内存）
    with torch.no_grad():
        for name in intermediates:
            # 获取中间体的结构数据
            struct = create_demo_structure(name)
            # 构建图表示
            graph = graph_builder.build_graph(struct['positions'], struct['elements'])
            
            # GNN前向传播
            pred = model(graph)
            
            # 提取预测结果
            energy = pred['energy'].item()  # 标量能量值
            # 对选择性logits应用softmax得到概率分布
            selectivity = F.softmax(pred['selectivity'], dim=-1).squeeze().numpy()
            
            # 保存结果
            results[name] = {
                'energy': energy,
                'selectivity': selectivity,
                'n_atoms': len(struct['elements'])
            }
            
            # 打印结果
            print(f"\n  {name} ({len(struct['elements'])} atoms):")
            print(f"    Predicted adsorption energy: {energy:.4f} eV")
            print(f"    Selectivity probabilities:")
            print(f"      CO     (C1 pathway): {selectivity[0]:.3f}")
            print(f"      HCOOH  (C1 pathway): {selectivity[1]:.3f}")
            print(f"      CH4    (deep reduction): {selectivity[2]:.3f}")
            print(f"      C2H4   (C-C coupling): {selectivity[3]:.3f}")
    
    # ===================================================================
    # 步骤5：反应路径分析
    # ===================================================================
    print("\n[Step 5] CO2RR Pathway Analysis...")
    energies_dict = {name: r['energy'] for name, r in results.items()}
    
    print("\n  CO2RR CO Pathway:")
    names = list(energies_dict.keys())
    for i in range(len(names)-1):
        dE = energies_dict[names[i+1]] - energies_dict[names[i]]
        print(f"    {names[i]} -> {names[i+1]}: {dE:+.4f} eV")
    
    # 计算总能量变化
    overall = energies_dict[names[-1]] - energies_dict[names[0]]
    print(f"    Overall energy change: {overall:.4f} eV")
    
    # 识别速控步骤（Rate-Limiting Step, RLS）
    # 速控步骤是能量上升最大的步骤，决定了整体反应速率
    barriers = [energies_dict[names[i+1]] - energies_dict[names[i]] for i in range(len(names)-1)]
    rls_idx = np.argmax(barriers)  # 找到最大能垒的索引
    print(f"    Rate-limiting step: {names[rls_idx]} -> {names[rls_idx+1]} ({barriers[rls_idx]:.4f} eV)")
    
    # ===================================================================
    # 步骤6：选择性分析
    # ===================================================================
    print("\n[Step 6] Selectivity Analysis...")
    for name, r in results.items():
        s = r['selectivity']
        products = ['CO', 'HCOOH', 'CH4', 'C2H4']
        favored = products[np.argmax(s)]  # 概率最高的产物
        print(f"  {name}: Most favorable product = {favored} (p={s[np.argmax(s)]:.3f})")
    
    # ===================================================================
    # 步骤7：绘制反应路径图
    # ===================================================================
    print("\n[Step 7] Generating Pathway Plot...")
    plot_pathway(energies_dict)
    
    # ===================================================================
    # 结束
    # ===================================================================
    print("\n" + "=" * 70)
    print("   Demo Complete!")
    print("=" * 70)
    
    # ===================================================================
    # 使用指南
    # ===================================================================
    print("\nUsage:")
    print("  1. Prepare structure: positions (Nx3), elements (list of symbols)")
    print("  2. Build graph: graph = graph_builder.build_graph(positions, elements)")
    print("  3. Predict: model = CO2RRGNN(); result = model(graph)")
    print("  4. Get energy: E_ads = result['energy'].item()")
    print("  5. Get selectivity: probs = F.softmax(result['selectivity'], dim=-1)")
    
    print("\nTraining:")
    print("  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)")
    print("  for epoch in range(200):")
    print("      pred = model(graph)")
    print("      loss = F.mse_loss(pred['energy'], target_energy)")
    print("      loss.backward(); optimizer.step()")


# ================================================================================
# 程序入口
# ================================================================================
# 当直接运行此脚本时（python co2rr_gnn_simple_annotated.py），执行main()函数
# 当作为模块导入时（import co2rr_gnn_simple_annotated），不自动执行
if __name__ == "__main__":
    main()
