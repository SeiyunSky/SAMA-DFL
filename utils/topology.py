"""
Network topology generation for decentralized FL
"""
import numpy as np
import networkx as nx


def generate_ring_topology(n):
    """
    生成环形拓扑（每个节点连接2个邻居）

    参数:
        n: int - 节点数

    返回:
        List[List[int]] - 邻接列表
    """
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        neighbors[i].append((i - 1) % n)  # 前一个
        neighbors[i].append((i + 1) % n)  # 后一个
    return neighbors


def generate_mesh_topology(n, degree=4):
    """
    生成部分网格拓扑（每个节点随机连接degree个邻居）

    参数:
        n: int - 节点数
        degree: int - 平均度数

    返回:
        List[List[int]] - 邻接列表
    """
    neighbors = [[] for _ in range(n)]

    for i in range(n):
        # 随机选择degree个不同的邻居
        candidates = list(range(n))
        candidates.remove(i)  # 不包含自身

        num_edges = min(degree, len(candidates))
        selected = np.random.choice(candidates, size=num_edges, replace=False)

        for j in selected:
            if j not in neighbors[i]:
                neighbors[i].append(j)
            if i not in neighbors[j]:
                neighbors[j].append(i)  # 无向图

    return neighbors


def generate_erdos_renyi_topology(n, p=0.3):
    """
    生成Erdos-Renyi随机图

    参数:
        n: int - 节点数
        p: float - 边存在概率

    返回:
        List[List[int]] - 邻接列表
    """
    G = nx.erdos_renyi_graph(n, p)
    neighbors = [list(G.neighbors(i)) for i in range(n)]
    return neighbors


def check_connectivity(neighbors, honest_nodes):
    """
    检查诚实节点子图是否连通

    参数:
        neighbors: List[List[int]] - 邻接列表
        honest_nodes: List[int] - 诚实节点ID列表

    返回:
        bool - 是否连通
    """
    honest_set = set(honest_nodes)

    # 构建诚实子图
    G = nx.Graph()
    G.add_nodes_from(honest_nodes)

    for i in honest_nodes:
        for j in neighbors[i]:
            if j in honest_set:
                G.add_edge(i, j)

    return nx.is_connected(G)


def compute_spectral_gap(neighbors, honest_nodes):
    """
    计算诚实节点子图的谱间隙γ

    谱间隙定义: γ = 1 - λ_2(W)
    其中λ_2是混合矩阵W的第二大特征值

    返回:
        float - 谱间隙
    """
    honest_set = set(honest_nodes)
    n_honest = len(honest_nodes)

    # 构建混合矩阵W
    W = np.zeros((n_honest, n_honest))
    node_to_idx = {node: idx for idx, node in enumerate(honest_nodes)}

    for idx, i in enumerate(honest_nodes):
        honest_neighbors = [j for j in neighbors[i] if j in honest_set]
        degree = len(honest_neighbors)

        if degree > 0:
            for j in honest_neighbors:
                j_idx = node_to_idx[j]
                W[idx, j_idx] = 1.0 / (degree + 1)
            W[idx, idx] = 1.0 / (degree + 1)  # 自环权重，保证行和=1

    # 计算特征值
    eigenvalues = np.linalg.eigvals(W)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # 降序

    lambda_1 = eigenvalues[0]  # 应该接近1
    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0

    gamma = 1 - lambda_2
    return gamma


if __name__ == "__main__":
    # 测试
    print("Testing topology generation...")

    n = 20
    honest = list(range(16))  # 前16个是诚实节点
    byzantine = list(range(16, 20))  # 后4个是拜占庭节点

    # Ring
    ring = generate_ring_topology(n)
    print(f"Ring: Node 0's neighbors = {ring[0]}")
    print(f"Ring: Honest subgraph connected = {check_connectivity(ring, honest)}")
    gamma_ring = compute_spectral_gap(ring, honest)
    print(f"Ring: Spectral gap γ = {gamma_ring:.4f}")

    # Mesh
    mesh = generate_mesh_topology(n, degree=6)
    print(f"\nMesh: Node 0's neighbors = {mesh[0]}")
    print(f"Mesh: Honest subgraph connected = {check_connectivity(mesh, honest)}")
    gamma_mesh = compute_spectral_gap(mesh, honest)
    print(f"Mesh: Spectral gap γ = {gamma_mesh:.4f}")
