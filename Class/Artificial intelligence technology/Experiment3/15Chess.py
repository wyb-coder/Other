import heapq
import time

# 目标状态 (0 代表空格)
# 1  2  3  4
# 5  6  7  8
# 9 10 11 12
# 13 14 15 0
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)

# 预计算目标状态下每个数字的位置 (x, y) 或 (row, col)
# {tile: (row, col)}
GOAL_POSITIONS = {tile: (i // 4, i % 4) for i, tile in enumerate(GOAL_STATE)}


class Node:
    """
    A* 算法的节点类
    """

    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state  # 状态 (tuple)
        self.parent = parent  # 父节点
        self.g = g  # g值: 从初始状态到当前状态的实际代价(步数)
        self.h = h  # h值: 启发式函数值 (曼哈顿距离)

    @property
    def f(self):
        """f值: f = g + h"""
        return self.g + self.h

    # heapq需要比较节点，定义 __lt__ (less than)
    def __lt__(self, other):
        # 当 f 值相同时，优先选择 g 值较小的 (即路径更短的)
        # 如果f值不同，选择f值小的
        if self.f == other.f:
            return self.g < other.g
        return self.f < other.f

    # 用于在 closed_set 中判断状态是否相同
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    # 用于 set 和 dict
    def __hash__(self):
        return hash(self.state)


def calculate_manhattan_distance(state):
    """
    计算当前状态的曼哈顿距离总和
    :param state: 状态元组 (16 elements)
    :return: 总曼哈顿距离 (int)
    """
    distance = 0
    for i, tile in enumerate(state):
        if tile == 0:  # 不计算空格
            continue

        current_pos = (i // 4, i % 4)
        goal_pos = GOAL_POSITIONS[tile]

        # |x1 - x2| + |y1 - y2|
        distance += abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])

    return distance


def get_neighbors(node):
    """
    获取当前节点的所有有效邻居节点 (移动一步)
    :param node: 当前 Node 对象
    :return: 邻居 Node 列表
    """
    neighbors = []
    state_list = list(node.state)

    # 找到空格 '0' 的位置
    blank_index = state_list.index(0)
    blank_row, blank_col = blank_index // 4, blank_index % 4

    # 定义可能的移动: (d_row, d_col, move_description)
    possible_moves = [
        (-1, 0),  # 上
        (1, 0),  # 下
        (0, -1),  # 左
        (0, 1)  # 右
    ]

    for dr, dc in possible_moves:
        new_row, new_col = blank_row + dr, blank_col + dc

        # 检查是否在 4x4 网格内
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_index = new_row * 4 + new_col

            # 交换的数字
            swapped_tile = state_list[new_index]

            # 创建新状态
            new_state_list = list(state_list)
            new_state_list[blank_index], new_state_list[new_index] = new_state_list[new_index], new_state_list[
                blank_index]

            new_state_tuple = tuple(new_state_list)

            # 创建邻居节点
            neighbor = Node(
                state=new_state_tuple,
                parent=node,
                g=node.g + 1,
                h=calculate_manhattan_distance(new_state_tuple)
            )

            # 存储交换信息，用于打印路径
            neighbor.move_info = (swapped_tile, "空格 (0)")

            neighbors.append(neighbor)

    return neighbors


def get_inversions(state):
    """计算逆序数 (忽略0)"""
    inversions = 0
    flat_state = [i for i in state if i != 0]
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            if flat_state[i] > flat_state[j]:
                inversions += 1
    return inversions


def is_solvable(state):
    """
    判断15数码问题是否有解
    规则: (逆序数) + (空格所在行号[0-indexed]) 的和必须为奇数
    (因为目标状态的 奇偶性 = 0 (逆序数) + 3 (行号) = 3 (奇数))
    """
    inversions = get_inversions(state)
    blank_index = state.index(0)
    blank_row = blank_index // 4

    parity = inversions + blank_row

    print(f"Checking solvability...")
    print(f"  Inversions: {inversions}")
    print(f"  Blank Row (from top, 0-indexed): {blank_row}")
    print(f"  Parity (Inversions + Blank Row): {parity}")

    return parity % 2 != 0


def print_state_grid(state):
    """以 4x4 网格形式打印状态"""
    for i in range(4):
        row = state[i * 4: i * 4 + 4]
        print("  " + " ".join(f"{tile:2}" for tile in row))


def print_solution_path(node):
    """
    从目标节点回溯并打印路径
    """
    path = []
    current = node
    while current:
        path.append(current)
        current = current.parent

    path.reverse()

    print("\n" + "=" * 30)
    print(f"Solution Found in {len(path) - 1} steps!")
    print("=" * 30 + "\n")

    for i, step_node in enumerate(path):
        print(f"Step {i}:")
        if i == 0:
            print("  # 初始状态")
        else:
            # move_info = (swapped_tile, "空格 (0)")
            tile, blank = step_node.move_info
            print(f"  交换 {tile} 与 {blank} 的位置 =>")

        print_state_grid(step_node.state)
        print("-" * 20)


def solve_puzzle(initial_state):
    """
    使用 A* 算法求解15数码问题
    """

    # 1. 检查可解性
    if not is_solvable(initial_state):
        print("This puzzle is NOT solvable.")
        return

    # 2. 初始化
    start_time = time.time()

    start_node = Node(
        state=tuple(initial_state),
        g=0,
        h=calculate_manhattan_distance(tuple(initial_state))
    )

    # open_list: 优先队列 (heapq)
    # 存储 (f, g, node)
    # f 和 g 用于排序, node 存储实际信息
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node.g, start_node))

    # closed_set: 字典 {state_tuple: g_value}
    # 存储已访问过的状态及其最小的 g 值
    # 这比存储Node对象更节省内存，并且可以处理重新访问节点的最优路径问题
    closed_set = {start_node.state: start_node.g}

    nodes_expanded = 0

    print("\nStarting A* Search...")

    # 3. A* 搜索循环
    while open_list:

        # 3.1. 从 open_list 取出 f 值最小的节点
        current_f, current_g, current_node = heapq.heappop(open_list)
        nodes_expanded += 1

        # 优化: 如果弹出的节点 g 值大于 closed_set 中的 g 值,
        # 说明我们已经找到了一条更短的路径到达此状态, 跳过这个旧节点
        if current_g > closed_set[current_node.state]:
            continue

        # 3.2. 检查是否为目标
        if current_node.state == GOAL_STATE:
            end_time = time.time()
            print(f"\nGoal Reached!")
            print(f"Time elapsed: {end_time - start_time:.4f} seconds")
            print(f"Nodes expanded: {nodes_expanded}")
            print_solution_path(current_node)
            return

        # 3.3. 扩展邻居
        for neighbor in get_neighbors(current_node):

            neighbor_state = neighbor.state
            new_g = neighbor.g

            # 3.4. 检查邻居是否在 closed_set 中
            # 如果不在, 或者找到了更短的路径 (new_g < old_g)
            if neighbor_state not in closed_set or new_g < closed_set[neighbor_state]:
                # 更新 closed_set (或添加新条目)
                closed_set[neighbor_state] = new_g

                # 将邻居加入 open_list
                heapq.heappush(open_list, (neighbor.f, new_g, neighbor))

    # 4. 未找到解
    print("No solution found (this should not happen for a solvable puzzle).")


# --- Main Execution ---
if __name__ == "__main__":
    # 一个中等难度的可解初始状态
    # Parity = 16 (inversions) + 3 (blank row) = 19 (Odd) -> Solvable
    initial_state = [
        5, 1, 2, 4,
        9, 6, 3, 8,
        13, 10, 7, 11,
        0, 14, 15, 12
    ]

    # 一个简单的可解状态 (2 步)
    # Parity = 2 (inversions) + 3 (blank row) = 5 (Odd) -> Solvable
    # initial_state = [
    #     1, 2, 3, 4,
    #     5, 6, 7, 8,
    #     9, 10, 11, 12,
    #     13, 0, 14, 15
    # ]

    # 一个不可解状态 (用于测试)
    # Parity = 1 (inversions) + 3 (blank row) = 4 (Even) -> Unsolvable
    # initial_state = [
    #     1, 2, 3, 4,
    #     5, 6, 7, 8,
    #     9, 10, 11, 12,
    #     13, 15, 14, 0
    # ]

    print("--- 15-Puzzle A* Solver ---")
    print("Initial State:")
    print_state_grid(initial_state)
    print("Goal State:")
    print_state_grid(GOAL_STATE)
    print("-" * 30)

    solve_puzzle(initial_state)