"""
传教士与野人渡河问题 - 状态空间搜索算法

问题重述：
- 河的左岸有 m 个传教士、m 个野人和一艘最多可乘 n 人的小船
- 约束条件：任何时刻（左岸、右岸或船上），传教士数量必须 >= 野人数量，
  否则野人会把传教士吃掉（特殊情况：无传教士时，任意数量野人都可以）
- 目标：找到一条方案使所有的野人和传教士安全渡到右岸

问题解答：
使用 BFS 求解：通过状态空间搜索找到最短的渡河方案
状态表示：(ml, wl, boat_pos) 其中：
  - ml: 左岸传教士数量
  - wl: 左岸野人数量
  - boat_pos: 船的位置 (0=左岸, 1=右岸)
"""

from collections import deque
from typing import List, Tuple, Optional



class MissionariesAndCannibals:
    """传教士与野人渡河问题求解器"""
    
    def __init__(self, m: int, n: int):
        # m: 传教士和野人的数量（各m个）、n: 小船最大容量
        self.m = m  # 传教士/野人数量
        self.n = n  # 船的容量
        self.visited = set()  # 已访问的状态集合
        self.parent = {}  # 状态转移记录（用于回溯路径）
    
    def is_safe(self, missionaries: int, cannibals: int) -> bool:
        # 无传教士时安全，或传教士 >= 野人时安全
        return missionaries == 0 or missionaries >= cannibals
    
    def is_valid_state(self, ml: int, wl: int, mr: int, wr: int) -> bool:
        """
        判断一个状态是否合法（所有位置都满足约束）

        ml, wl: 左岸的传教士和野人数量
        mr, wr: 右岸的传教士和野人数量
        """
        # 检查数量范围
        if ml < 0 or wl < 0 or mr < 0 or wr < 0:
            return False
        if ml + mr != self.m or wl + wr != self.m:
            return False
        return self.is_safe(ml, wl) and self.is_safe(mr, wr)
    
    def get_next_states(self, state: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        # 从当前状态出发，小船可以到达的所有合法状态
        ml, wl, boat_pos = state
        mr = self.m - ml
        wr = self.m - wl
        next_states = []
        
        if boat_pos == 0:  # 船在左岸，从左岸到右岸
            # 尝试所有可能的上船组合 (i个传教士，j个野人)
            for i in range(self.m + 1):
                for j in range(self.m + 1):
                    # 检查上船的人数
                    if 1 <= i + j <= self.n and i <= ml and j <= wl:
                        # 计算新的状态
                        new_ml = ml - i
                        new_wl = wl - j
                        new_mr = mr + i
                        new_wr = wr + j
                        
                        # 检查状态合法性
                        if self.is_valid_state(new_ml, new_wl, new_mr, new_wr):
                            next_states.append((new_ml, new_wl, 1))
        
        else:  # 船在右岸，从右岸到左岸
            # 尝试所有可能的上船组合 (i个传教士，j个野人)
            for i in range(self.m + 1):
                for j in range(self.m + 1):
                    # 检查上船的人数
                    if 1 <= i + j <= self.n and i <= mr and j <= wr:
                        # 计算新的状态
                        new_ml = ml + i
                        new_wl = wl + j
                        new_mr = mr - i
                        new_wr = wr - j

                        if self.is_valid_state(new_ml, new_wl, new_mr, new_wr):
                            next_states.append((new_ml, new_wl, 0))
        
        return next_states
    
    def search(self) -> Optional[List[Tuple[int, int, int]]]:
        # BFS
        # 初始状态：所有传教士和野人都在左岸，船也在左岸
        initial_state = (self.m, self.m, 0)
        # 目标状态：所有传教士和野人都在右岸，船也在右岸
        goal_state = (0, 0, 1)
        
        # 初始化
        queue = deque([initial_state])
        self.visited.add(initial_state)
        self.parent[initial_state] = None
        
        # BFS
        while queue:
            current_state = queue.popleft()

            if current_state == goal_state:
                return self.reconstruct_path(goal_state)

            for next_state in self.get_next_states(current_state):
                if next_state not in self.visited:
                    self.visited.add(next_state)
                    self.parent[next_state] = current_state
                    queue.append(next_state)
        
        return None  # 无解
    
    def reconstruct_path(self, goal_state: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        从目标状态回溯得到完整的路径
        """
        path = []
        current = goal_state
        
        while current is not None:
            path.append(current)
            current = self.parent[current]
        
        path.reverse()
        return path
    
    def format_state(self, state: Tuple[int, int, int]) -> str:
        """
        状态转换
        """
        ml, wl, boat_pos = state
        mr = self.m - ml
        wr = self.m - wl
        
        boat_location = "左岸" if boat_pos == 0 else "右岸"
        
        return f"左岸: {ml}传 {wl}野 | 右岸: {mr}传 {wr}野 | 船在: {boat_location}"
    
    def print_solution(self, path: List[Tuple[int, int, int]]) -> None:
        print("\n" + "="*60)
        print(f"传教士与野人渡河问题求解结果 (m={self.m}, 船容量={self.n})")
        print("="*60)
        
        print(f"\n初始状态：")
        print(f"  {self.format_state(path[0])}")
        
        print(f"\n渡河步骤：")
        for step in range(1, len(path)):
            prev_state = path[step - 1]
            curr_state = path[step]
            
            # 计算这一步上船的人数
            missionaries_moved = abs(curr_state[0] - prev_state[0])
            cannibals_moved = abs(curr_state[1] - prev_state[1])
            
            direction = "左→右" if prev_state[2] == 0 else "右→左"
            
            print(f"\n  步骤 {step}：")
            print(f"    方向：{direction}")
            print(f"    上船：{missionaries_moved}个传教士, {cannibals_moved}个野人")
            print(f"    {self.format_state(curr_state)}")
        
        print(f"\n目标达成！共用 {len(path) - 1} 步骤完成渡河")
        print("="*60 + "\n")


def main():

    print("="*60)
    
    # 设置问题参数
    m = 100  # 传教士和野人各m个
    n = 98  # 小船最多乘n人
    
    print(f"问题参数：")
    print(f"  初始左侧传教士数量: {m}个")
    print(f"  初始左侧野人数量: {m}个")
    print(f"  初始小船容量: {n}人")
    print(f"  初始状态: 所有人都在左岸")
    print(f"  目标状态: 所有人都到右岸")
    
    # 创建求解器并搜索
    solver = MissionariesAndCannibals(m, n)
    solution = solver.search()
    
    if solution:
        solver.print_solution(solution)
    else:
        print(f"\n不存在可能的解决方案")
        print(f"(m={m}, 船容量={n})")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
