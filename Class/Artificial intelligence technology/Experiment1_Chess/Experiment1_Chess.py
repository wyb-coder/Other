import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import random
import pygame

# 环境参数，辅助画布定位、边界判断
CANVAS_WEIGHT = 600                                 # 画布长
CANVAS_HEIGHT = 600
BOARD_SIZE = 16                                     # 棋盘网格数目
GRID_LEFT_RATIO, GRID_TOP_RATIO = 0.2, 0.2          # 坐上定位点比例
GRID_RIGHT_RATIO, GRID_BOTTOM_RATIO = 0.8, 0.8
BACKGROUND_IMAGE = "images/Chess1.png"                       # 背景图
BLACK_IMAGE = "images/Black_Chessman.png"
WHITE_IMAGE = "images/White_Chessman.png"
BACKGROUND_MUSIC = "music/Chess_Bgm.mp3"


class GomokuGUI:
    def __init__(self, root, game_mode="PVP", ai_player=None, ai_level=1):
        self.root = root
        self.root.title("人工智能技术_实验一_五子棋")

        # 记录属性值
        self.game_mode = game_mode                  # PVP or PVE
        self.ai_player = ai_player                  # AI 执棋黑/白
        self.ai_level = ai_level                    # 困难度 1——5
        self.winner = 0                             # 辅助战况判断是否胜利标志

        # 初始化画布Canvas
        self.canvas = tk.Canvas(root, width=CANVAS_WEIGHT, height=CANVAS_HEIGHT)
        self.canvas.pack()

        # 加载背景图
        self.background_image = self.load_and_fit_background(BACKGROUND_IMAGE, CANVAS_WEIGHT, CANVAS_HEIGHT)
        self.canvas.create_image(0, 0, anchor="nw", image=self.background_image)

        # 绘制棋盘网格，计算棋盘的上下边界，给背景图四周美化留空间
        self.grid_left   = int(CANVAS_WEIGHT * GRID_LEFT_RATIO)
        self.grid_top    = int(CANVAS_HEIGHT * GRID_TOP_RATIO)
        self.grid_right  = int(CANVAS_WEIGHT * GRID_RIGHT_RATIO)
        self.grid_bottom = int(CANVAS_HEIGHT * GRID_BOTTOM_RATIO)
        self.grid_cell_size = (self.grid_right - self.grid_left) / (BOARD_SIZE - 1)      # 单个网格大小
        # 加载棋子图片
        stone_size = int(self.grid_cell_size * 1.2)
        self.black_image = ImageTk.PhotoImage(Image.open(BLACK_IMAGE).resize((stone_size, stone_size), Image.Resampling.LANCZOS))
        self.white_image = ImageTk.PhotoImage(Image.open(WHITE_IMAGE).resize((stone_size, stone_size), Image.Resampling.LANCZOS))
        self.draw_grid()

        # 绘制状态栏（部分战况）
        self.status_var = tk.StringVar(value="黑棋回合")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x")

        # 绘制菜单栏
        menubar = tk.Menu(root)
        game_menu = tk.Menu(menubar, tearoff=0)
        game_menu.add_command(label="重新开始", command=self.reset_game)
        game_menu.add_separator()
        game_menu.add_command(label="退出游戏", command=root.quit)
        menubar.add_cascade(label="游戏", menu=game_menu)
        root.config(menu=menubar)

        # 棋局状态 + 具体战况
        self.board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.stone_refs = []                        # 保存棋子图片的引用，以防被垃圾回收
        # 新增战况菜单
        status_menu = tk.Menu(menubar, tearoff=0)
        status_menu.add_command(label="查看战况", command=self.show_status)
        menubar.add_cascade(label="战况", menu=status_menu)

        # 鼠标点击事件绑定
        self.canvas.bind("<Button-1>", self.on_click)

        # 设定AI执黑棋时先手下一个子
        if self.game_mode == "PVE" and self.ai_player == 1:
            self.ai_move()

        # 背景音乐
        pygame.mixer.init()
        pygame.mixer.music.load(BACKGROUND_MUSIC)
        pygame.mixer.music.play(-1)             # 循环播放BGM


    # 战况统计：计算黑/白棋数目，定义显示框
    def show_status(self):
        # self.board存储棋盘状态
        black_chess_count = sum(row.count(1) for row in self.board)
        white_chess_count = sum(row.count(2) for row in self.board)

        if self.game_over:
            #self.winner记录胜者
            winner_text = "黑棋" if self.winner == 1 else "白棋"
            msg = f"当前战况：\n{winner_text}获胜！"
        else:
            turn_text = "黑棋" if self.current_player == 1 else "白棋"
            msg = (
                f"当前战况：\n"
                f"黑棋：{black_chess_count} 子\n"
                f"白棋：{white_chess_count} 子\n"
                f"轮到：{turn_text}"
            )
        messagebox.showinfo("战况", msg)

    # 拉伸图片以适合画布
    def load_and_fit_background(self, path, target_weight, target_height):
        img = Image.open(path)
        img_resized = img.resize((target_weight, target_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img_resized)

    # 绘制网格
    def draw_grid(self):
        line_color = "#3A2F1E"
        for i in range(BOARD_SIZE):
            y = self.grid_top + i * self.grid_cell_size
            # 画横线
            self.canvas.create_line(self.grid_left, y, self.grid_right, y, fill=line_color, width=2)
            x = self.grid_left + i * self.grid_cell_size
            # 画竖线
            self.canvas.create_line(x, self.grid_top, x, self.grid_bottom, fill=line_color, width=2)

    # 鼠标点击事件
    def on_click(self, event):
        if self.game_over:
            return
        if self.game_mode == "PVE" and self.current_player == self.ai_player:
            return  # AI的回合，阻止玩家操作

        # 通过定位计算网格中的行列
        row, col = self.pixel_to_grid(event.x, event.y)
        if row is None or col is None:
            return
        if self.board[row][col] != 0:
            return

        # 玩家落子
        self.place_stone(row, col, self.current_player)

    # AI下棋算法
    def ai_move(self):
        if self.ai_level == 1:
            self.ai_move1()                     # 随机下法，决策完全随机
        elif self.ai_level == 2:
            self.ai_move2()                     # 基于规则的启发式算法，设定阻止四连、五连等
        elif self.ai_level == 3:
            self.ai_move3()                     # 极大化极小值搜尋 + αβ剪枝 + 简单评估函数
        elif self.ai_level == 4:
            self.ai_move4()                     # 极大化极小值搜尋 + αβ剪枝 + 复杂
        elif self.ai_level == 5:
            self.ai_move5()                     # 多重優化的高級搜尋演算法

    def ai_move1(self):
        # 纯随机落子，做少许优化：仅在已有棋子周围选择。
        candidate_positions = self.get_candidate_moves()
        if not candidate_positions:
            return  # 下满了
        row, col = random.choice(candidate_positions)
        self.place_stone(row, col, self.current_player)

    def ai_move2(self):
        # 一般难度，死板的匹配规则：优先进攻（造四连/赢棋），其次选择防守，最后随机
        ai_player = self.current_player
        opponent_player = 1 if ai_player == 2 else 2

        # 仅在已有棋子周围选择。
        candidate_positions = self.get_candidate_moves()

        for row, col in candidate_positions:
            self.board[row][col] = ai_player
            # 1. AI 自己能赢（五连子）
            if self.check_win(row, col):
                self.board[row][col] = 0
                self.place_stone(row, col, ai_player)
                return
            # 2. 防守：堵对方五连
            if self.check_win(row, col):
                self.board[row][col] = 0
                self.place_stone(row, col, ai_player)
                return
            # 3. AI 能造四连
            if self.count_max_line(row, col, ai_player) >= 4:
                self.board[row][col] = 0
                self.place_stone(row, col, ai_player)
                return
            # 4. 防守：堵对方四连
            self.board[row][col] = 0
            if self.count_max_line(row, col, opponent_player) >= 4:
                self.board[row][col] = 0
                self.place_stone(row, col, ai_player)
                return

        # 随机下
        self.ai_move1()

    # 辅助计算最长连子数
    def count_max_line(self, row, col, player_value):
        max_connected_count = 1
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row_step, col_step in directions:
            connected_count = 1
            next_row, next_col = row + row_step, col + col_step
            while 0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE and self.board[next_row][next_col] == player_value:
                connected_count += 1
                next_row += row_step
                next_col += col_step

            prev_row, prev_col = row - row_step, col - col_step
            while 0 <= prev_row < BOARD_SIZE and 0 <= prev_col < BOARD_SIZE and self.board[prev_row][prev_col] == player_value:
                connected_count += 1
                prev_row -= row_step
                prev_col -= col_step
            max_connected_count = max(max_connected_count, connected_count)
        return max_connected_count

    def ai_move3(self):
        # 极大化极小值搜尋 + αβ剪枝 + 简单评估函数
        self.ai_search(depth=2, eval_func=self.evaluate_board)

    # 获取候选点，只在已有棋子的周围下
    def get_candidate_moves(self):
        candidate_moves = set()

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] != 0:
                    # 在已有棋子周围2格范围内寻找空位
                    for row_offset in range(-2, 3):
                        for col_offset in range(-2, 3):
                            neighbor_row = row + row_offset
                            neighbor_col = col + col_offset
                            if 0 <= neighbor_row < BOARD_SIZE and 0 <= neighbor_col < BOARD_SIZE and self.board[neighbor_row][neighbor_col] == 0:
                                candidate_moves.add((neighbor_row, neighbor_col))
        # 如果棋盘完全为空，默认返回中心点
        if not candidate_moves:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        return list(candidate_moves)

    # 判断游戏终局
    def is_terminal(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] != 0 and self.check_win(r, c):
                    self.winner = self.board[r][c]
                    return True
        self.winner = 0
        return False

    # 简单评估函数
    def evaluate_board(self):
        score = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] != 0:
                    player = self.board[row][col]
                    line = self.count_max_line(row, col, player)
                    if player == self.current_player:
                        if line >= 4:
                            score += 1000
                        elif line == 3:
                            score += 100
                        elif line == 2:
                            score += 10
                    else:
                        if line >= 4:
                            score -= 1000
                        elif line == 3:
                            score -= 100
                        elif line == 2:
                            score -= 10
        return score

    # 困难难度：极大极小搜索 + αβ剪枝 + 优化评估函数
    def ai_move4(self):
        self.ai_search(depth=2, eval_func=self.evaluate_advanced)

    def ai_search(self, depth, eval_func):
        best_score = float("-inf")
        best_move = None
        empty_positions = self.get_candidate_moves()

        for row, col in empty_positions:
            self.board[row][col] = self.current_player
            score = self.minimax_with_eval(depth - 1, False, float("-inf"), float("inf"), eval_func)
            self.board[row][col] = 0
            if score > best_score:
                best_score = score
                best_move = (row, col)

        if best_move:
            self.place_stone(best_move[0], best_move[1], self.current_player)
        else:
            self.ai_move1()

    # 博弈决策树
    def minimax_with_eval(self, depth, is_maximizing, alpha, beta, eval_func):
        # 如果胜利/对方失败，则直接返回大
        if self.is_terminal():
            if self.winner == self.current_player:
                return 100000000
            elif self.winner != 0:
                return -100000000
            else:
                return 0
        if depth == 0:
            return eval_func()
        empty_positions = self.get_candidate_moves()

        # is_maximizing True我方下棋，尽量让分数最大/False对手下棋，尽量让分数最小
        if is_maximizing:
            max_eval = float("-inf")
            for row, col in empty_positions:
                self.board[row][col] = self.current_player
                eval = self.minimax_with_eval(depth - 1, False, alpha, beta, eval_func)  # ✅ 加上 eval_func
                self.board[row][col] = 0
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            opponent = 1 if self.current_player == 2 else 2
            for row, col in empty_positions:
                self.board[row][col] = opponent
                eval = self.minimax_with_eval(depth - 1, True, alpha, beta, eval_func)  # ✅ 加上 eval_func
                self.board[row][col] = 0
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    # 优化评估函数：区分活三、眠三、活四等棋型
    def evaluate_advanced(self):
        score = 0
        patterns = {
            "five": 100000,
            "live_four": 10000,
            "sleep_four": 5000,
            "live_three": 1000,
            "sleep_three": 200,
            "live_two": 50,
        }

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] != 0:
                    player = self.board[row][col]
                    val = self.evaluate_point(row, col, player, patterns)
                    if player == self.current_player:
                        score += val
                    else:
                        score -= val
        return score

    # 评估某个点的棋型价值
    def evaluate_point(self, row, col, player, patterns):
        total = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            line = self.get_line(row, col, dr, dc, player)
            total += self.match_pattern(line, patterns)
        return total

    # 获取某点在某方向上的连续棋子和空位情况
    def get_line(self, row, col, row_step, col_step, player_value):
        line_sequence = [player_value]

        # 正方向延伸
        next_row, next_col = row + row_step, col + col_step
        while 0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE and self.board[next_row][
            next_col] == player_value:
            line_sequence.append(player_value)
            next_row += row_step
            next_col += col_step
        if 0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE and self.board[next_row][next_col] == 0:
            line_sequence.append(0)

        # 反方向延伸
        prev_row, prev_col = row - row_step, col - col_step
        while 0 <= prev_row < BOARD_SIZE and 0 <= prev_col < BOARD_SIZE and self.board[prev_row][
            prev_col] == player_value:
            line_sequence.insert(0, player_value)
            prev_row -= row_step
            prev_col -= col_step
        if 0 <= prev_row < BOARD_SIZE and 0 <= prev_col < BOARD_SIZE and self.board[prev_row][prev_col] == 0:
            line_sequence.insert(0, 0)

        return line_sequence

    # 打分函数
    def match_pattern(self, line, patterns):
        s = "".join(str(x) for x in line)
        if "11111" in s:
            return patterns["five"]
        if "011110" in s:
            return patterns["live_four"]
        if "11110" in s or "01111" in s:
            return patterns["sleep_four"]
        if "01110" in s:
            return patterns["live_three"]
        if "0111" in s or "1110" in s:
            return patterns["sleep_three"]
        if "0110" in s:
            return patterns["live_two"]
        return 0

    # 地狱难度：αβ剪枝 + 置换表缓存 + 威胁空间搜索（活三/冲四/必胜与必堵优先）+ 中心偏好 + 候选点排序
    def ai_move5(self):
        player = self.current_player
        opponent = 1 if player == 2 else 2

        # 初始化置换表缓存
        if not hasattr(self, "_transposition_table"):
            self._transposition_table = {}
        empty_positions = [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if
                           self.board[row][col] == 0]

        # 1. 直接必胜：如果当前玩家能立即获胜，直接下子
        for row, col in empty_positions:
            self.board[row][col] = player
            if self.check_win(row, col):
                self.board[row][col] = 0
                return self.place_stone(row, col, player)
            self.board[row][col] = 0

        # 2. 必堵：如果对手能立即获胜，必须堵住
        for row, col in empty_positions:
            self.board[row][col] = opponent
            if self.check_win(row, col):
                self.board[row][col] = 0
                return self.place_stone(row, col, player)
            self.board[row][col] = 0

        # 3. 威胁空间候选：优先考虑威胁点，否则使用排序后的候选点
        threat_moves = self._collect_threat_moves(player)
        candidate_moves = threat_moves if threat_moves else self._ordered_candidates(player)

        # 4. αβ搜索（固定深度=2）
        best_score = float("-inf")
        best_move = None
        for row, col in candidate_moves:
            self.board[row][col] = player
            score = self._alphabeta(depth=2, is_max=False, alpha=float("-inf"), beta=float("inf"), player=player)
            self.board[row][col] = 0
            if score > best_score:
                best_score = score
                best_move = (row, col)

        # 执行最佳落子或兜底策略
        if best_move:
            return self.place_stone(best_move[0], best_move[1], player)
        else:
            return self.ai_move2()

    def _board_hash(self):
        return tuple(tuple(row) for row in self.board)

    def _evaluate_with_perspective(self, player):
        prev = self.current_player
        self.current_player = player
        val = self.evaluate_advanced()
        self.current_player = prev
        # 中心偏好
        center = (BOARD_SIZE - 1) / 2.0
        bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] == player:
                    dist = abs(row - center) + abs(col - center)
                    bonus += (3.0 - dist) * 2
        return val + bonus

    def _collect_threat_moves(self, player):
        opponent = 1 if player == 2 else 2
        threats = set()
        empties = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r][c] == 0]

        for row, col in empties:
            self.board[row][col] = player
            score = self._evaluate_with_perspective(player)
            self.board[row][col] = 0
            if score >= 1200:
                threats.add((row, col))

        for row, col in empties:
            self.board[row][col] = opponent
            opp_score = self._evaluate_with_perspective(opponent)
            self.board[row][col] = 0
            if opp_score >= 1200:
                threats.add((row, col))

        center = (BOARD_SIZE - 1) / 2.0
        return sorted(list(threats), key=lambda mv: abs(mv[0]-center) + abs(mv[1]-center))

    def _ordered_candidates(self, player):
        center = (BOARD_SIZE - 1) / 2.0
        moves = self.get_candidate_moves()
        scored = []
        for row, col in moves:
            self.board[row][col] = player
            s = self._evaluate_with_perspective(player)
            self.board[row][col] = 0
            dist = abs(row - center) + abs(col - center)
            s += (3.0 - dist) * 2
            scored.append(((row, col), s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [mv for mv, _ in scored[:8]]

    def _alphabeta(self, depth, is_max, alpha, beta, player):
        opponent = 1 if player == 2 else 2

        # 终局判定
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] != 0 and self.check_win(row, col):
                    return 1000000 if self.board[row][col] == player else -1000000

        if depth == 0:
            return self._evaluate_with_perspective(player)

        key = (self._board_hash(), player, depth, is_max)
        if key in self._transposition_table:
            return self._transposition_table[key]

        if is_max:
            moves = self._collect_threat_moves(player) or self._ordered_candidates(player)
            best = float("-inf")
            for row, col in moves:
                self.board[row][col] = player
                val = self._alphabeta(depth - 1, False, alpha, beta, player)
                self.board[row][col] = 0
                best = max(best, val)
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            self._transposition_table[key] = best
            return best
        else:
            moves = self._collect_threat_moves(opponent) or self._ordered_candidates(opponent)
            best = float("inf")
            for row, col in moves:
                self.board[row][col] = opponent
                val = self._alphabeta(depth - 1, True, alpha, beta, player)
                self.board[row][col] = 0
                best = min(best, val)
                beta = min(beta, val)
                if alpha >= beta:
                    break
            self._transposition_table[key] = best
            return best

    def place_stone(self, row, col, player):
        self.board[row][col] = player
        cx = self.grid_left + col * self.grid_cell_size
        cy = self.grid_top + row * self.grid_cell_size
        img = self.black_image if player == 1 else self.white_image
        self.canvas.create_image(cx, cy, image=img)
        self.stone_refs.append(img)

        # 胜负判断
        if self.check_win(row, col):
            winner = "黑棋" if player == 1 else "白棋"
            self.winner = player  # 记录胜者：1=黑，2=白
            self.status_var.set(f"{winner}胜！")
            self.game_over = True
            messagebox.showinfo("战况", f"{winner}获胜！")
            return

        # 切换回合
        self.current_player = 2 if player == 1 else 1
        self.status_var.set("黑棋回合" if self.current_player == 1 else "白棋回合")

        # AI 下
        if self.game_mode == "PVE" and self.current_player == self.ai_player and not self.game_over:
            self.root.after(100, self.ai_move)

    def pixel_to_grid(self, x, y):
        col = round((x - self.grid_left) / self.grid_cell_size)
        row = round((y - self.grid_top) / self.grid_cell_size)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None, None

    def reset_game(self):
        self.board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        self.canvas.delete("all")
        self.background_image = self.load_and_fit_background(BACKGROUND_IMAGE, CANVAS_WEIGHT, CANVAS_HEIGHT)
        self.canvas.create_image(0, 0, anchor="nw", image=self.background_image)
        self.draw_grid()
        self.status_var.set("黑棋回合")
        self.stone_refs.clear()
        if self.game_mode == "PVE" and self.ai_player == 1:
            self.ai_move()

    def check_win(self, row, col):
        player = self.board[row][col]
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            count += self.count_dir(row, col, dr, dc, player)
            count += self.count_dir(row, col, -dr, -dc, player)
            if count >= 5:
                return True
        return False

    def count_dir(self, row, col, row_step, col_step, player):
        connected_count = 0
        next_row, next_col = row + row_step, col + col_step
        while 0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE and self.board[next_row][next_col] == player:
            connected_count += 1
            next_row += row_step
            next_col += col_step
        return connected_count

# ===== 初始选择界面 =====
def start_menu(root):
    menu = tk.Toplevel(root)
    menu.title("模式选择")
    menu.geometry("300x200")
    menu.grab_set()  # 模态窗口

    tk.Label(menu, text="请选择游戏模式", font=("Arial", 14)).pack(pady=10)

    # PVP 模式按钮
    def start_pvp():
        menu.destroy()
        root.deiconify()
        GomokuGUI(root, game_mode="PVP")

    tk.Button(menu, text="玩家 vs 玩家 (PVP)", width=20, command=start_pvp).pack(pady=5)

    # PVE 模式按钮
    def choose_pve():
        # 难度选择窗口
        diff_win = tk.Toplevel(menu)
        diff_win.title("选择难度")
        diff_win.geometry("250x300")
        diff_win.grab_set()

        tk.Label(diff_win, text="请选择难度", font=("Arial", 12)).pack(pady=10)

        def select_difficulty(level):
            diff_win.destroy()
            choose_side(level)

        tk.Button(diff_win, text="简单", width=15, command=lambda: select_difficulty(1)).pack(pady=5)
        tk.Button(diff_win, text="一般", width=15, command=lambda: select_difficulty(2)).pack(pady=5)
        tk.Button(diff_win, text="较困难", width=15, command=lambda: select_difficulty(3)).pack(pady=5)
        tk.Button(diff_win, text="困难", width=15, command=lambda: select_difficulty(4)).pack(pady=5)
        tk.Button(diff_win, text="地狱", width=15, command=lambda: select_difficulty(5)).pack(pady=5)

    def choose_side(ai_level):
        side_win = tk.Toplevel(menu)
        side_win.title("选择执棋方")
        side_win.geometry("250x150")
        side_win.grab_set()

        tk.Label(side_win, text="请选择你的棋子颜色", font=("Arial", 12)).pack(pady=10)

        def start_as_black():
            side_win.destroy()
            menu.destroy()
            root.deiconify()
            GomokuGUI(root, game_mode="PVE", ai_player=2, ai_level=ai_level)  # 玩家黑棋，AI 白棋

        def start_as_white():
            side_win.destroy()
            menu.destroy()
            root.deiconify()
            GomokuGUI(root, game_mode="PVE", ai_player=1, ai_level=ai_level)  # 玩家白棋，AI 黑棋

        tk.Button(side_win, text="执黑 (先手)", width=15, command=start_as_black).pack(pady=5)
        tk.Button(side_win, text="执白 (后手)", width=15, command=start_as_white).pack(pady=5)

    tk.Button(menu, text="玩家 vs AI (PVE)", width=20, command=choose_pve).pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口，先显示选择界面
    start_menu(root)
    root.mainloop()
