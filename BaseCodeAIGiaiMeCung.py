import random
import time
import heapq
import queue
import threading
import tkinter as tk
from collections import deque
from tkinter import ttk


# ============================================================================
# Heuristics
# ============================================================================
class Heuristics:
    @staticmethod
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def euclidean(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def chebyshev(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def octile(a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return dx + dy + (1.41421356237 - 2) * min(dx, dy)


# ============================================================================
# Path reconstruction
# ============================================================================
def reconstruct_path(parent, start, end):
    """Lưu vết đường đi O(N) bộ nhớ thay vì nối chuỗi mảng O(N^2)"""
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = parent.get(curr)
    path.reverse()
    return path if path and path[0] == start else None


# ============================================================================
# Search algorithms
# ============================================================================
def solve_bfs(maze, start, end):
    start_time = time.perf_counter()
    q = deque([start])
    visited_order = []
    parent = {start: None}

    while q:
        curr = q.popleft()
        visited_order.append(curr)

        if curr == end:
            return reconstruct_path(parent, start, end), visited_order, (time.perf_counter() - start_time) * 1000

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and maze[nr][nc] != 1:
                next_node = (nr, nc)
                if next_node not in parent:
                    parent[next_node] = curr
                    q.append(next_node)

    return None, visited_order, (time.perf_counter() - start_time) * 1000


def solve_dfs(maze, start, end):
    start_time = time.perf_counter()
    stack = [start]
    visited_order = []
    parent = {start: None}

    while stack:
        curr = stack.pop()
        if curr not in visited_order:
            visited_order.append(curr)

        if curr == end:
            return reconstruct_path(parent, start, end), visited_order, (time.perf_counter() - start_time) * 1000

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and maze[nr][nc] != 1:
                next_node = (nr, nc)
                if next_node not in parent:
                    parent[next_node] = curr
                    stack.append(next_node)

    return None, visited_order, (time.perf_counter() - start_time) * 1000


def solve_astar(maze, start, end, heuristic_name="manhattan"):
    start_time = time.perf_counter()
    h_func = getattr(Heuristics, heuristic_name, Heuristics.manhattan)

    pq = [(h_func(start, end), 0, start)]
    g_score = {start: 0}
    parent = {start: None}
    visited_order = []
    closed_set = set()

    while pq:
        _, g, curr = heapq.heappop(pq)
        if curr in closed_set:
            continue

        closed_set.add(curr)
        visited_order.append(curr)

        if curr == end:
            return reconstruct_path(parent, start, end), visited_order, (time.perf_counter() - start_time) * 1000

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and maze[nr][nc] != 1:
                nxt = (nr, nc)
                tentative_g = g + 1
                if nxt not in g_score or tentative_g < g_score[nxt]:
                    g_score[nxt] = tentative_g
                    parent[nxt] = curr
                    f_score = tentative_g + h_func(nxt, end)
                    heapq.heappush(pq, (f_score, tentative_g, nxt))

    return None, visited_order, (time.perf_counter() - start_time) * 1000


def solve_jps(maze, start, end):
    """Triển khai chuẩn Jump Point Search bằng cơ chế Nhảy (Jump Function)"""
    start_time = time.perf_counter()
    rows, cols = len(maze), len(maze[0])

    def jump(r, c, dr, dc):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols) or maze[nr][nc] == 1:
            return None

        if (nr, nc) == end:
            return (nr, nc)

        # Check forced neighbors (điểm uốn buộc phải rẽ)
        if dr != 0:
            if (c > 0 and maze[nr][c - 1] == 1 and 0 <= nc - 1 and maze[nr + dr][c - 1] != 1) or \
               (c < cols - 1 and maze[nr][c + 1] == 1 and nc + 1 < cols and maze[nr + dr][c + 1] != 1):
                return (nr, nc)
        elif dc != 0:
            if (r > 0 and maze[r - 1][nc] == 1 and 0 <= nr - 1 and maze[r - 1][nc + dc] != 1) or \
               (r < rows - 1 and maze[r + 1][nc] == 1 and nr + 1 < rows and maze[r + 1][nc + dc] != 1):
                return (nr, nc)

            if jump(nr, nc, 1, 0) or jump(nr, nc, -1, 0):
                return (nr, nc)

        return jump(nr, nc, dr, dc)

    pq = [(Heuristics.manhattan(start, end), 0, start)]
    g_score = {start: 0}
    parent = {start: None}
    visited_order = []

    while pq:
        _, g, curr = heapq.heappop(pq)
        visited_order.append(curr)

        if curr == end:
            return reconstruct_path(parent, start, end), visited_order, (time.perf_counter() - start_time) * 1000

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            jp = jump(r, c, dr, dc)
            if jp:
                dist = abs(jp[0] - r) + abs(jp[1] - c)
                tentative_g = g + dist
                if jp not in g_score or tentative_g < g_score[jp]:
                    g_score[jp] = tentative_g
                    parent[jp] = curr
                    f_score = tentative_g + Heuristics.manhattan(jp, end)
                    heapq.heappush(pq, (f_score, tentative_g, jp))

    return None, visited_order, (time.perf_counter() - start_time) * 1000


# ============================================================================
# Q‑Learning Agent
# ============================================================================
class QLearningAgent:
    def __init__(self, maze, start, end, alpha=0.1, gamma=0.9, episodes=2000):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.start = start
        self.end = end
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.q_table = {}

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        q_vals = [self.get_q(state, a) for a in range(4)]
        max_q = max(q_vals)
        best_actions = [a for a in range(4) if q_vals[a] == max_q]
        return random.choice(best_actions)

    def train(self, progress_callback=None):
        for ep in range(self.episodes):
            state = self.start
            epsilon = max(0.01, 1.0 - (ep / self.episodes))

            for _ in range(500):
                if state == self.end:
                    break

                action = self.choose_action(state, epsilon)
                dr, dc = self.actions[action]
                nxt = (state[0] + dr, state[1] + dc)

                if not (0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols) or self.maze[nxt[0]][nxt[1]] == 1:
                    reward = -10.0
                    nxt = state
                elif nxt == self.end:
                    reward = 100.0
                else:
                    reward = -1.0

                best_next_q = max([self.get_q(nxt, a) for a in range(4)]) if nxt != self.end else 0.0
                old_q = self.get_q(state, action)
                self.q_table[(state, action)] = old_q + self.alpha * (reward + self.gamma * best_next_q - old_q)
                state = nxt

            if progress_callback and ep % 200 == 0:
                progress_callback(int(100 * ep / self.episodes))

    def get_path(self):
        state = self.start
        path = [state]
        visited_order = [state]
        curr = state

        for _ in range(self.rows * self.cols):
            if curr == self.end:
                return path, visited_order

            q_vals = [self.get_q(curr, a) for a in range(4)]
            best_action = max(range(4), key=lambda a: q_vals[a])
            dr, dc = self.actions[best_action]
            nxt = (curr[0] + dr, curr[1] + dc)

            if nxt == curr or nxt in path or self.maze[nxt[0]][nxt[1]] == 1:
                break

            curr = nxt
            path.append(curr)
            visited_order.append(curr)

        return (path, visited_order) if curr == self.end else (None, visited_order)


# ============================================================================
# Maze utilities
# ============================================================================
def create_maze(rows=21, cols=31):
    maze = [[1 for _ in range(cols)] for _ in range(rows)]
    stack = [(1, 1)]
    maze[1][1] = 0

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and maze[nr][nc] == 1:
                neighbors.append((nr, nc))
        if neighbors:
            nr, nc = random.choice(neighbors)
            maze[r + (nr - r) // 2][c + (nc - c) // 2] = 0
            maze[nr][nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    maze[rows // 2][0] = 'S'
    maze[rows // 2][1] = 0
    maze[rows // 2][cols - 1] = 'E'
    maze[rows // 2][cols - 2] = 0
    return maze


def analyze_maze(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    wall_count = sum(row.count(1) for row in maze)
    open_cells = (rows * cols) - wall_count

    if open_cells < 200:
        recommended = ("BFS", "Mê cung nhỏ, BFS duyệt tối ưu và chính xác nhất.")
    else:
        recommended = ("A*", "Mê cung kích thước lớn, A* tiết kiệm thời gian duyệt ô.")

    return {
        "rows": rows, "cols": cols, "open_cells": open_cells,
        "wall_ratio": wall_count / (rows * cols), "recommended": recommended
    }


# ============================================================================
# GUI Application
# ============================================================================
class MazeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Pathfinding Solver")
        self.rows, self.cols = 21, 31
        self.cell_size = 25

        self.maze = create_maze(self.rows, self.cols)
        self.start, self.end = (self.rows // 2, 0), (self.rows // 2, self.cols - 1)

        self.canvas = tk.Canvas(root, width=self.cols * self.cell_size, height=self.rows * self.cell_size, bg="white")
        self.canvas.pack(pady=10)

        # Lưu lại IDs của Rectangle trên Canvas để đổi màu trực tiếp, không xóa vẽ lại
        self.grid_ids = {}
        self.init_canvas()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Mê Cung Mới", command=self.new_maze).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="BFS", command=lambda: self.run_algo(solve_bfs)).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="DFS", command=lambda: self.run_algo(solve_dfs)).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="A*", command=lambda: self.run_algo(solve_astar)).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="JPS", command=lambda: self.run_algo(solve_jps)).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Q-Learning", command=self.run_qlearning).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Phân Tích", command=self.show_analysis).pack(side=tk.LEFT, padx=3)

        self.status_label = tk.Label(root, text="Sẵn sàng", font=("Arial", 10))
        self.status_label.pack(pady=5)

        self.anim_id = None

    def init_canvas(self):
        self.canvas.delete("all")
        self.grid_ids.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                val = self.maze[r][c]
                color = "black" if val == 1 else "lightgreen" if (r, c) == self.start else "lightcoral" if (r, c) == self.end else "white"
                rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#ddd")
                self.grid_ids[(r, c)] = rect_id

    def reset_colors(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        for (r, c), rect_id in self.grid_ids.items():
            val = self.maze[r][c]
            color = "black" if val == 1 else "lightgreen" if (r, c) == self.start else "lightcoral" if (r, c) == self.end else "white"
            self.canvas.itemconfig(rect_id, fill=color)

    def new_maze(self):
        self.maze = create_maze(self.rows, self.cols)
        self.init_canvas()
        self.status_label.config(text="Đã tạo mê cung mới")

    def run_algo(self, algo_func):
        self.reset_colors()
        path, visited, execution_time = algo_func(self.maze, self.start, self.end)
        self.animate(visited, path, 0, execution_time)

    def animate(self, visited, path, idx, execution_time):
        if idx < len(visited):
            cell = visited[idx]
            if cell not in (self.start, self.end):
                self.canvas.itemconfig(self.grid_ids[cell], fill="#e0f7fa")
            self.anim_id = self.root.after(10, lambda: self.animate(visited, path, idx + 1, execution_time))
        elif path and idx < len(visited) + len(path):
            p_idx = idx - len(visited)
            cell = path[p_idx]
            if cell not in (self.start, self.end):
                self.canvas.itemconfig(self.grid_ids[cell], fill="#ffd54f")
            self.anim_id = self.root.after(20, lambda: self.animate(visited, path, idx + 1, execution_time))
        else:
            steps = len(path) if path else 0
            self.status_label.config(text=f"Hoàn thành | Đã duyệt: {len(visited)} ô | Đường đi: {steps} bước | Thời gian: {execution_time:.2f} ms")

    def run_qlearning(self):
        self.reset_colors()
        self.status_label.config(text="Đang huấn luyện Q-Learning... 0%")
        q = queue.Queue()
        agent = QLearningAgent(self.maze, self.start, self.end)

        def worker():
            agent.train(progress_callback=lambda p: q.put(("progress", p)))
            path, visited = agent.get_path()
            q.put(("done", path, visited))

        threading.Thread(target=worker, daemon=True).start()
        self.poll_qlearning(q)

    def poll_qlearning(self, q):
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "progress":
                    self.status_label.config(text=f"Đang huấn luyện Q-Learning... {msg[1]}%")
                elif msg[0] == "done":
                    _, path, visited = msg
                    self.animate(visited, path, 0, 0.0)
                    return
        except queue.Empty:
            pass
        self.root.after(50, lambda: self.poll_qlearning(q))

    def show_analysis(self):
        info = analyze_maze(self.maze, self.start, self.end)
        algo_name, reason = info["recommended"]
        self.status_label.config(text=f"Gợi ý: {algo_name} — {reason}")


# ============================================================================
# Main entry point
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGUI(root)
    root.mainloop()
