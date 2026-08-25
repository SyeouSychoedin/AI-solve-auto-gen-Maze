import tkinter as tk
from collections import deque
from tkinter import ttk
import heapq
import random
import time
import threading
import queue
import math

# ================== TẠO MÊ CUNG ==================
def create_large_maze(rows=41, cols=41):
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

def make_unsolvable(maze, start, end):
    maze_copy = [row[:] for row in maze]
    while True:
        path, _, _ = solve_bfs(maze_copy, start, end)
        if path is None:
            break
        candidates = [cell for cell in path if maze_copy[cell[0]][cell[1]] not in ('S','E')]
        if not candidates:
            break
        r, c = random.choice(candidates)
        maze_copy[r][c] = 1
    return maze_copy

# ================== THUẬT TOÁN CỔ ĐIỂN ==================
def solve_bfs(maze, start, end):
    start_time = time.perf_counter()
    queue = deque([(start, [start])])
    visited = {start}
    visited_order = [start]
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            return path, visited_order, (time.perf_counter() - start_time) * 1000
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and (nr, nc) not in visited:
                if maze[nr][nc] != 1:
                    visited.add((nr, nc))
                    visited_order.append((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
    return None, visited_order, 0

def solve_dfs(maze, start, end):
    start_time = time.perf_counter()
    stack = [(start, [start])]
    visited = {start}
    visited_order = [start]
    while stack:
        (r, c), path = stack.pop()
        if (r, c) == end:
            return path, visited_order, (time.perf_counter() - start_time) * 1000
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and (nr, nc) not in visited:
                if maze[nr][nc] != 1:
                    visited.add((nr, nc))
                    visited_order.append((nr, nc))
                    stack.append(((nr, nc), path + [(nr, nc)]))
    return None, visited_order, 0

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) #|hàng_a − hàng_b| + |cột_a − cột_b|

def solve_ucs(maze, start, end):
    start_time = time.perf_counter()
    pq = []
    heapq.heappush(pq, (0, start, [start]))
    cost_so_far = {start: 0}
    visited_order = []
    while pq:
        cost, (r, c), path = heapq.heappop(pq)
        if cost > cost_so_far.get((r, c), float('inf')):
            continue
        visited_order.append((r, c))
        if (r, c) == end:
            return path, visited_order, (time.perf_counter() - start_time) * 1000
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and maze[nr][nc] != 1:
                new_cost = cost + 1
                if new_cost < cost_so_far.get((nr, nc), float('inf')):
                    cost_so_far[(nr, nc)] = new_cost
                    heapq.heappush(pq, (new_cost, (nr, nc), path + [(nr, nc)]))
    return None, visited_order, (time.perf_counter() - start_time) * 1000

def solve_greedy(maze, start, end):
    start_time = time.perf_counter()
    pq = []
    heapq.heappush(pq, (heuristic(start, end), start, [start]))
    visited_order = []
    visited_set = set()
    while pq:
        h, (r, c), path = heapq.heappop(pq)
        if (r, c) in visited_set:
            continue
        visited_set.add((r, c))
        visited_order.append((r, c))
        if (r, c) == end:
            return path, visited_order, (time.perf_counter() - start_time) * 1000
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]):
                if maze[nr][nc] != 1 and (nr, nc) not in visited_set:
                    heapq.heappush(pq, (heuristic((nr, nc), end), (nr, nc), path + [(nr, nc)]))
    return None, visited_order, 0

# ================== THUẬT TOÁN NÂNG CAO ==================

def solve_iddfs(maze, start, end):
    """Iterative Deepening DFS - Tìm đường tối ưu với bộ nhớ thấp"""
    start_time = time.perf_counter()
    rows, cols = len(maze), len(maze[0])
    max_depth = rows * cols
    
    def dfs_limit(node, depth, path, visited):
        if node == end:
            return path
        if depth == 0:
            return None
        
        r, c = node
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] != 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    result = dfs_limit((nr, nc), depth - 1, path + [(nr, nc)], visited)
                    if result:
                        return result
                    visited.remove((nr, nc))
        return None
    
    visited_order = [start]
    for depth in range(1, max_depth + 1):
        visited = {start}
        result = dfs_limit(start, depth, [start], visited)
        if result:
            return result, visited_order, (time.perf_counter() - start_time) * 1000
        visited_order.extend(list(visited))
    
    return None, visited_order, 0

def solve_bidirectional_bfs(maze, start, end):
    """BFS hai chiều - Tìm đường nhanh hơn BFS thông thường"""
    start_time = time.perf_counter()
    
    if start == end:
        return [start], [start], 0
    
    rows, cols = len(maze), len(maze[0])
    queue_start = deque([start])
    queue_end = deque([end])
    
    visited_start = {start: None}
    visited_end = {end: None}
    visited_order = [start, end]
    
    def reconstruct_path():
        intersection = None
        for node in visited_start:
            if node in visited_end:
                intersection = node
                break
        
        if not intersection:
            return None
        
        path_start = []
        current = intersection
        while current is not None:
            path_start.append(current)
            current = visited_start[current]
        path_start.reverse()
        
        path_end = []
        current = visited_end[intersection]
        while current is not None:
            path_end.append(current)
            current = visited_end[current]
        
        return path_start + path_end
    
    while queue_start and queue_end:
        # Mở rộng từ phía start
        for _ in range(len(queue_start)):
            current = queue_start.popleft()
            r, c = current
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if maze[nr][nc] != 1 and (nr, nc) not in visited_start:
                        visited_start[(nr, nc)] = current
                        visited_order.append((nr, nc))
                        if (nr, nc) in visited_end:
                            path = reconstruct_path()
                            if path:
                                return path, visited_order, (time.perf_counter() - start_time) * 1000
                        queue_start.append((nr, nc))
        
        # Mở rộng từ phía end
        for _ in range(len(queue_end)):
            current = queue_end.popleft()
            r, c = current
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if maze[nr][nc] != 1 and (nr, nc) not in visited_end:
                        visited_end[(nr, nc)] = current
                        visited_order.append((nr, nc))
                        if (nr, nc) in visited_start:
                            path = reconstruct_path()
                            if path:
                                return path, visited_order, (time.perf_counter() - start_time) * 1000
                        queue_end.append((nr, nc))
    
    return None, visited_order, 0

def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) #|Δx| + |Δy|

def heuristic_euclidean(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5 #√(Δx² + Δy²) Duong Chim Bay

def heuristic_chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) #max(|Δx|, |Δy|) Duyet O Cheo

def heuristic_octile(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy + (2**0.5 - 2) * min(dx, dy) #|Δx| + |Δy| + (√2 − 2)·min(|Δx|,|Δy|)

def solve_astar_adaptive(maze, start, end, heuristic_type='manhattan'):
    """A* với nhiều lựa chọn heuristic"""
    start_time = time.perf_counter()
    
    heuristics = {
        'manhattan': heuristic_manhattan,
        'euclidean': heuristic_euclidean,
        'chebyshev': heuristic_chebyshev,
        'octile': heuristic_octile
    }
    
    h_func = heuristics.get(heuristic_type, heuristic_manhattan)
    
    pq = []
    heapq.heappush(pq, (h_func(start, end), 0, start, [start]))
    g_score = {start: 0}
    visited_order = []
    visited_set = set()
    
    while pq:
        f, g, (r, c), path = heapq.heappop(pq)
        
        if (r, c) in visited_set:
            continue
        
        visited_set.add((r, c))
        visited_order.append((r, c))
        
        if (r, c) == end:
            return path, visited_order, (time.perf_counter() - start_time) * 1000
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]):
                if maze[nr][nc] != 1 and (nr, nc) not in visited_set:
                    tentative_g = g + 1
                    if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                        g_score[(nr, nc)] = tentative_g
                        heapq.heappush(pq, (tentative_g + h_func((nr, nc), end), 
                                          tentative_g, (nr, nc), path + [(nr, nc)]))
    
    return None, visited_order, 0

def solve_jps(maze, start, end):
    """Jump Point Search - A* tối ưu cho mê cung"""
    start_time = time.perf_counter()
    rows, cols = len(maze), len(maze[0])
    
    def get_neighbors(node, parent):
        r, c = node
        neighbors = []
        
        if parent is None:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if maze[nr][nc] != 1:
                        neighbors.append((nr, nc))
        else:
            pr, pc = parent
            dr, dc = r - pr, c - pc
            
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] != 1:
                    neighbors.append((nr, nc))
            
            if dr != 0:
                for dc2 in [-1, 1]:
                    nr, nc = r, c + dc2
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if maze[nr][nc] != 1:
                            neighbors.append((nr, nc))
            else:
                for dr2 in [-1, 1]:
                    nr, nc = r + dr2, c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if maze[nr][nc] != 1:
                            neighbors.append((nr, nc))
        
        return neighbors
    
    pq = []
    heapq.heappush(pq, (heuristic_manhattan(start, end), 0, start, [start], None))
    g_score = {start: 0}
    visited_order = [start]
    visited_set = {start}
    
    while pq:
        f, g, node, path, parent = heapq.heappop(pq)
        
        if node == end:
            return path, visited_order, (time.perf_counter() - start_time) * 1000
        
        for neighbor in get_neighbors(node, parent):
            if neighbor in visited_set:
                continue
            
            visited_set.add(neighbor)
            visited_order.append(neighbor)
            new_g = g + 1
            
            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                heapq.heappush(pq, (new_g + heuristic_manhattan(neighbor, end),
                                  new_g, neighbor, path + [neighbor], node))
    
    return None, visited_order, 0

def solve_dijkstra_optimized(maze, start, end):
    """Dijkstra tối ưu với early exit"""
    start_time = time.perf_counter()
    
    rows, cols = len(maze), len(maze[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[start[0]][start[1]] = 0
    
    pq = [(0, start, [start])]
    visited_order = [start]
    visited = {start}
    
    while pq:
        cost, (r, c), path = heapq.heappop(pq)
        
        if cost > dist[r][c]:
            continue
            
        if (r, c) == end:
            return path, visited_order, (time.perf_counter() - start_time) * 1000
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != 1:
                new_cost = cost + 1
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        visited_order.append((nr, nc))
                    heapq.heappush(pq, (new_cost, (nr, nc), path + [(nr, nc)]))
    
    return None, visited_order, 0

# ================== AI ADVISOR ==================
class SmartAIAdvisor:
    """AI phân tích và chọn thuật toán tốt nhất"""
    def __init__(self):
        pass
    
    def analyze_maze(self, maze, start, end):
        """Phân tích chi tiết mê cung"""
        rows, cols = len(maze), len(maze[0])
        total_cells = rows * cols
        wall_count = sum(row.count(1) for row in maze)
        wall_ratio = wall_count / total_cells if total_cells > 0 else 0
        
        open_cells = total_cells - wall_count
        complexity = "thấp" if open_cells < 200 else "trung bình" if open_cells < 500 else "cao"
        
        min_distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
        
        dead_ends = self._count_dead_ends(maze)
        branch_factor = self._calculate_branching(maze)
        
        return {
            'rows': rows,
            'cols': cols,
            'total_cells': total_cells,
            'wall_ratio': wall_ratio,
            'open_cells': open_cells,
            'complexity': complexity,
            'min_distance': min_distance,
            'dead_ends': dead_ends,
            'branch_factor': branch_factor
        }
    
    def _count_dead_ends(self, maze):
        """Đếm số ngõ cụt trong mê cung"""
        dead_ends = 0
        rows, cols = len(maze), len(maze[0])
        
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if maze[r][c] != 1:
                    open_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != 1:
                            open_neighbors += 1
                    if open_neighbors == 1:
                        dead_ends += 1
        return dead_ends
    
    def _calculate_branching(self, maze):
        """Tính hệ số nhánh của mê cung"""
        rows, cols = len(maze), len(maze[0])
        total_open = 0
        total_branches = 0
        
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if maze[r][c] != 1:
                    total_open += 1
                    open_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != 1:
                            open_neighbors += 1
                    total_branches += open_neighbors
        
        return total_branches / total_open if total_open > 0 else 0
    
    def suggest_algorithm(self, maze, start, end, has_delivery=False):
        """Đề xuất thuật toán tốt nhất"""
        analysis = self.analyze_maze(maze, start, end)
        
        suggestions = []
        
        if has_delivery:
            suggestions.append(("A* (Manhattan)", "Tối ưu cho nhiều điểm giao hàng", 90))
        
        if analysis['complexity'] == 'cao':
            if analysis['min_distance'] < 50:
                suggestions.append(("Bidirectional BFS", "Khoảng cách ngắn, tìm nhanh", 85))
            suggestions.append(("A* (Manhattan)", "Tìm đường tối ưu với heuristic tốt", 80))
            suggestions.append(("JPS", "Tối ưu hóa cho mê cung lớn", 75))
        
        elif analysis['complexity'] == 'trung bình':
            if analysis['branch_factor'] > 2:
                suggestions.append(("A* (Euclidean)", "Heuristic phù hợp với nhiều nhánh", 85))
            else:
                suggestions.append(("IDDFS", "Ít nhánh, tiết kiệm bộ nhớ", 80))
        
        else:
            suggestions.append(("BFS", "Mê cung nhỏ, đảm bảo đường đi ngắn nhất", 90))
            suggestions.append(("DFS", "Mê cung nhỏ, tốc độ nhanh", 85))
        
        if analysis['dead_ends'] > analysis['open_cells'] * 0.3:
            suggestions.append(("A* (Manhattan)", "Nhiều ngõ cụt, cần heuristic tốt", 70))
        
        suggestions.sort(key=lambda x: x[2], reverse=True)
        
        return suggestions[0] if suggestions else ("BFS", "Mặc định", 50)

# ================== Q-LEARNING HỢP NHẤT ==================
class QLearningAgent:
    def __init__(self, maze, start, end, delivery_points=None,
                 alpha=0.7, gamma=0.95, episodes=5000, max_steps=10000,
                 eps_start=1.0, eps_end=0.05):
        self.maze      = maze
        self.rows      = len(maze)
        self.cols      = len(maze[0])
        self.start     = start
        self.end       = end
        self.delivery_points = set(delivery_points) if delivery_points else set()
        self.has_delivery = len(self.delivery_points) > 0
        self.alpha     = alpha
        self.gamma     = gamma
        self.episodes  = episodes
        self.max_steps = max_steps
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.actions   = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        if self.has_delivery:
            self.q = {}
        else:
            self.q = [[[0.0] * 4 for _ in range(self.cols)] for _ in range(self.rows)]

    def valid_actions(self, state):
        valid = []
        for action in range(4):
            if self.step(state, action) != state:
                valid.append(action)
        return valid

    def _get_q_values(self, state, collected_set=None):
        if self.has_delivery:
            key = (state[0], state[1], frozenset(collected_set))
            return self.q.get(key, [0.0] * 4)
        else:
            r, c = state
            return self.q[r][c]

    def _set_q_values(self, state, collected_set, values):
        if self.has_delivery:
            key = (state[0], state[1], frozenset(collected_set))
            self.q[key] = values
        else:
            r, c = state
            self.q[r][c] = values

    def choose_action(self, state, epsilon, collected_set=None):
        valid = self.valid_actions(state)
        if not valid:
            return random.randint(0, 3)
        if random.random() < epsilon:
            return random.choice(valid)
        
        q_vals = self._get_q_values(state, collected_set)
        best_value = max(q_vals[action] for action in valid)
        best_actions = [action for action in valid if q_vals[action] == best_value]
        return random.choice(best_actions)

    def step(self, state, action):
        r, c = state
        dr, dc = self.actions[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.maze[nr][nc] != 1:
            return (nr, nc)
        return state

    def _distance(self, state):
        return abs(state[0] - self.end[0]) + abs(state[1] - self.end[1])

    def _nearest_delivery(self, state, collected_set):
        if not self.has_delivery:
            return 0
        uncollected = [p for p in self.delivery_points if p not in collected_set]
        if not uncollected:
            return 0
        return min(heuristic(state, p) for p in uncollected)

    def train(self, progress_callback=None):
        for ep in range(self.episodes):
            state = self.start
            collected = set()
            epsilon = max(self.eps_end,
                          self.eps_start - (self.eps_start - self.eps_end) * ep / max(1, self.episodes - 1))
            visited_this_episode = {state}
            last_new_step = 0

            for step in range(self.max_steps):
                action = self.choose_action(state, epsilon, collected)
                next_state = self.step(state, action)
                next_collected = set(collected)
                done = False

                if self.has_delivery and next_state == self.end:
                    if len(collected) == len(self.delivery_points):
                        reward, done = 5000.0, True
                    else:
                        next_state = state
                        reward, done = -5.0, False
                elif next_state == self.end:
                    reward, done = 5000.0, True
                elif next_state == state:
                    reward, done = -20.0, False
                else:
                    old_dist = self._distance(state)
                    new_dist = self._distance(next_state)
                    reward = (old_dist - new_dist) * 10.0 - 0.1
                    done = False

                    if next_state not in visited_this_episode:
                        reward += 15.0
                        visited_this_episode.add(next_state)
                        last_new_step = step
                    elif step - last_new_step > 30:
                        reward -= 2.0

                if self.has_delivery:
                    if next_state in self.delivery_points and next_state not in collected:
                        reward += 500.0
                        next_collected.add(next_state)
                    if next_state != state and next_collected != collected:
                        old_del_dist = self._nearest_delivery(state, collected)
                        new_del_dist = self._nearest_delivery(next_state, next_collected)
                        if new_del_dist < old_del_dist:
                            reward += 2.0
                    reward += 1.0 * len(next_collected)

                q_vals = self._get_q_values(state, collected)
                next_q_vals = self._get_q_values(next_state, next_collected)
                best_next = 0.0 if done else max(next_q_vals)
                q_vals[action] += self.alpha * (
                    reward + self.gamma * best_next - q_vals[action])
                self._set_q_values(state, collected, q_vals)

                state = next_state
                collected = next_collected
                if done:
                    break

            if progress_callback and ep % 500 == 0:
                progress_callback(ep, self.episodes)

    def get_path(self):
        state = self.start
        collected = set()
        path = [state]
        visited_order = [state]
        
        if self.has_delivery:
            visited_set = {(state, frozenset(collected))}
        else:
            visited_set = {state}

        for _ in range(self.max_steps):
            if self.has_delivery:
                if state == self.end and len(collected) == len(self.delivery_points):
                    return path, visited_order
            else:
                if state == self.end:
                    return path, visited_order

            valid = self.valid_actions(state)
            if not valid:
                break

            q_vals = self._get_q_values(state, collected)
            best_value = max(q_vals[action] for action in valid)
            best_actions = [action for action in valid if q_vals[action] == best_value]
            
            next_state = None
            next_collected = None

            for action in best_actions:
                candidate = self.step(state, action)
                if candidate == state:
                    continue
                
                new_col = set(collected)
                if self.has_delivery and candidate in self.delivery_points and candidate not in new_col:
                    new_col.add(candidate)
                
                if self.has_delivery:
                    new_pair = (candidate, frozenset(new_col))
                    if new_pair not in visited_set:
                        if candidate == self.end and len(collected) < len(self.delivery_points):
                            continue
                        next_state = candidate
                        next_collected = new_col
                        break
                else:
                    if candidate not in visited_set:
                        next_state = candidate
                        break

            if next_state is None:
                action = random.choice(best_actions)
                next_state = self.step(state, action)
                next_collected = set(collected)
                if self.has_delivery and next_state in self.delivery_points and next_state not in next_collected:
                    next_collected.add(next_state)

            if next_state == state:
                break

            path.append(next_state)
            visited_order.append(next_state)
            
            if self.has_delivery:
                visited_set.add((next_state, frozenset(next_collected)))
                collected = next_collected
            else:
                visited_set.add(next_state)

            state = next_state

        if self.has_delivery:
            if state == self.end and len(collected) == len(self.delivery_points):
                return path, visited_order
        else:
            if state == self.end:
                return path, visited_order
                
        return None, visited_order

# ================== GIAO DIỆN TKINTER ==================
class MazeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Giải Mê Cung - AI Advisor & Algorithms")
        self.rows, self.cols = 21, 31
        self.cell_size = 25
        self.maze = create_large_maze(self.rows, self.cols)
        self.start = self.find_cell('S')
        self.end = self.find_cell('E')
        self.delivery_points = set()

        canvas_width = self.cols * self.cell_size
        canvas_height = self.rows * self.cell_size
        self.canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        control_frame = tk.Frame(root)
        control_frame.pack()

        # Hàng 1: Điều khiển cơ bản
        row1 = tk.Frame(control_frame)
        row1.pack(pady=2)
        tk.Button(row1, text="New Maze", command=self.new_maze).pack(side=tk.LEFT, padx=2)
        tk.Button(row1, text="CantMap", command=self.cant_map).pack(side=tk.LEFT, padx=2)
        tk.Button(row1, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=2)
        tk.Button(row1, text="So sánh", command=self.compare_all).pack(side=tk.LEFT, padx=2)

        # Hàng 2: AI
        row2 = tk.Frame(control_frame)
        row2.pack(pady=2)
        tk.Button(row2, text="🤖 AI Suggest", command=self.ai_suggest,
                 bg="lightblue", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=2)
        tk.Button(row2, text="🚀 Smart Solve", command=self.smart_solve,
                 bg="lightgreen", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=2)

        # Hàng 3: Thuật toán cơ bản
        row3 = tk.Frame(control_frame)
        row3.pack(pady=2)
        tk.Button(row3, text="BFS", command=lambda: self.run_algorithm("bfs")).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="DFS", command=lambda: self.run_algorithm("dfs")).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="Bidirectional", command=lambda: self.run_algorithm("bidirectional")).pack(side=tk.LEFT, padx=2)
        tk.Button(row3, text="IDDFS", command=lambda: self.run_algorithm("iddfs")).pack(side=tk.LEFT, padx=2)

        # Hàng 4: Thuật toán nâng cao
        row4 = tk.Frame(control_frame)
        row4.pack(pady=2)
        tk.Button(row4, text="A*", command=lambda: self.run_algorithm("astar")).pack(side=tk.LEFT, padx=2)
        tk.Button(row4, text="JPS", command=lambda: self.run_algorithm("jps")).pack(side=tk.LEFT, padx=2)
        tk.Button(row4, text="Dijkstra", command=lambda: self.run_algorithm("dijkstra")).pack(side=tk.LEFT, padx=2)
        tk.Button(row4, text="Greedy", command=lambda: self.run_algorithm("greedy")).pack(side=tk.LEFT, padx=2)

        # Hàng 5: Heuristic và Q-Learning
        row5 = tk.Frame(control_frame)
        row5.pack(pady=2)
        tk.Label(row5, text="Heuristic:").pack(side=tk.LEFT, padx=(5, 0))
        self.heuristic_var = tk.StringVar(value="manhattan")
        heuristic_menu = ttk.Combobox(row5, textvariable=self.heuristic_var,
                                     values=["manhattan", "euclidean", "chebyshev", "octile"],
                                     width=10, state="readonly")
        heuristic_menu.pack(side=tk.LEFT, padx=5)
        tk.Button(row5, text="Q-Learning", command=self.run_qlearning).pack(side=tk.LEFT, padx=5)

        # Hàng 6: Tốc độ
        row6 = tk.Frame(control_frame)
        row6.pack(pady=2)
        tk.Label(row6, text="Tốc độ:").pack(side=tk.LEFT, padx=(10, 0))
        self.speed_var = tk.IntVar(value=50)
        tk.Scale(row6, from_=1, to=100, orient=tk.HORIZONTAL,
                variable=self.speed_var, length=100).pack(side=tk.LEFT)

        self.info_label = tk.Label(root, text="", font=("Arial", 10))
        self.info_label.pack(pady=5)

        self.animation_id = None
        self.current_path = None
        self.current_visited = None
        self.current_time = 0
        self.solved = False
        self.running = False
        self.q_queue = None

        self.draw_maze()

    def on_canvas_click(self, event):
        if self.running or self.animation_id is not None:
            return
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols and self.maze[r][c] == 0:
            pt = (r, c)
            if pt in self.delivery_points:
                self.delivery_points.remove(pt)
            else:
                self.delivery_points.add(pt)
            self.draw_maze()

    def find_cell(self, char):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze[r][c] == char:
                    return (r, c)
        return None

    def new_maze(self):
        if self.running:
            return
        self.maze = create_large_maze(self.rows, self.cols)
        self.start = self.find_cell('S')
        self.end = self.find_cell('E')
        self.delivery_points.clear()
        self.reset()

    def cant_map(self):
        if self.running:
            return
        maze = create_large_maze(self.rows, self.cols)
        start = (self.rows // 2, 0)
        end = (self.rows // 2, self.cols - 1)
        self.maze = make_unsolvable(maze, start, end)
        self.start = self.find_cell('S')
        self.end = self.find_cell('E')
        self.delivery_points.clear()
        self.reset()

    def reset(self):
        if self.running:
            return
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
        self.current_path = None
        self.current_visited = None
        self.solved = False
        self.info_label.config(text="")
        self.draw_maze()

    def draw_maze(self, highlight_path=None, highlight_visited=None):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                cell = self.maze[r][c]
                color = ("black" if cell == 1 else
                         "lightgreen" if cell == 'S' else
                         "lightcoral" if cell == 'E' else "white")
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

        for (r, c) in self.delivery_points:
            x1, y1 = c * self.cell_size, r * self.cell_size
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size, y1 + self.cell_size,
                                         fill="yellow", outline="gray")

        if highlight_visited:
            for r, c in highlight_visited:
                x1, y1 = c * self.cell_size, r * self.cell_size
                self.canvas.create_rectangle(x1, y1, x1 + self.cell_size, y1 + self.cell_size,
                                             fill="lightblue", outline="gray")

        if highlight_path:
            for r, c in highlight_path:
                x1, y1 = c * self.cell_size, r * self.cell_size
                self.canvas.create_rectangle(x1, y1, x1 + self.cell_size, y1 + self.cell_size,
                                             fill="gold", outline="gray")

        for char, color in [(self.start, "lightgreen"), (self.end, "lightcoral")]:
            if char:
                r, c = char
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                label = 'S' if char == self.start else 'E'
                self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text=label)

    def run_algorithm(self, algo):
        if self.running or self.animation_id:
            return
        if not self.start or not self.end:
            self.info_label.config(text="Lỗi: không tìm thấy điểm S/E!")
            return

        self.running = True
        
        funcs = {
            "bfs": solve_bfs,
            "dfs": solve_dfs,
            "ucs": solve_ucs,
            "astar": lambda m, s, e: solve_astar_adaptive(m, s, e, self.heuristic_var.get()),
            "greedy": solve_greedy,
            "bidirectional": solve_bidirectional_bfs,
            "iddfs": solve_iddfs,
            "jps": solve_jps,
            "dijkstra": solve_dijkstra_optimized
        }
        
        if algo in funcs:
            path, visited, t = funcs[algo](self.maze, self.start, self.end)
        else:
            path, visited, t = None, [], 0
        
        self.current_path = path
        self.current_visited = visited
        self.current_time = t
        self.solved = path is not None
        self.running = False
        self.animate(0)

    def ai_suggest(self):
        """AI đề xuất thuật toán phù hợp nhất"""
        if not self.start or not self.end:
            self.info_label.config(text="Không tìm thấy điểm S/E!")
            return
        
        advisor = SmartAIAdvisor()
        has_delivery = len(self.delivery_points) > 0
        suggested, reason, score = advisor.suggest_algorithm(
            self.maze, self.start, self.end, has_delivery
        )
        analysis = advisor.analyze_maze(self.maze, self.start, self.end)
        
        popup = tk.Toplevel(self.root)
        popup.title("🤖 Gợi ý từ AI")
        popup.geometry("500x400")
        
        tk.Label(popup, text="📊 PHÂN TÍCH MÊ CUNG", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        info_frame = tk.Frame(popup)
        info_frame.pack(pady=5)
        
        stats = [
            f"Kích thước: {analysis['rows']}x{analysis['cols']}",
            f"Tổng số ô: {analysis['total_cells']}",
            f"Tỷ lệ tường: {analysis['wall_ratio']*100:.1f}%",
            f"Độ phức tạp: {analysis['complexity']}",
            f"Ngõ cụt: {analysis['dead_ends']}",
            f"Hệ số nhánh: {analysis['branch_factor']:.2f}"
        ]
        
        for stat in stats:
            tk.Label(info_frame, text=f"• {stat}", 
                    font=("Arial", 10)).pack(anchor="w", padx=20)
        
        tk.Label(popup, text="\n🎯 AI ĐỀ XUẤT", 
                font=("Arial", 12, "bold"), fg="blue").pack(pady=5)
        
        tk.Label(popup, text=f"✅ Thuật toán: {suggested}", 
                font=("Arial", 12, "bold"), fg="green").pack()
        tk.Label(popup, text=f"📝 Lý do: {reason}", 
                font=("Arial", 10), wraplength=450).pack(pady=5)
        tk.Label(popup, text=f"⭐ Độ phù hợp: {score}/100", 
                font=("Arial", 10)).pack()
        
        def run_suggested():
            popup.destroy()
            algo_map = {
                'BFS': 'bfs',
                'DFS': 'dfs',
                'Bidirectional BFS': 'bidirectional',
                'IDDFS': 'iddfs',
                'A* (Manhattan)': 'astar',
                'A* (Euclidean)': 'astar',
                'JPS': 'jps',
                'Dijkstra': 'dijkstra'
            }
            if suggested in algo_map:
                self.run_algorithm(algo_map[suggested])
            elif suggested == 'Q-Learning':
                self.run_qlearning()
        
        tk.Button(popup, text=f"🚀 Chạy {suggested}", 
                 command=run_suggested, bg="lightgreen",
                 font=("Arial", 11, "bold")).pack(pady=15)

    def smart_solve(self):
        """Tự động chọn và chạy thuật toán tốt nhất"""
        advisor = SmartAIAdvisor()
        has_delivery = len(self.delivery_points) > 0
        algo, _, _ = advisor.suggest_algorithm(
            self.maze, self.start, self.end, has_delivery
        )
        
        self.info_label.config(text=f"🧠 AI đã chọn: {algo}")
        
        algo_map = {
            'BFS': 'bfs',
            'DFS': 'dfs',
            'Bidirectional BFS': 'bidirectional',
            'IDDFS': 'iddfs',
            'A* (Manhattan)': 'astar',
            'A* (Euclidean)': 'astar',
            'JPS': 'jps',
            'Dijkstra': 'dijkstra'
        }
        
        if algo in algo_map:
            self.run_algorithm(algo_map[algo])
        elif algo == 'Q-Learning':
            self.run_qlearning()

    def run_qlearning(self):
        if self.running or self.animation_id:
            return
        if not self.start or not self.end:
            self.info_label.config(text="Lỗi: không tìm thấy điểm S/E!")
            return

        self.running = True
        self.info_label.config(text="Đang huấn luyện Q-Learning... 0%")
        self.q_queue = queue.Queue()
        threading.Thread(target=self._qlearning_worker, args=(self.q_queue,), daemon=True).start()
        self._poll_queue()

    def _poll_queue(self):
        try:
            while True:
                msg = self.q_queue.get_nowait()
                if msg[0] == "progress":
                    _, percent = msg
                    self.info_label.config(text=f"Đang huấn luyện Q-Learning... {percent}%")
                elif msg[0] == "done":
                    _, path, visited, elapsed = msg
                    self._finish_qlearning(path, visited, elapsed)
                    return
                elif msg[0] == "error":
                    _, err_msg = msg
                    self.info_label.config(text=f"Lỗi Q-Learning: {err_msg}")
                    self.running = False
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _qlearning_worker(self, q):
        try:
            # Luôn dùng QLearningAgent hợp nhất (tự xử lý cả 2 trường hợp)
            agent = QLearningAgent(
                self.maze, self.start, self.end,
                delivery_points=self.delivery_points if self.delivery_points else None,
                episodes=5000, max_steps=10000, alpha=0.7
            )

            def progress_callback(ep, total):
                percent = int(100 * ep / total)
                q.put(("progress", percent))

            t0 = time.perf_counter()
            agent.train(progress_callback=progress_callback)
            elapsed = (time.perf_counter() - t0) * 1000
            path, visited = agent.get_path()
            q.put(("done", path, visited, elapsed))
        except Exception as e:
            q.put(("error", str(e)))

    def _finish_qlearning(self, path, visited, train_time):
        self.info_label.config(text="")
        self.current_path = path
        self.current_visited = visited
        self.current_time = train_time
        self.solved = path is not None
        self.running = False
        self.animate(0)

    def compare_all(self):
        if self.running or self.animation_id:
            return
        if not self.start or not self.end:
            self.info_label.config(text="Không tìm thấy điểm S/E!")
            return

        algorithms = [
            ("BFS", solve_bfs),
            ("DFS", solve_dfs),
            ("Bidirectional BFS", solve_bidirectional_bfs),
            ("IDDFS", solve_iddfs),
            ("A* Manhattan", lambda m,s,e: solve_astar_adaptive(m,s,e,'manhattan')),
            ("A* Euclidean", lambda m,s,e: solve_astar_adaptive(m,s,e,'euclidean')),
            ("A* Chebyshev", lambda m,s,e: solve_astar_adaptive(m,s,e,'chebyshev')),
            ("A* Octile", lambda m,s,e: solve_astar_adaptive(m,s,e,'octile')),
            ("Greedy", solve_greedy),
            ("JPS", solve_jps),
            ("Dijkstra", solve_dijkstra_optimized)
        ]
        
        results = []
        for name, func in algorithms:
            path, visited, t = func(self.maze, self.start, self.end)
            results.append({
                'name': name,
                'visited': len(visited),
                'path_length': len(path) if path else "Không có",
                'time': t,
                'found': path is not None
            })
        
        popup = tk.Toplevel(self.root)
        popup.title("📊 So Sánh Tất Cả Thuật Toán")
        popup.geometry("700x500")
        
        tree = ttk.Treeview(popup, columns=("Số ô duyệt", "Độ dài đường", "Thời gian (ms)", "Kết quả"),
                           show="headings", height=15)
        
        for col in ("Số ô duyệt", "Độ dài đường", "Thời gian (ms)", "Kết quả"):
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        results.sort(key=lambda x: x['time'] if isinstance(x['time'], float) else float('inf'))
        
        for result in results:
            status = "✅" if result['found'] else "❌"
            tree.insert("", tk.END, text=result['name'],
                       values=(result['visited'], result['path_length'], 
                              f"{result['time']:.2f}" if isinstance(result['time'], float) else result['time'],
                              status))
        
        tree.pack(padx=10, pady=10)
        
        best = results[0]
        tk.Label(popup, text=f"🏆 Tốt nhất: {best['name']} - {best['time']:.2f}ms", 
                font=("Arial", 12, "bold"), fg="green").pack()
        
        tk.Button(popup, text="Đóng", command=popup.destroy).pack(pady=5)

    def animate(self, step):
        visited = self.current_visited
        path = self.current_path
        if visited is None:
            return

        delay = max(1, 101 - self.speed_var.get())

        if not self.solved:
            if step < len(visited):
                self.draw_maze(highlight_visited=visited[:step + 1])
                self.animation_id = self.root.after(delay, lambda: self.animate(step + 1))
            else:
                self.draw_maze(highlight_visited=visited)
                self.info_label.config(text="Map không thể giải!")
                self.animation_id = None
        else:
            if step < len(visited):
                self.draw_maze(highlight_visited=visited[:step + 1])
                self.animation_id = self.root.after(delay, lambda: self.animate(step + 1))
            elif step < len(visited) + len(path):
                path_step = step - len(visited)
                self.draw_maze(highlight_visited=visited, highlight_path=path[:path_step + 1])
                self.animation_id = self.root.after(delay, lambda: self.animate(step + 1))
            else:
                self.draw_maze(highlight_visited=visited, highlight_path=path)
                self.info_label.config(
                    text=f"Đã duyệt: {len(visited)} ô | "
                         f"Đường đi: {len(path)} bước | "
                         f"Thời gian: {self.current_time:.2f} ms")
                self.animation_id = None

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGUI(root)
    root.mainloop()