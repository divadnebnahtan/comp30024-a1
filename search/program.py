from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor, BOARD_N
from typing import Iterable
from collections import deque
import heapq

# length: BOARD_N * BOARD_N
# 0 = empty, positive = red, negative = blue, abs = height
BoardState = tuple[int, ...]

class ParentState:
    def __init__(self, prev_state: BoardState | None, action: Action | None):
        self.prev_state = prev_state
        self.action = action

    def __eq__(self, other):
        return self.prev_state == other.prev_state and self.action == other.action


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_N and 0 <= col < BOARD_N


def any_blue(state: BoardState) -> bool:
    return any(height < 0 for height in state)


# kuhn's algorithm - maximum cardinality bipartite matching
# uses dfs
def kuhn(row: int, adjacency_list: list[list[int]], max_matching: list[int], seen: list[bool]) -> bool:
    for c in adjacency_list[row]:
        if seen[c]:
            continue
        
        seen[c] = True
        
        if max_matching[c] == -1 or kuhn(max_matching[c], adjacency_list, max_matching, seen):
            max_matching[c] = row
            return True
        
    return False


# number of blue stacks where no two share a row or col
def heuristic(state: BoardState) -> int:
    # row -> cols with blue stacks
    # stack only has one row and one column therefore graph is bipartite
    adjacency_list: list[list[int]] = [[] for _ in range(BOARD_N)]
    for idx, height in enumerate(state):
        if height < 0:
            row, col = divmod(idx, BOARD_N)
            adjacency_list[row].append(col)

    # maximum matching row -> col
    max_matching = [-1] * BOARD_N
    
    matched_rows = 0
    
    for row in range(BOARD_N):
        seen = [False] * BOARD_N
        if kuhn(row, adjacency_list, max_matching, seen):
            matched_rows += 1
    
    return matched_rows


def run_move(state: BoardState, src_idx: int, dir: Direction) -> BoardState | None:
    src_height = state[src_idx]
    # only red stacks can move
    if src_height <= 0:
        return None
    
    src_row, src_col = divmod(src_idx, BOARD_N)
    dst_row, dst_col = src_row + dir.r, src_col + dir.c
    if not in_bounds(dst_row, dst_col):
        return None
    
    dst_idx = dst_row * BOARD_N + dst_col
    dst_height = state[dst_idx]
    # can't move into blue stack
    if dst_height < 0:
        return None
    
    grid = list(state)
    grid[src_idx] = 0
    grid[dst_idx] = src_height + dst_height # merge

    return tuple(grid)


def run_eat(state: BoardState, src_idx: int, dir: Direction) -> BoardState | None:
    src_height = state[src_idx]
    # only red stacks can eat
    if src_height <= 0:
        return None
    
    src_row, src_col = divmod(src_idx, BOARD_N)
    dst_row, dst_col = src_row + dir.r, src_col + dir.c
    if not in_bounds(dst_row, dst_col):
        return None
    
    dst_idx = dst_row * BOARD_N + dst_col
    dst_height = state[dst_idx]
    # can't eat red stack
    if dst_height >= 0:
        return None
    # can't eat taller stack
    if src_height < -dst_height:
        return None
    
    grid = list(state)
    grid[src_idx] = 0
    grid[dst_idx] = src_height
    
    return tuple(grid)


def cascade_push_helper(grid: list[int], start_idx: int, dir: Direction) -> None:
    src_height = grid[start_idx]
    
    # nothing to push
    if src_height == 0:
        return
    
    src_row, src_col = divmod(start_idx, BOARD_N)
    dst_row, dst_col = src_row + dir.r, src_col + dir.c
    grid[start_idx] = 0
    
    if not in_bounds(dst_row, dst_col):
        return
    
    dst_idx = dst_row * BOARD_N + dst_col
    
    # push next cell too if occupied
    if grid[dst_idx] != 0:
        cascade_push_helper(grid, dst_idx, dir)
    
    # put piece down after pushing everything else
    if grid[dst_idx] == 0:
        grid[dst_idx] = src_height


def run_cascade(state: BoardState, src_idx: int, dir: Direction) -> BoardState | None:
    # assume: cascades don't merge pushed stacks
    src_height = state[src_idx]
    # only red >= 2 can cascade
    if src_height < 2:
        return None
    
    src_row, src_col = divmod(src_idx, BOARD_N)
    grid = list(state)
    grid[src_idx] = 0

    for dist in range(1, src_height + 1):
        dst_row = src_row + dir.r * dist
        dst_col = src_col + dir.c * dist
        
        if not in_bounds(dst_row, dst_col):
            continue
        
        dst_idx = dst_row * BOARD_N + dst_col
        dst_height = grid[dst_idx]
        
        if dst_height != 0:
            cascade_push_helper(grid, dst_idx, dir)
        
        # put first piece down after pushing everything else
        grid[dst_idx] = 1

    return tuple(grid)


def next_states(state: BoardState) -> Iterable[tuple[Action, BoardState]]:
    red_indices = [idx for idx, height in enumerate(state) if height > 0]
    for src_idx in red_indices:
        src_row, src_col = divmod(src_idx, BOARD_N)
        src = Coord(src_row, src_col)
        
        for dir in Direction:
            next_state = run_eat(state, src_idx, dir)
            if next_state is not None:
                yield EatAction(src, dir), next_state
            
            next_state = run_move(state, src_idx, dir)
            if next_state is not None:
                yield MoveAction(src, dir), next_state
            
            next_state = run_cascade(state, src_idx, dir)
            if next_state is not None:
                yield CascadeAction(src, dir), next_state


def reverse_actions(parent: dict[BoardState, ParentState], goal: BoardState) -> list[Action]:
    actions = []
    current_state = goal
    
    while True:
        parent_state = parent[current_state]
        if parent_state.prev_state is None:
            break
        
        actions.append(parent_state.action)
        current_state = parent_state.prev_state
        
    actions.reverse()
    return actions

def astar(start: BoardState) -> list[Action] | None:
    if not any_blue(start):
        return []

    g_score = {start: 0}
    parent = {start: ParentState(None, None)}

    counter = 0
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic(start), 0, counter, start))

    while priority_queue:
        _priority, g_popped, _tiebreak, state = heapq.heappop(priority_queue)

        g_best = g_score.get(state)
        if g_best is None or g_popped != g_best:
            continue

        if not any_blue(state):
            return reverse_actions(parent, state)

        next_g = g_best + 1
        for action, nxt in next_states(state):
            old = g_score.get(nxt)
            if old is not None and old <= next_g:
                continue
            
            g_score[nxt] = next_g
            parent[nxt] = ParentState(state, action)
            counter += 1
            heapq.heappush(priority_queue, (next_g + heuristic(nxt), next_g, counter, nxt))

    return None

def bfs(start: BoardState) -> list[Action] | None:
    if not any_blue(start):
        return []

    parent = {start: ParentState(None, None)}
    queue = deque([start])

    while queue:
        state = queue.popleft()

        if not any_blue(state):
            return reverse_actions(parent, state)

        for action, nxt in next_states(state):
            if nxt in parent:
                continue
            
            parent[nxt] = ParentState(state, action)
            queue.append(nxt)

    return None


def encode_board(board: dict[Coord, CellState]) -> BoardState:
    grid = [0] * (BOARD_N * BOARD_N)
    
    for coord, cell in board.items():
        if cell.color is None:
            continue
        
        idx = coord.r * BOARD_N + coord.c
        mult = 1 if cell.color == PlayerColor.RED else -1
        grid[idx] = cell.height * mult
    
    return tuple(grid)


def search(board: dict[Coord, CellState]) -> list[Action] | None:
    start = encode_board(board)
    return bfs(start)
    # return astar(start)
