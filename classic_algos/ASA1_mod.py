from heapq import heappop, heappush
from math import ceil, sqrt
import random
from symbol import atom
import time


class Site:
    def __init__(self, coords, prev=None):
        self.state = coords
        self.set = False
        self.prev = prev

    def l1_dist(self, other: tuple[int, int]):
        return abs(self.state[0] - other[0]) + abs(self.state[1] - other[1])

class ASAMod1Planner:
    def __init__(self, n_rows: int, n_cols: int, target_sites: list[tuple[int, int]]):
        self.max_r = n_rows - 1
        self.max_c = n_cols - 1
        self.target_sites = target_sites
        self.target_set = set(target_sites)

        self.layers = []
        self.grid = [[0 for j in range(n_cols)] for i in range(n_rows)]
        for site in target_sites:
            self.set_type(site, -1)

        self.table = []
        self.atom_sites = None

        self.identify_layers()

    def get_adjacents(self, site: tuple[int, int]) -> list[tuple[int, int]]:
        out = []
        for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            new_x, new_y = site[0] + dx, site[1] + dy
            if new_x < 0 or new_y < 0 or new_x > self.max_r or new_y > self.max_c:
                continue
            out.append((new_x, new_y))
        return out

    def get_type(self, site_coord):
        return self.grid[site_coord[0]][site_coord[1]]

    def set_type(self, site_coord, type):
        self.grid[site_coord[0]][site_coord[1]] = type

    def identify_layers(self):
        current_sites = self.target_sites

        # While sites yet to be identified
        while len(current_sites) > 0:
            next_sites = []
            identified_sites = []

            for site in current_sites:
                marked = False
                # Check if site is surrounded by internal sites
                for adj_site in self.get_adjacents(site):
                    if self.get_type(adj_site) != -1:
                        identified_sites.append(site)
                        marked = True
                        break

                if not marked:  # Not in current layer
                    next_sites.append(site)

            current_sites = next_sites
            for site in identified_sites:
                self.set_type(site, 0)
            self.layers.append(identified_sites)

    def retrace(self, start, end, parent):
        if end == start:
            return None

        if PRINT:
            print(end, " -> ", start)

        # Retrace back to start
        moves = [(start, end)]
        # curr = end
        # while curr != start:
        #     prev = parent[curr]
        #     moves.append((curr, prev, 0))
        #     curr = prev

        # moves.append((start, start, -1))
        return moves

    def simple_heuristic(self, next_site):
        min_dist = self.max_r + self.max_c
        for site in self.atom_sites:
            dist = abs(next_site[0] - site[0]) + abs(next_site[1] - site[1])
            if site in self.target_set:
                dist += 3
            if dist > min_dist:
                # return min_dist
                continue
            min_dist = dist
        return min_dist

    def astar_search(self, start_site):
        visited = set()
        parent = {}
        heap = []
        heappush(heap, (0, 0, start_site, None))
        while len(heap) > 0:
            f_curr, g_curr, curr, prev = heappop(heap)

            if curr in visited:
                continue

            visited.add(curr)
            parent[curr] = prev

            if self.get_type(curr) == 1:    # Found reservoir
                self.set_type(curr, 0)
                self.atom_sites.remove(curr)
                return self.retrace(start_site, curr, parent)

            g_adj = g_curr + 1
            for adj_site in self.get_adjacents(curr):
                if self.get_type(adj_site) == -1 or adj_site in visited:
                    continue

                f_adj = g_adj + self.simple_heuristic(adj_site)
                heappush(heap, (f_adj, g_adj, adj_site, curr))


    def get_moves(self, atom_sites: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
        if len(atom_sites) < len(self.target_sites):
            raise ValueError("Not enough source traps")

        for site in atom_sites:
            self.set_type(site, 1)

        random.shuffle(atom_sites)
        self.atom_sites = set(atom_sites)

        if PRINT:
            for row in self.grid:
                print(row)

        all_moves = []
        num_moves = 0
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]

            for site in layer:
                moves = self.astar_search(site)

                if moves is not None:
                    all_moves.extend(moves)
                    num_moves += 1

            for site in layer:
                self.set_type(site, -1)

        return all_moves, num_moves

    def visualize_layers(self):
        grid = [[0 for i in range(self.max_c + 1)] for j in range(self.max_r + 1)]

        for i, layer in enumerate(self.layers):
            level = i + 1
            for r, c in layer:
                grid[r][c] = level

        for row in grid:
            print(row)

PRINT = False

def main(n_traps):
    N_len = ceil(sqrt(n_traps))
    n_rows = ceil(sqrt(n_traps * 2))
    n_cols = n_rows

    print(f"Running for {n_traps} traps with {N_len} length in {n_rows} matrix")

    shift = int((n_rows - N_len) / 2)
    targets = []
    for i in range(N_len):
        for j in range(N_len):
            targets.append((i+shift, j+shift))

    # targets = [(2, 5), (3, 4), (3, 5), (3, 6), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 4), (5, 5), (5, 6), (6, 5)]

    arr_planner = ASAPlanner(n_rows, n_cols, targets)

    if PRINT:
        arr_planner.visualize_layers()

        print()
        print('-'*100)
        print()

    # random.seed(n_rows * n_cols)

    atom_sites = []
    for i in range(n_rows):
        for j in range(n_cols):
            if random.random() < 0.5:
                atom_sites.append((i, j))

    start = time.time_ns()
    all_moves, num_moves = arr_planner.get_moves(atom_sites)
    end = time.time_ns()

    print(f"Estimated moves: {len(all_moves)}, {num_moves}")
    return end - start


if __name__ == '__main__':
    RUNS = 1
    N_TRAPS = 1000

    times = []
    for i in range(RUNS):
        tm = main(N_TRAPS)
        times.append(tm)

    avg = sum(times) / len(times)
    print(f"Took {avg / 1000} mu s")