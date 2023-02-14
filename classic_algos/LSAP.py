from math import ceil, sqrt
import random
import time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class LSAPPlanner:
    def __init__(self, n_rows, n_cols, target_sites, metric):
        self.grid = [[0 for _ in range(n_cols)] for __ in range(n_rows)]
        self.metric = metric

        self.target_sites = target_sites

    def split_moves(self, atom_sites, moves):
        for r, c in atom_sites:
            self.grid[r][c] = 1

        grid = self.grid
        split_moves = []
        for src, dst in moves:
            src_r, src_c = src
            dst_r, dst_c = dst
            x_sign = -1 if src_c < dst_c else 1
            y_sign = -1 if src_r < dst_r else 1

            # Count collisions for horizontal move first
            hor_count = 0
            for i in range(dst_c, src_c+x_sign, x_sign):
                hor_count += grid[dst_r][i]
            for j in range(dst_r, src_r+y_sign, y_sign):
                hor_count += grid[j][i]

            # Count for vertical move
            ver_count = 0
            for j in range(dst_r, src_r+y_sign, y_sign):
                ver_count += grid[j][dst_c]
            for i in range(dst_c, src_c+x_sign, x_sign):
                ver_count += grid[j][i]

            # Break moves along the path with fewer
            if ver_count > hor_count:
                c_r, c_c = dst_r, dst_c
                adjust = False
                for i in range(dst_c, src_c+x_sign, x_sign):
                    if grid[dst_r][i] == 1:
                        split_moves.append(((dst_r, i), (c_r, c_c)))
                        c_c = i

                if c_r == src_r:
                    continue
                # if c_c != src_c:
                #     adjust = True
                #     temp = c_c
                #     c_c = src_c

                for j in range(dst_r, src_r+y_sign, y_sign):
                    if grid[j][i] == 1:
                        split_moves.append(((j, i), (c_r, c_c)))
                        c_r = j

                # if adjust:
                #     split_moves.append(((dst_r, src_c), (dst_r, temp)))
            else:
                c_r, c_c = dst_r, dst_c
                adjust = False
                for j in range(dst_r, src_r+y_sign, y_sign):
                    if grid[j][dst_c] == 1:
                        split_moves.append(((j, dst_c), (c_r, c_c)))
                        c_r = j

                if c_c == src_c:
                    continue
                # if c_r != src_r:
                #     adjust = True
                #     temp = c_r
                #     c_r = src_r

                for i in range(dst_c, src_c+x_sign, x_sign):
                    if grid[j][i] == 1:
                        split_moves.append(((j, i), (c_r, c_c)))
                        c_c = i

                # if adjust:
                #     split_moves.append(((src_r, dst_c), (temp, dst_c)))

            # Reflect move in the grid
            grid[src_r][src_c] = 0
            grid[dst_r][dst_c] = 1
        return split_moves

    def merge_moves(self, moves):
        ind_dict = {}
        for i, move in enumerate(moves):
            src, dst = move

            if src in ind_dict:
                # Check collision and merge
                collides = False
                prev_ind = ind_dict[src]
                hor_dist = abs(src[1] - dst[1])
                ver_dist = abs(src[0] - dst[0])

                for j in range(prev_ind+1, i):
                    inter = moves[j]
                    if inter is None:
                        continue

                    # Check collision
                    if src[0] == dst[0] == inter[0]:
                        collides = (abs(inter[1] - src[1]) + abs(inter[1] - dst[1])) != hor_dist
                    elif src[1] == dst[1] == inter[1]:
                        collides = (abs(inter[0] - src[0]) + abs(inter[0] - dst[0])) != ver_dist

                    if collides:
                        break

                if not collides:
                    del ind_dict[src]
                    moves[i] = (moves[prev_ind][0], dst)   # Replace with merged move
                    moves[prev_ind] = None                 # Tombstone previous move

            ind_dict[dst] = i

        return filter(lambda x: x is not None, moves)

    def get_moves(self, atom_sites):
        if len(atom_sites) < len(self.target_sites):
            raise ValueError("Not enough source traps")

        # Calculate matrix
        cost_matrix = cdist(atom_sites, self.target_sites, self.metric)

        # Solve LSAP -> Source, Dest
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        dists = cost_matrix[row_inds, col_inds]

        # Sort and map moves
        moves_inds = sorted(zip(dists, row_inds, col_inds))
        targets = self.target_sites
        moves = filter(lambda mv: mv[0] != mv[1], map(lambda x: (atom_sites[x[1]], targets[x[2]]), moves_inds))

        # print(moves)

        # Split moves
        splitted = self.split_moves(atom_sites, moves)
        # print()
        # print(splitted)

        # Merge moves
        merged = list( self.merge_moves(splitted) )

        # print()
        # print(merged)

        return merged, len(merged)

def main(n_traps, alpha=1):
    metric = "cityblock"
    if alpha == 2:
        metric = "euclidean"

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

    arr_planner = LSAPPlanner(n_rows, n_cols, targets, metric)

    # random.seed(n_rows * n_cols)

    grid = [["_" for i in range(n_cols)] for j in range(n_rows)]
    atom_sites = []
    for i in range(n_rows):
        for j in range(n_cols):
            if random.random() < 0.7:
                atom_sites.append((i, j))
                grid[i][j] = "o"
    for row in grid:
        print(row)
    print()

    start = time.time_ns()
    all_moves, num_moves = arr_planner.get_moves(atom_sites)
    end = time.time_ns()

    print(f"Estimated moves: {len(all_moves)}, {num_moves}")
    return end - start

if __name__ == '__main__':
    RUNS = 1
    N_TRAPS = 9

    times = []
    for i in range(RUNS):
        tm = main(N_TRAPS)
        times.append(tm)

    avg = sum(times) / len(times)
    print(f"Took {avg / 1000} mu s")