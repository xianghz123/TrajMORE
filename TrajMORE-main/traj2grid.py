import utils
import numpy as np


class Traj2Grid:
    def __init__(self, m, n, min_lon, min_lat, max_lon, max_lat, grid2idx=None):
        self.grid2idx = {}
        if grid2idx:
            self.grid2idx = grid2idx
        self.row_num = m
        self.column_num = n
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.h = (max_lat - min_lat) / m
        self.l = (max_lon - min_lon) / n
        p0 = (min_lat, min_lon)
        p1 = (min_lat + (max_lat - min_lat) / m, min_lon)
        p2 = (min_lat, min_lon + (max_lon - min_lon) / n)

    def point2grid(self, point):
        # point : (lon, lat)
        return (
            int((point[0] - self.min_lon) // self.l),
            int((point[1] - self.min_lat) // self.h),
        )

    def build_vocab(self, grid_count: dict, lower_bound=1):
        self.grid2idx.clear()
        idx = 0
        for grid in grid_count:
            if grid_count[grid] >= lower_bound:
                self.grid2idx[grid] = idx
                idx += 1
        return self.grid2idx

    def set_vocab(self, grid2idx):
        self.grid2idx = grid2idx

    def convert1d(self, original_traj, diff=True):
        traj_1d = []
        coord_traj = []
        for p in original_traj:
            idx = self.grid2idx.get(self.point2grid(p))
            if idx:
                if diff:
                    if not traj_1d or idx != traj_1d[-1]:
                        traj_1d.append(idx)
                        coord_traj.append(p)
                else:
                    traj_1d.append(idx)
                    coord_traj.append(p)
        return traj_1d, coord_traj

    def draw_grid(self, grid_count: dict, file_name="grids.png"):
        from PIL import Image

        img = Image.new("RGB", (self.row_num, self.column_num))
        mean = np.mean(list(grid_count.values()))
        std = np.std(list(grid_count.values()))
        for grid in self.grid2idx:
            percent = 50 * (grid_count[grid] - mean) / std + 50
            if percent < 50:
                green = 255
                red = percent * 5.12
            else:
                red = 255
                green = 256 - (percent - 50) * 5.12
            color = (int(red), int(green), 0, 100)
            img.putpixel((grid[0], grid[1]), color)
        img = img.resize((800, 800))
        img.save(file_name)

