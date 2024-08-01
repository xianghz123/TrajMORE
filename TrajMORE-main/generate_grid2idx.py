import numpy as np
import torch
from grid2vec import *
import json
import pandas as pd
from utils import Timer, copy_file
from joblib import Parallel, delayed
from traj2grid import Traj2Grid
import traj_dist.distance as tdist
from logging import raiseExceptions
from parameters import *
import numpy as np
import time
import modin.pandas as pd
import ray
ray.init(address='auto')


timer = Timer()


row_num = 400
column_num = 400

timer = utils.Timer()
t2g = Traj2Grid(row_num, column_num, min_lon, min_lat, max_lon, max_lat)

timer.tik()
value_counts = None
for i in range(1, 2):
    df = pd.read_csv(
        f"{data_dir}/augmen/gps_201611{str(i).zfill(2)}",
        names=["name", "order_id", "time", "lon", "lat"],
        usecols=["lon", "lat"],

    )  
    timer.tok(f"read{str(i).zfill(2)}")
    df = df.apply(t2g.point2grid, axis=1).squeeze()
    timer.tok(f"apply{str(i).zfill(2)}")
    if value_counts is not None:
        value_counts = value_counts.add(df.value_counts(), fill_value=0)
    else:
        value_counts = df.value_counts()
    timer.tok(f"value_counts{str(i).zfill(2)}")
value_counts = dict(value_counts)


grid2idx = t2g.build_vocab(value_counts, lower_bound=10)
print(f"剩{len(grid2idx)}/{len(value_counts)}，筛掉{round(100 - 100 * len(grid2idx) / len(value_counts))}%")


str_grid2idx = {f"({grid[0]},{grid[1]})": grid2idx[grid] for grid in grid2idx}
json.dump(str_grid2idx, open(f"data/grid/str_grid2idx_{row_num}_{len(str_grid2idx)}.json", "w"))