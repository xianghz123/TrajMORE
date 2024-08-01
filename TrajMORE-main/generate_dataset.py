import json
import pandas as pd
from utils import Timer
from joblib import Parallel, delayed
from traj2grid import Traj2Grid
import traj_dist.distance as tdist
from parameters import *
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import math


timer = Timer()


file_path = "data/full/gps_20161103"
dict_path = "data/grid/str_grid2idx_400_31925.json"
nrows = 600000
vocab_size = 400



timer.tik("read data")
df = pd.read_csv(file_path, names=["name", "id", "time", "lon", "lat"],
                 usecols=["id", "time", "lon", "lat"], nrows=nrows)
timer.tok("read {}".format(file_path))

pad = 0.002


print(df.head())


l = len(df)
df = df[(df["lon"] > 104.04214 + pad) & (df["lon"] < 104.12958 - pad)]
df = df[(df["lat"] > 30.65294 + pad) & (df["lat"] < 30.72775 - pad)]
print(f"剩{len(df)}/{l}个点，筛掉{round(100 - 100 * len(df) / l)}%")

if len(df) == 0:
    raise ValueError("筛选条件过于严格，没有剩余数据点。请检查筛选条件和数据内容。")

# 加载字典并初始化 Traj2Grid
str_grid2idx = json.load(open(dict_path))
t2g = Traj2Grid(row_num, column_num, min_lon, min_lat, max_lon, max_lat)
grid2idx = {eval(g): str_grid2idx[g] for g in list(str_grid2idx)}
t2g.set_vocab(grid2idx)
timer.tok(f"load dict{dict_path}")


def haversine(coord1, coord2):
    R = 6371.0  
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def calculate_avg_distance(coords):
    avg_distances = []
    for i in range(len(coords)):
        if i == 0:
            dist = haversine(coords[i], coords[i + 1]) if len(coords) > 1 else 0
        elif i == len(coords) - 1:
            dist = haversine(coords[i], coords[i - 1]) if len(coords) > 1 else 0
        else:
            dist1 = haversine(coords[i], coords[i - 1])
            dist2 = haversine(coords[i], coords[i + 1])
            dist = (dist1 + dist2) / 2
        avg_distances.append(dist)
    return avg_distances

def calculate_avg_speed(time, avg_distances):
    avg_speeds = []
    for i in range(len(time)):
        if i == 0:
            time_diff = time[i + 1] - time[i] if len(time) > 1 else 0
        elif i == len(time) - 1:
            time_diff = time[i] - time[i - 1] if len(time) > 1 else 0
        else:
            time_diff = (time[i + 1] - time[i - 1]) / 2.0
        avg_speed = avg_distances[i] / (time_diff / 2.0) if time_diff != 0 else 0
        avg_speeds.append(avg_speed)
    return avg_speeds

def calculate_forward_angles(coords):
    forward_angles = []
    for i in range(len(coords) - 1):
        lon_diff = coords[i + 1][1] - coords[i][1]
        lat_diff = coords[i + 1][0] - coords[i][0]
        distance = math.sqrt(lon_diff ** 2 + lat_diff ** 2)
        if distance != 0:
            cos_theta = lon_diff / distance
            forward_angles.append(cos_theta)
        else:
            forward_angles.append(0)
    forward_angles.append(0)  
    return forward_angles


def group_concat(group: pd.DataFrame):
    origin_traj = [((row["lon"], row["lat"], int(row["time"]))) for _, row in group.iterrows()]  # 将时间转换为整数
    traj_1d, coord_traj = t2g.convert1d(origin_traj)
    lon, lat, time = zip(*origin_traj)  # 解包 origin_traj

  
    coords = list(zip(lat, lon))
    avg_distances = calculate_avg_distance(coords)

   
    avg_speeds = calculate_avg_speed(time, avg_distances)

    
    forward_angles = calculate_forward_angles(coords)

    series = pd.Series({
        "origin_trajs": [(lon[i], lat[i], time[i], avg_distances[i], avg_speeds[i], forward_angles[i]) for i in range(len(origin_traj))],
        "trajs": traj_1d,
        "len": len(traj_1d),
        'start_time': group["time"].min(),
        'end_time': group["time"].max(),
    })
    return series


res = Parallel(n_jobs=44)(delayed(group_concat)(group) for name, group in df.groupby("id"))
df = pd.DataFrame(res)
timer.tok("group-apply")


if 'origin_trajs' in df.columns:
    print("轨迹数据列存在。")
else:
    print("轨迹数据列不存在。请检查数据处理步骤。")


print(df.head())


dff = df[(df["len"] >= 0)]
print(f"剩{len(dff)}/{len(df)}条轨迹，筛掉{round(100 - 100 * len(dff) / len(df))}%")


origin_trajs = dff["origin_trajs"].to_list()
arr = [np.array(origin_traj) for origin_traj in origin_trajs]
length = len(arr)
dis_matrix = np.zeros((length, length))
dis_func_name = "discret_frechet"
dis_func = getattr(tdist, dis_func_name)

def cal_dis(i, j, x, y, n):
    dis = dis_func(x[:, :2], y[:, :2])
    if i == j + 1 and i % 100 == 1:
        timer.tok(f'{i}-{round((i * i) / (n * n) * 100, 2)}%')
    return i, j, dis

res = Parallel(n_jobs=44)(delayed(cal_dis)(i, j, arr[i], arr[j], length - 1) for i in range(length) for j in range(i))
timer.tok("calculate distance")
for (i, j, dis) in res:
    dis_matrix[i, j] = dis
    dis_matrix[j, i] = dis


isForTrain = True
file_name = file_path.split("/")[-1]
save_path = "data/valid/"
file_path = save_path + file_name
origin_trajs = dff["origin_trajs"].to_list()

if isForTrain:
    dict_save = {'trajs': dff["trajs"].to_list(), 'origin_trajs': origin_trajs, "dis_matrix": dis_matrix.tolist()}
    json.dump(dict_save, open(file_path + f"_{len(origin_trajs)}_{vocab_size}_{dis_func_name}_dataset.json", "w"))
else:
    df_save = df[['len','start_time','end_time','origin_trajs']]
    df_save.to_csv(file_path + f"_{len(origin_trajs)}_info.csv", index=False)
timer.tok("save")
