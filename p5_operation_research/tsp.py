# 利用脚本编程的方式来一步一步实现调用GAEATpy求解单目标优化问题
import numpy as np
import geatpy as ea # 导入geatpy库
import time 
import matplotlib.pyplot as plt

# TSP 坐标点
places=[[35.0, 35.0], [41.0, 49.0], [35.0, 17.0], [55.0, 45.0], [55.0, 20.0], [15.0, 30.0],
        [25.0, 30.0], [20.0, 50.0], [10.0, 43.0], [55.0, 60.0], [30.0, 60.0], [20.0, 65.0], 
        [50.0, 35.0], [30.0, 25.0], [15.0, 10.0], [30.0, 5.0], [10.0, 20.0], [5.0, 30.0],
        [20.0, 40.0], [15.0, 60.0]]
places = np.array(places)
distance = np.array([[np.linalg.norm(places[i]-places[j],2) for i in range(len(places))] for j in range(len(places))])

Phen = [np.arange(len(places)) for i in range(10)]
for i in range(10):
    np.random.shuffle(Phen[i])
print(len(places))