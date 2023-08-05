# 同济交通人工智能基础课程

## 问题1：搜索🔍方法（40）

### 1.1 寻找游戏作为环境的案例

4分 给出将你身边的事物（运动、游戏）抽象成为状态

### 1.2 实现基本的搜索算法

**用中文写出伪代码**

6分：深度优先搜索 DFS（Stack实现）
6分：广度优先搜索 BFS（Queue实现）

6分：一致代价搜索UCS（Priority Queue实现）

6分：A star 搜索

### 1.3 定义状态空间，并利用A star来启发式求解（12）

6分：Corners Problem：DFS用DFS吃掉角落的豆

6分：Corners Problem：Heuristic

### 1.4 bonus 吃掉所有的豆子

All the dots

次优搜索 Suboptimal Search

## 问题2：深度学习（30）

### 2.1 计算机视觉

尝试使用PAC对图片进行降纬，后使用卷积神经网络的AlexNet架构进行识别，其中激活函数自己确定

需要完成的工作

1. 完成PAC+AlexNet训练和识别 （2）
2. 如果使用dropout和relu在LeNet-5会提升效果吗 （2）
3. 简化模型来加快训练速度 （2） 
4. 设计更好的模型可以使用在28*28的模型中，对比0，1，2，3四种模型效果 （2）
5. 前后的图片可视化处理 （2）

### 2.2 自然语言处理

（5） 去除语气词，给出词云图

（5）利用pre-train的权重来对时刻表相关搜索进行情感分析

### 2.3 对抗生成网络

（3） 尝试写出baseline model

（4）利用下列模型修改baseline

WGAN

[](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan)

WGAN-GP

[](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan_gp)

LSGAN

SNGAN

（3）尝试观察得到model collapse的方法

并利用文献给出常见的解决方法

## 问题3：强化学习（30）

### 3.1 论文阅读

（10） 阅读下列论文，总结DQN的不同trick

### 3.2 Baseline

（10）给下列baseline增加CNN模块

[GitHub - Farama-Foundation/HighwayEnv: A minimalist environment for decision-making in autonomous driving](https://github.com/Farama-Foundation/HighwayEnv/tree/master)

### 3.3 提升

（10）找到最好的方法，写出你的训练过程伪代码
