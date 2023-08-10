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

### 2.1 CV: 降纬预处理

尝试使用PAC对图片进行降纬，后使用卷积神经网络的AlexNet架构进行识别，其中激活函数自己确定

需要完成的工作

1. 完成PAC+AlexNet训练和识别 （2）
2. 如果使用dropout和relu在LeNet-5会提升效果吗 （2）
3. 简化模型来加快训练速度 （2） 
4. 设计更好的模型可以使用在28*28的模型中，对比0，1，2，3四种模型效果 （2）
5. 前后的图片可视化处理 （2）

### 2.2 NLP: 投诉情绪分析

（5） 去除语气词，给出词云图

（5）利用pre-train的权重来对时刻表相关搜索进行情感分析

### 2.3 GAN: 生成动漫头像

（3）尝试写出baseline model，写出DCGAN的训练过程

（3）尝试观察得到model collapse的方法

并写出文献给出常见的解决方法

（4）利用下列模型修改baseline，对模型进行一些调整

[WGAN](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan)

[WGAN-GP](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan_gp)

LSGAN

SNGAN

## 问题3：强化学习（30）

### 3.1 论文阅读

（10） 阅读下列论文，总结DQN的不同trick

### 3.2 基于强化学习的车辆行为决策 Baseline
（10）下列给出Highway-env的DQN网络的框架，请在其中增加对应的CNN模块
同时对比不同CNN的框架来找到最好的结果
[GitHub - Farama-Foundation/HighwayEnv: A minimalist environment for decision-making in autonomous driving](https://github.com/Farama-Foundation/HighwayEnv/tree/master)

### 3.3 环境修改
在强化学习算法中已经有成功的开源库Stable- baseline3，同时一些优秀的环境，我们可以尝试修改环境来完成自己的研究
现在要求您对环境中增加障碍物来实现车辆智能化的行为决策。


