# 同济交通人工智能基础课程

## 问题1：搜索🔍方法（40）（参考CS188）

### 1.1 寻找游戏作为环境的案例

4分 给出将你身边的事物（运动、游戏）抽象成为状态

### 1.2 实现基本的搜索算法

**用中文写出伪代码**
修改search.py中的不同Algorithm
寻找：第二问需要修改的地方

6分：深度优先搜索 DFS（Stack实现）
6分：广度优先搜索 BFS（Queue实现）
6分：一致代价搜索UCS（Priority Queue实现）
6分：A star 搜索

### 1.3 定义状态空间，并利用A star来启发式求解（12）
修改search agent中的corner problem
寻找：第三问需要修改的地方

6分：Corners Problem：用DFS吃掉角落的豆

6分：Corners Problem：Heuristic

### 1.4 bonus 吃掉所有的豆子

All the dots

次优搜索 Suboptimal Search

## 问题2：深度学习（30）

### 2.1 CV: 降纬预处理

mnist-NN中给出tf实现的简单CNN结构，本体要求尝试使用PAC对图片进行降纬，后使用卷积神经网络的AlexNet架构进行识别，其中激活函数自己确定

需要完成的工作
0. 将所给的tensor flow代码转换成为PyTorch，完成PAC+LeNet-5训练和识别 （2）
2. 如果使用dropout和relu在LeNet-5会提升效果吗？尝试AlexNet （2）
3. 简化模型来加快训练速度 （2） 
4. 设计更好的模型可以使用在28*28的模型中，对比0，1，2，3四种模型效果 ；利用PPT绘制模型的输入输出（2）
5. 前后的图片可视化处理 （2）

### 2.2 NLP: 投诉情绪分析
nlp中的prebert.ipynb给出完整的过程，要求完善model和训练过程的函数

（5） 去除语气词，给出词云图,对文本进行简单的分析

（5）利用pre-train的权重实现文本情感分析

#### bonus
实现complaints下的时刻表文本分析

### 2.3 GAN: 生成动漫头像

（3） 理解GAN文件夹下的文件和结果，实现DCGAN的训练过程

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
[Rainbow: Combining Improvements in Deep Reinforcement Learning.](https://arxiv.org/abs/1710.02298)

### 3.2 基于强化学习的车辆行为决策 Baseline
（10）下列给出Highway-env的DQN网络的框架，修改输入算法CNN的框架
[GitHub - Farama-Foundation/HighwayEnv: A minimalist environment for decision-making in autonomous driving](https://github.com/Farama-Foundation/HighwayEnv/tree/master)

### 3.3 对比分析
比较不同决策策略、奖励函数；输出决策轨迹
1. 比如修改决策频率
2. 根据交通知识修改奖励函数
3. 输出可视化的决策轨迹

