# 人工智能基础课程作业

建立本仓库的核心在于：

1. 交通学院人工智能导论课程提供作业问题
2. 为师弟师妹们提供学习指导，为了帮助大家更快速的融入，这里为大家推荐一些课程和资料。

|                           课程链接                           |                           课程简介                           |                           课程备注                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [CS 188：introduction to AI](https://inst.eecs.berkeley.edu/~cs188/fa24/) | **人工智能导论基础**，从行为流派介绍如何实现帮助人们工作的人工智能。从搜索问题、约束规划问题、马尔科夫决策过程、强化学习、机器学习、神经网络等内容 |  课程内容为科普性，内容简单，其中吃豆人小游戏作为本次作业1   |
| [CS 61A: Structure and Interpretation of Computer Programs](https://cs61a.org/) | **程序构造与运行原理**，课程地位类似于同济本科学习的程序设计，这里主要以 Python 出发介绍程序设计中的函数式编程、控制语句、类和对象等等 |            推荐观看并做完课程，提升自身的代码能力            |
| [CS 285：Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) | Sergey Levine老师的**深度强化学习**基础，从深度强化学习领域的基础知识到最先进展的延伸，非常值得学习 |                 难度较高，推荐观看和写完作业                 |
|    [CS229: Machine Learning](https://cs229.stanford.edu/)    | Andrew Wu（吴恩达）老师的**机器学习课程**，主要从概率统计和数学推导介绍机器学习中模型基本原理，包括但不限于广义线性模型、聚类模型、树模型、集成学习理论等 | 非常推荐观看，同时把提供的[课程作业](https://github.com/learning511/cs229-assignments)给做了 |
|           [《动手学深度学习》](https://zh.d2l.ai/)           | 李沐老师的**深度学习课程**，基于 PyTorch 编程为大家介绍卷积神经网络、循环神经网络、注意力机制带来的计算机视觉、自然语言处理中的常见模型原理 |           非常推荐观看，也记得关注李沐老师的 B 站            |
|                           优化课程                           |                             待定                             |                             待定                             |



> [!IMPORTANT]
>
> a. 请以清晰、简洁的答案回复下列问题
>
> b. 如果你对作业有任何问题，欢迎直接提出 issue 或者联系[助教邮箱](xuchenlong796@gmail.com) 
>
> c. 除了作业本身之外，希望你可以做出更多的创新



## 问题1： [20 分] 搜索问题（Search problem）

本部分目录为[search]，更详细的材料可以阅读：https://inst.eecs.berkeley.edu/~cs188/fa24/

**搜索问题（Search problem）**是已知一个问题的初始状态和目标状态，希望找到一个操作序列来是的原问题的状态能从初始状态转移到目标状态，搜索问题的基本要素：

a.   初始状态

b.   转移函数，从当前状态输出下一个状态

c.   目标验证，是否达到最终的目标状态

d.   代价函数，状态转移需要的代驾

**搜索算法（Search algorithm）**指的是利用计算机的高性能来有目的穷举一个问题解空间的部分或者所有可能情况，从而求出对于问题的解的一种方法，常见的搜索算法包括枚举算法、深度优先、广度有限、A star算法、回溯算法、蒙特卡洛树搜索等等。搜索算法的基本过程：

a.   从初始或者目标状态出发

b.   扫描操作的算子，来得到候选状态

c.   检查是否为最终状态

a)    满足，则得到解

b)    不满足，则将新状态作为当前状态，进行搜索



### 1.1 问题描述 （Problem statement）

选择一个你身边的示例，抽象为搜索问题

*例子：[使用 BFS求解八皇后问题 *](https://leetcode.cn/problems/eight-queens-lcci/description/)

### 1.2 搜索算法 （Search algorithm implementation）

自行学习下列算法，写出中文伪代码，利用[class SearchProble]的提供的 API 基础上时在[Search.py]中利用代码实现：

\-    深度优先搜索 （Deep Search First，DFS）

\-    广度有限搜索 （Board Search First，DFS）

\-    一致代价搜索 （Uniform Cost Search，UCS）

\-    A star 搜索

### 1.3 构造智能体 （Construct search agent）
完成搜索算法基础上，修改 [SearchAgents.py] 实现吃豆人游戏（Corner problem） 的求解

\-    完善Corner problem

\-    利用 DFS 吃掉角落的豆

\-    利用启发式（heuristic） 

### 1.4 bonus 吃掉所有的豆子 （Eat them all！）



## 问题2： [20 分] 机器学习（Machine Learning）

### 2.1 离散选择模型 （DCM）

### 2.2 聚类模型（Unsupervised learning）



## 问题2： [30分] 深度学习（Deep learning）

本章节对应目录为 deeplearning，更多详细信息可以参考阅读：https://zh-v1.d2l.ai/index.html

深度学习是人工智能和机器学习的一个分支。其主要希望利用多层的人工神经网络来实现特定的任务目标，包括但不限于目标检测、语音识别、语音翻译、文本生成、图文匹配等功能。其优点是可以自动的是图像、视频、文本、音频中自动的学习特征，而不需要引入人类领域的知识。

### 2.1 计算机视觉（Computer Vision）

CV 文件夹中展现了如何利用 LeNet 训练识别 Mnist 手写数据集的任务。现在希望做的几部分优化：

a.   从数据层面，考虑增加 PCA 来对输入特征进行降维

b.   从模型调优，尝试对 baseline增加 dropout 和 relu 是否会提升模型效果

c.   从模型对比，参考AlextNet网络架构- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 更新识别模型，并介绍 AlexNet 对比 LeNet 的具体改进有哪些

*PS：可以考虑使用PyTorch来替换上述的Tensorflow代码*

### 2.2 自然语言处理（Natural language processing）
NLP 文件夹下的 prebert.ipynb 初步完成了基于transformer 库的 BERT-[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 微调方案来实现情绪识别的方案。其中数据集处理、模型训练已经完善，需要你完成

a.  在原有 BERT 模型上增加一层 MLP 来满足二分类任务，需要修改BertSST2Model 的信息

b.  展示训练过程的损失函数

### 2.3 生成对抗网络（Generative adversarial network）

利用生成对抗网络 GAN - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) 生成动漫头像，本题任务有

a.   理解GAN目录文件，实现DCGAN的结果

b.   观察模型坍塌（Model callapse） 的方法，并总结常见的解释方法

c.   阅读相关文献并尝试对baseline 进行修改



## 问题3：[20 分] 深度强化学习（DRL，Deep reinforcement learning）

深度强化学习侧重于如何基于环境行动并最大化预期收益，随着深度学习发展，原有的强化学习算法中的智能体可以有更高的感知能力，配合自身算法的“探索-利用”来实现对复杂问题的决策和解决。

### 3.1 论文阅读（Proof reading）

DQN 是经典的深度强化学习算法，后人对其有诸多的改进，这里要求阅读其中[Rainbow: Combining Improvements in Deep Reinforcement Learning.](https://arxiv.org/abs/1710.02298)，增强理解

### 3.2 基准实现（Implementation of baseline）
Highway中使用Highway-env构造训练环境，现在希望能够增加 CNN 模块来实现深度强化学习框架。

### 3.3 对比分析（）
比较不同决策策略、奖励函数；输出决策轨迹
1. 比如修改决策频率
2. 根据交通知识修改奖励函数
3. 输出可视化的决策轨迹



## 问题5：[10 分] 运筹优化（Operation  research）



