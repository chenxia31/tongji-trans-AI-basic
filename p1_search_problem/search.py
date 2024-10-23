# search.py
# ---------
# 许可信息：您可以出于教育目的自由使用或扩展这些项目，前提是
# （1）您不分发或发布解决方案，
# （2）您保留本通知，
# （3）您向加州大学伯克利分校提供明确的归属，包括http://ai.berkeley.edu的链接。
# 
# 归因信息：Pacman AI项目是在加州大学伯克利分校开发的。
# 核心项目和自动分级器主要由John DeNero（denero@cs.berkeley.edu）和Dan Klein（klein@cs.berkeley.edu）创建。
# Brad Miller、Nick Hay和Pieter Abbeel（pabbeel@cs.berkeley.edu）添加了学生侧自动评分。

"""
在search.py中, 您将实现由Pacman智能体(在searchAgents.py中)调用的通用搜索算法。
"""

import util

class SearchProblem:
    """
    该类概述了搜索问题的结构，但没有实现任何方法（在面向对象的术语中：抽象类）。
    
    Attention: 不需要修改这个问题
    """

    def getStartState(self):
        """
        返回问题初始状态
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        state: 搜索状态

        判断是否到达最终状态
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        state: Search state
        

        对于给定的状态,这应该返回一个三元组列表(继任者、行动、stepCost)
        其中“继任者”是当前状态的继承者,“行动”是到达那里所需的行动,
        “stepCost”是扩展到该继承者的增量成本。
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
        actions: A list of actions to take

        此方法返回特定操作序列的总成本。该序列必须由规定动作组成。
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    返回解决tinyMaze的动作序列。对于任何其他迷宫,移动顺序将不正确,因此仅将其用于 tinyMaze。
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    DFS 搜索算法实现
    
    要开始，您可能想尝试其中一些简单的命令来了解正在传递的搜索问题：
    print("Start:", problem.getStartState()) 
    print("Is the start a goal?", problem.isGoalState(problem.getStartState())) 
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "NOTE *** 问题2开始编程的地方 ***"
    # 问题2: 实现深度优先搜索算法
    temp = problem.getStartState()
    found = []
    fringe = util.Queue()
    fringe.push((temp,[]))
    while not fringe.isEmpty():
        temp, path = fringe.pop()
        if problem.isGoalState(temp):
            return path
        if not temp in found:
            found.append(temp)
            for child in problem.getSuccessors(temp):
                if child[0] not in found:
                    fringe.push((child[0], path + [child[1]]))
    "*** 问题2结束编程的地方 ***"
    util.raiseNotDefined()


def breadthFirstSearch(problem: SearchProblem):
    """
    BFS 搜索算法实现
    """

    "NOTE *** 问题2开始编程的地方 ***"

    "*** 问题2结束编程的地方 ***"
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """
    UCS 搜索算法实现
    """
    "NOTE *** 问题2开始编程的地方 ***"

    "*** 问题2结束编程的地方 ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    启发式函数估计从当前状态到提供的SearchProblem中最近目标的成本。这种启发式是微不足道的。
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """
    A* 搜索算法实现
    """
    "NOTE *** 问题2开始编程的地方 ***"

    "*** 问题2结束编程的地方 ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
