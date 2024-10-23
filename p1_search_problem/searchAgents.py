# searchAgents.py
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
此文件包含所有可以选择控制Pacman的代理。要选择代理,
请在运行pacman.py时使用“-p”选项。可以使用“-a”将参数传递给您的代理。
例如,要加载使用深度优先搜索(dfs)的SearchAgent,请运行以下命令:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

调用其他搜索策略的命令可以在项目描述中找到。


请仅更改要求您更改的文件部分。寻找那些:

"**********************"
"*                    *"
"*** YOUR CODE HERE ***"
"*                    *"
"**********************"

你填写的部分大约从3/4处开始。有关详细信息,请按照项目描述进行操作。

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

class GoWestAgent(Agent):
    "一直往西走的智能体"

    def getAction(self, state):
        "根据接受到的state作出action (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# ····这一段的search agent已经写完，但是使用的前提是·····   #
#               需要填充完 search.py                    #
#######################################################

class SearchAgent(Agent):
    """
    这个非常通用的搜索代理使用提供的搜索算法为提供的搜索问题找到一条路径，然后返回遵循该路径的操作。

    默认情况下,该智能体在PositionSearchProblem上运行DFS以查找位置(1,1)
    搜索方法fn的选项包括:
        depthFirstSearch 或 dfs 
        breadthFirstSearch或bfs

    Note: 不需要修改Search Agent中的代码
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        这是智能体第一次看到游戏板的布局。
        在这里，我们选择一条通往目标的道路。
        在此阶段，代理应计算到目标的路径并将其存储在局部变量中。
        所有工作都是用这种方法完成的！

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        返回之前选择的路径中的下一个操作(在registerInitialState中)。
        返回方向。如果没有采取进一步行动，请停止。

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """

    搜索问题定义了状态空间、开始状态、目标测试、继任函数和成本函数。
    
    此搜索问题可用于查找到pacman板上特定点的路径。状态空间由pacman游戏中的(x,y)位置组成。
    Note: 此搜索问题已完全指定；您不应更改它。
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        返回继承状态、它们所需的操作以及1的成本。
        
        如search.py中所述:
        对于给定状态,这应该返回一个三元组列表(继任者、行动、stepCost)
        其中“继任者”是当前状态的继任者
        “行动”是到达那里所需的行动，
        而“stepCost”是扩展到该继任者的增量成本
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        返回特定操作序列的成本。如果这些行动包括非法移动,请返回999999。

        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# 这一部分没有写完，需要您填充     Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    此搜索问题通过布局的所有四个角落找到路径。您必须选择合适的状态空间和后继函数

    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        存储墙壁、pacman的起始位置和角落。
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "**********************"
        "*                    *"
        "*** 第三问需要编程的地方 ***"
        "*                    *"
        "**********************"
 

    def getStartState(self):
        """
        返回开始状态(在您的状态空间中,而不是完整的Pacman状态空间)
        """
        "**********************"
        "*                    *"
        "*** 第三问需要编程的地方 ***"
        "*                    *"
        "**********************"


    def isGoalState(self, state: Any):
        """
        返回此搜索状态是否是问题的目标状态。
        """
        "**********************"
        "*                    *"
        "*** 第三问需要编程的地方 ***"
        "*                    *"
        "**********************"

    def getSuccessors(self, state: Any):
        """
        返回继承状态、它们所需的操作以及1的成本。
        
        如search.py中所述:
        对于给定状态,这应该返回一个三元组列表(继任者、行动、stepCost)
        其中“继任者”是当前状态的继任者
        “行动”是到达那里所需的行动，
        而“stepCost”是扩展到该继任者的增量成本
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "**********************"
            "*                    *"
            "*** 第三问需要编程的地方 ***"
            "*                    *"
            "**********************"

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        返回特定操作序列的成本。如果这些行动包括非法移动,请返回999999。这是为你实施的。
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    您定义的CororsProblem的启发式方法。

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    此函数应始终返回一个数字，该数字是从状态到问题目标的最短路径的下界；即它应该是可接受的（以及一致的）。
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
        # 以离豆豆的曼哈顿距离的最小值作为启发值
    x, y = state[0]  # Pacman的初始位置
    heuristicAccumulator = 0  # 累加的启发值
    foodRemain = {}  # 还没吃掉的豆豆
    for i in range(len(state[1])):  # 设置初始值为0，因为后面要存放曼哈顿距离
        if state[1][i] == 0:
            foodRemain[corners[i]] = 0
    while len(foodRemain) > 0:  # 循环累加启发值
        for nextNode in foodRemain.keys():  # 算没吃掉的豆豆的曼哈顿距离
            foodRemain[nextNode] = abs(x - nextNode[0]) + abs(y - nextNode[1])
        x, y = min(foodRemain, key=lambda key: foodRemain[key])  # 最小曼哈顿距离的豆豆的坐标
        heuristicAccumulator += foodRemain[(x, y)]  # 累加启发值
        del foodRemain[(x, y)]  # 计算完后删掉这个豆豆
    return heuristicAccumulator

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "**********************"
    "*                    *"
    "*** YOUR CODE HERE ***"
    "*                    *"
    "**********************"
    return 0

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "**********************"
        "*                    *"
        "*** YOUR CODE HERE ***"
        "*                    *"
        "**********************"
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        "**********************"
        "*                    *"
        "*** YOUR CODE HERE ***"
        "*                    *"
        "**********************"
        util.raiseNotDefined()

def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
