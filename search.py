# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import searchAgents

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
'''
def depthFirstSearch(problem):
    visitedList = {}
    pathToGoal = collections.OrderedDict()
    startState = (problem.getStartState())    
    visitedList[startState] = 1
    directions = depthFirstSearch1(problem,visitedList,startState, pathToGoal)
    print "directions ", directions
    return directions    

def depthFirstSearch1(problem,visitedList, currState, pathToGoal, directions=[]):
    if (problem.isGoalState(currState) == True):
      directions = extractDirections(problem, currState, pathToGoal)
      return directions
    successors = problem.getSuccessors(currState) 
    for st in successors:
      if st[0] not in visitedList: 
        visitedList[st[0]] = 1
        pathToGoal[st[0]] = (currState, st[1])
        directions = depthFirstSearch1(problem, visitedList, st[0], pathToGoal)
        if len(directions) > 0:
          break
    print "Directions before returning ", directions
    return directions            
    
'''


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # Get start state, push it to stack and mark it as visited.
    startState = (problem.getStartState())
    fringeList = util.Stack()
    fringeList.push(startState)
    visitedList = {}
    visitedList[startState] = 1    
    pathToGoal = {}

    while not fringeList.isEmpty():
        #Peek the top node on the stack
        currState = fringeList.pop()
        fringeList.push(currState)

        if problem.isGoalState(currState):
            return extractDirections(problem, currState, pathToGoal)
        successors = problem.getSuccessors(currState)
        count = 0
        '''
        for st in successors:
            if st[0] not in visitedList:
                fringeList.push(st[0])
                visitedList[st[0]] = 1
                pathToGoal[st[0]] = (currState, st[1])
                count += 1
        '''

        #'''
        for i in range(len(successors)-1, -1, -1):
            if successors[i][0] not in visitedList:
                fringeList.push(successors[i][0])
                visitedList[successors[i][0]] = 1
                pathToGoal[successors[i][0]] = (currState, successors[i][1])
                count += 1
        #'''
        if count == 0:
            fringeList.pop()
    return []


def extractDirections(problem, goalState, pathToGoal):
    directions = []
    if goalState in pathToGoal:
        parent = pathToGoal[goalState]
    else:
        return directions
    while parent[0] in pathToGoal:
        #print parent[1]
        directions.append(parent[1])
        parent = pathToGoal[parent[0]]
    directions.append(parent[1])
    return directions[::-1]

oD = {"South":"North", "North":"South","West":"East","East":"West"}


# Returns LCA if node n1 , n2 are present in the given
# binary tre otherwise return -1
def findLCA(path1, path2, pathToGoal):
    # To store paths to n1 and n2 fromthe root
    # Compare the paths to get the first different value
    path1 = path1[::-1]
    path2 = path2[::-1]
    i = 0
    while (i < len(path1) and i < len(path2)):
        if path1[i] == path2[i]:
            break
        i += 1
    #path1[i]
    first = path1[-1]
    second = path2[-1]
    parent = pathToGoal[first]
    directions1 = []
    while parent[0][0][0] in pathToGoal:
        directions1.append(parent[1])
        parent = pathToGoal[parent[0][0][0]]
    directions1.append(parent[1])
    directions1 = directions1[::-1]
    leng = len(directions1)
    for i in xrange(leng):
        directions1.append(oD[directions1[leng-i-1]])

    directions2 = []
    parent = pathToGoal[second]
    while parent[0][0][0] in pathToGoal:
        directions2.append(parent[1])
        parent = pathToGoal[parent[0][0][0]]
    directions2.append(parent[1])
    directions2 = directions2[::-1]
    leng = len(directions2)
    for i in xrange(leng):
        directions2.append(oD[directions2[leng-i-1]])
    return directions1[:]+directions2[:]


def extractDirectionsCorners(problem, goalState, pathToGoal):
    corners = goalState[0][1]
    directions = []
    if goalState[0][0] in pathToGoal:
        parent = pathToGoal[goalState[0][0]]
    else:
        return directions
    li = range(len(corners))
    for i in li[0:len(corners):2]:
        first = corners[i]
        second = corners[i+1]
        firstPath = [first]
        secondPath = [second]
        parent = pathToGoal[first]
        while parent[0][0][0] in pathToGoal:
            firstPath.append(parent[0][0][0])
            parent = pathToGoal[parent[0][0][0]]
        firstPath.append(parent[0][0][0])
        parent = pathToGoal[second]
        while parent[0][0][0] in pathToGoal:
            secondPath.append(parent[0][0][0])
            parent = pathToGoal[parent[0][0][0]]
        secondPath.append(parent[0][0][0])
        temp = findLCA(firstPath, secondPath, pathToGoal)
        directions+= temp[:]
    return directions


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Get start state, push it to stack and mark it as visited.
    start_state = (problem.getStartState())
    if len(start_state) == 2:
        startState = start_state
    else:
        startState = start_state
    fringeList = util.Queue()
    fringeList.push(startState)
    visitedList = {}

    if len(start_state) == 2:
        visitedList[startState] = 1
    else:
        visitedList[startState[0][0]] = 1
    pathToGoal = {}

    while not fringeList.isEmpty():
        # Pop the first node from Queue
        currState = fringeList.pop()
        if problem.isGoalState(currState):
            if len(start_state) == 2:
                return extractDirections(problem, currState, pathToGoal)
            else:
                return extractDirectionsCorners(problem, currState, pathToGoal)
        successors = problem.getSuccessors(currState)
        for sc in successors:
            # push all the nodes that are not visited into Queue
            if len(start_state) == 2:
                if sc[0] not in visitedList:
                    fringeList.push(sc[0])
                    visitedList[sc[0]] = 1
                    pathToGoal[sc[0]] = (currState, sc[1])
            else:
                if sc[0][0] not in visitedList:
                    fringeList.push(sc)
                    visitedList[sc[0][0]] = 1
                    pathToGoal[sc[0][0]] = (currState, sc[1])
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    startState = (problem.getStartState())
    fringeList = util.PriorityQueue()
    fringeList.push(startState, 0)
    visitedList = {}
    visitedList[startState] = 1    
    pathToGoal = {}

    while not fringeList.isEmpty() == False:
        #Pop the first node from Queue
        currState = fringeList.pop()
        if problem.isGoalState(currState):
          return extractDirections(problem, currState, pathToGoal)
        successors = problem.getSuccessors(currState) 
        '''for st in successors:
          #push all the nodes that are not visited into Queue
          if st[0] not in visitedList: 
            fringeList.push(st[0], st[2])
            visitedList[st[0]] = 1
            pathToGoal[st[0]] = (currState, st[1])'''

        for i in range(len(successors) - 1, -1, -1):
          #push all the nodes that are not visited into Queue
            if successors[i][0] not in visitedList:
                fringeList.push(successors[i][0], successors[i][2])
                visitedList[successors[i][0]] = 1
                pathToGoal[successors[i][0]] = (currState, successors[i][1])
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startState = (problem.getStartState())
    fringeList = util.PriorityQueue()
    #fringeList.push(startState, nullHeuristic(startState, problem))
    fringeList.push(startState, searchAgents.manhattanHeuristic(startState, problem))
    visitedList = {}
    visitedList[startState] = 1    
    pathToGoal = {}

    while not fringeList.isEmpty():
        #Pop the first node from Queue
        currState = fringeList.pop()
        if problem.isGoalState(currState):
            return extractDirections(problem, currState, pathToGoal)
        successors = problem.getSuccessors(currState)
        '''
        for st in successors:
          #push all the nodes that are not visited into Queue
          if st[0] not in visitedList: 
            fringeList.push(st[0], st[2] + nullHeuristic(startState, problem))
            #fringeList.push(st[0], st[2] + searchAgents.manhattanHeuristic(startState, problem))
            visitedList[st[0]] = 1
            pathToGoal[st[0]] = (currState, st[1])
        '''

        for i in range(len(successors) - 1, -1, -1):
            if successors[i][0] not in visitedList:
                #fringeList.push(successors[i][0], successors[i][2]+nullHeuristic(startState, problem))
                fringeList.push(successors[i][0], successors[i][2] + searchAgents.manhattanHeuristic(startState, problem))
                visitedList[successors[i][0]] = 1
                pathToGoal[successors[i][0]] = (currState, successors[i][1])
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
