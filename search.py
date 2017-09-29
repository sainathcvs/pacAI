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
    pathToGoal = {}

    while not fringeList.isEmpty():
        currState = fringeList.pop()
        if currState in visitedList:
          continue

        visitedList[currState] = 1
        if problem.isGoalState(currState):
            return extractDirections(problem, currState, pathToGoal)
        successors = problem.getSuccessors(currState)
        count = 0
        for sc in successors:
            if sc[0] not in visitedList:
                fringeList.push(sc[0])
                pathToGoal[sc[0]] = (currState, sc[1])
                count += 1
    return []


def extractDirections(problem, goalState, pathToGoal):
    directions = []
    if goalState in pathToGoal:
        parent = pathToGoal[goalState]
    else:
        return directions
    while parent[0] in pathToGoal:
        directions.append(parent[1])
        parent = pathToGoal[parent[0]]
    directions.append(parent[1])
    return directions[::-1]


def extractDirectionsN(problem, goalState, pathToGoal):
    directions = []
    if goalState in pathToGoal:
        parent = pathToGoal[goalState]
    else:
        return directions
    directions.append(goalState[1])
    while parent in pathToGoal:
        directions.append(parent[1])
        parent = pathToGoal[parent]
    return directions[::-1]

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Get start state, push it to stack and mark it as visited.
    start_state = problem.getStartState()
    startState = (start_state,'','')
    fringeList = util.Queue()
    fringeList.push(startState)
    visitedList = {}
    pathToGoal = {}

    while not fringeList.isEmpty():
        # Pop the first node from Queue
        currState = fringeList.pop() 
        
        if currState[0] in visitedList:
          continue
        visitedList[currState[0]] = 1
        if problem.isGoalState(currState[0]):
          return extractDirectionsN(problem, currState, pathToGoal)

        successors = problem.getSuccessors(currState[0])
        for sc in successors:
            # push all the nodes that are not visited into Queue
            if sc[0] not in visitedList:
              fringeList.push(sc)
              pathToGoal[sc] = currState
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    startState = (problem.getStartState(),'',0)
    fringeList = util.PriorityQueue()
    fringeList.push(startState, 0)
    visitedList = {}    
    pathToGoal = {}
    updatedCost = {}
    updatedCost[startState] = 0
    while fringeList.isEmpty() == False:
        #Pop the first node from Queue
        currState = fringeList.pop()
        if currState[0] in visitedList:
          continue

        visitedList[currState[0]]=1        
        if problem.isGoalState(currState[0]):
          return extractDirectionsN(problem, currState, pathToGoal)
        successors = problem.getSuccessors(currState[0]) 
        #print "successors ",successors
        for st in successors:
          #push all the nodes that are not visited into Queue
          if st[0] not in visitedList: 
            updatedCost[st] = updatedCost[currState] + st[2] 
            fringeList.push(st, updatedCost[st])
            #print "pushed node to queue ", st, "cost is ", currState[2] ,"  ", st[2] 
            pathToGoal[st] = currState
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
    startState = (problem.getStartState(),'',0)
    fringeList = util.PriorityQueue()
    fringeList.push(startState, 0)
    visitedList = {}    
    pathToGoal = {}
    updatedCost = {}
    updatedCost[startState] = 0
    while fringeList.isEmpty() == False:
        #Pop the first node from Queue
        currState = fringeList.pop()
        if currState[0] in visitedList:
          continue

        visitedList[currState[0]]=1        
        if problem.isGoalState(currState[0]):
          return extractDirectionsN(problem, currState, pathToGoal)
        successors = problem.getSuccessors(currState[0]) 
        #print "successors ",successors
        for st in successors:
          #push all the nodes that are not visited into Queue
          if st[0] not in visitedList: 
            updatedCost[st] = updatedCost[currState] + st[2] 
            fringeList.push(st, updatedCost[st] + heuristic(st[0], problem) )
            #print "pushed node to queue ", st, "cost is ", currState[2] ,"  ", st[2] 
            pathToGoal[st] = currState
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
