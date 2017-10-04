# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #print "legalmoves---", legalMoves
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print "scores---", scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #print "bestindex---", bestIndices
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        "*** YOUR CODE HERE ***"
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        food = []
        for foodpos in newFood:
            tmp_dist = manhattanDistance(foodpos, newPos)
            food.append(tmp_dist)
        food_nearest = min(food)
        successorGameState.data.score += 1/(food_nearest*food_nearest)
        newGhostStates = successorGameState.getGhostStates()
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        ghost_dist = []
        for ghostpos in ghostPositions:
            tmp_dist = manhattanDistance(ghostpos, newPos)
            ghost_dist.append(tmp_dist)
        ghost_nearest = max(ghost_dist)
        if ghost_nearest > 3:
            ghost_nearest = 3
            successorGameState.data.score += 200
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            successorGameState.data.score += 300
        successorGameState.data.score += 10*ghost_nearest
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        if 0 not in newScaredTimes and action != Directions.STOP:
            successorGameState.data.score += 1000
        if action == Directions.STOP:
            successorGameState.data.score -= 100
        capsule_locations = currentGameState.getCapsules()
        if successorGameState.getPacmanPosition() in capsule_locations:
            successorGameState.data.score += 100

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        path = self.maxvalue(gameState, 0, 0, num_agents)
        return path[1]

    def maxvalue(self, gameState, agentIndex, depth, num_agents):
		final_val = (-sys.maxint, Directions.STOP)
		for action in gameState.getLegalActions(agentIndex):
			val = self.minmax_change(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, num_agents)
			if val > final_val[0]:
				final_val = (val,action)
		return final_val

    def minvalue(self, gameState, agentIndex, depth, num_agents):
		final_val = (sys.maxint, Directions.STOP)
		legalActions = gameState.getLegalActions(agentIndex)
		for action in legalActions:
			val = self.minmax_change(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, num_agents)
			if val < final_val[0]:
				final_val = (val,action)
		return final_val
        
    def minmax_change(self, gameState, agentIndex, depth, num_agents):
    	if agentIndex >= num_agents:
    		agentIndex = 0
    		depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
    	if agentIndex == 0:
    		return self.maxvalue(gameState, agentIndex, depth, num_agents)[0]
    	else:
    		return self.minvalue(gameState, agentIndex, depth, num_agents)[0]		

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        alpha = -sys.maxint
        beta  = sys.maxint
        path = self.maxvalue(gameState, 0, 0, num_agents,alpha,beta)
        return path[1]

    def maxvalue(self, gameState, agentIndex, depth, num_agents,alpha,beta):
		final_val = (-sys.maxint, Directions.STOP)
		for action in gameState.getLegalActions(agentIndex):
			val = self.alpha_beta(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, num_agents,alpha,beta)
			if val > final_val[0]:
				final_val = (val,action)
			if final_val[0] > beta:
				return final_val
			if final_val[0] > alpha:
				alpha = final_val[0]
		return final_val

    def minvalue(self, gameState, agentIndex, depth, num_agents,alpha,beta):
		final_val = (sys.maxint, Directions.STOP)
		legalActions = gameState.getLegalActions(agentIndex)
		for action in legalActions:
			val = self.alpha_beta(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, num_agents,alpha,beta)
			if val < final_val[0]:
				final_val = (val,action)
			if final_val[0] < alpha:
				return final_val
			if final_val[0] < beta:
				beta = final_val[0]
		return final_val
        
    def alpha_beta(self, gameState, agentIndex, depth, num_agents,alpha,beta):
    	if agentIndex >= num_agents:
    		agentIndex = 0
    		depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
    	if agentIndex == 0:
    		return self.maxvalue(gameState, agentIndex, depth, num_agents,alpha,beta)[0]
    	else:
    		return self.minvalue(gameState, agentIndex, depth, num_agents,alpha,beta)[0]	
        util.raiseNotDefined()
        
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        path = self.maxvalue(gameState, 0, 0, num_agents)
        return path[1]

    def maxvalue(self, gameState, agentIndex, depth, num_agents):
		final_val = (-sys.maxint, Directions.STOP)
		for action in gameState.getLegalActions(agentIndex):
			val = self.expecti_minimax(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, num_agents)
			if val > final_val[0]:
				final_val = (val,action)
		return final_val

    def chance_minvalue(self, gameState, agentIndex, depth, num_agents):
		final_val = (sys.maxint, Directions.STOP)
		final_val_sum = 0
		legalActions = gameState.getLegalActions(agentIndex)
		for action in legalActions:
			val = self.expecti_minimax(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, num_agents)
			final_val_sum += val
			if val < final_val[0]:
				final_val = (val,action)
		final_val_prob = final_val_sum/len(legalActions) #probability
		return (final_val_prob,action)
        
    def expecti_minimax(self, gameState, agentIndex, depth, num_agents):
    	if agentIndex >= num_agents:
    		agentIndex = 0
    		depth += 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
    	if agentIndex == 0:
    		return self.maxvalue(gameState, agentIndex, depth, num_agents)[0]
    	else:
    		return self.chance_minvalue(gameState, agentIndex, depth, num_agents)[0]	
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
