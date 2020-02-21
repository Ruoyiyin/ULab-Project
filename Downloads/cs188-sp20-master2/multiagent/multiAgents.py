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
import numpy

from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print("newScaredTimes is {}".format(newScaredTimes))

        newFood = newFood.asList()
        # newGhostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        newGhostPos = [G.getPosition() for G in newGhostStates]
        min_scared = min(newScaredTimes)

        if min_scared <= 0 and newPos in newGhostPos:
            return float("-inf")

        if newPos in currentGameState.getFood().asList():
            return float("inf")

        # food distance
        food_d_min = min([manhattanDistance(f, newPos) for f in newFood])
        # ghost distance
        ghost_d_min = min([manhattanDistance(g, newPos) for g in newGhostPos])

        return 1.0/food_d_min - 1.0/ghost_d_min
        # return ghost_d_min - food_d_min


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.ghosts = numpy.arange(1, gameState.getNumAgents())
        # print("self.depth is {}".format(self.depth))
        if self.is_terminal(gameState, 0):
            return self.evaluationFunction(gameState)
        pac_actions = [action for action in gameState.getLegalActions(0)]
        # pac max
        # ghost after pac, ghost min
        # a max_value function but need to return what the action is
        v = float("-inf")
        result_action = pac_actions[0]
        for action in pac_actions:
            min_ghost = self.min_value(gameState.generateSuccessor(0, action), 1, 0)
            if min_ghost > v:
                result_action = action
                v = min_ghost
        return result_action


    def min_value(self, state, ghostIndex, d):
        """
        def min-value(state):
            initialize v = +∞
            for each successor s of state:
                v = min(v, max - value(s))
        return v
        """
        if self.is_terminal(state, d):
            return self.evaluationFunction(state)
        # loop through all ghosts
        v = float("inf")
        for action in state.getLegalActions(ghostIndex):
            if ghostIndex == self.ghosts[-1]:
                v = min(v, self.max_value(state.generateSuccessor(ghostIndex, action), d+1))
            else:
                v = min(v, self.min_value(state.generateSuccessor(ghostIndex, action), ghostIndex + 1, d))
        return v

    def max_value(self, state, d):
        """
        def max-value(state):
            initialize v = −∞.
            for each successor s of state:
                v = max(v, min - value(s))
        return v
        """
        if self.is_terminal(state, d):
            return self.evaluationFunction(state)
        v = float("-inf")
        for action in state.getLegalActions(0):
            v = max(v, self.min_value(state.generateSuccessor(0, action),  1, d))
        return v

    def is_terminal(self, state, d):
        return state.isWin() or state.isLose() or self.depth == d


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.ghosts = numpy.arange(1, gameState.getNumAgents())
        # print("self.depth is {}".format(self.depth))
        if self.is_terminal(gameState, 0):
            return self.evaluationFunction(gameState)
        pac_actions = [action for action in gameState.getLegalActions(0)]
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        result_action = pac_actions[0]
        for action in pac_actions:
            min_ghost = self.min_value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if min_ghost > v:
                result_action = action
                v = min_ghost
            # if v > beta:
            # beta is inf right??
            #     return "danl"
            alpha = max(alpha, v)
        return result_action

    def min_value(self, state, ghostIndex, d, alpha, beta):
        """
        def min-value(state , α, β):
            initialize v = +∞.
            for each successor s of state:
                v = min(v, value(s, α, β))
                if v ≤ α return v
                β = min(β,v)
            return v
        """
        if self.is_terminal(state, d):
            return self.evaluationFunction(state)
        # loop through all ghosts
        v = float("inf")
        for action in state.getLegalActions(ghostIndex):
            if ghostIndex == self.ghosts[-1]:
                v = min(v, self.max_value(state.generateSuccessor(ghostIndex, action), d + 1, alpha, beta))
            else:
                v = min(v, self.min_value(state.generateSuccessor(ghostIndex, action), ghostIndex + 1, d, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(self, state, d, alpha, beta):
        """
        alpha: max's best option on path to root
        beta: min's best option on path to root
        def max-value(state， alpha, beta):
            initialize v = −∞.
            for each successor s of state:
                v = max(v, min - value(successor, alpha, beta))
                # this layer is to choose max, if v already > beta, then
                if v > beta:
                    return v
                alpha = max(alpha, v)
        return v
        """
        if self.is_terminal(state, d):
            return self.evaluationFunction(state)
        v = float("-inf")
        for action in state.getLegalActions(0):
            v = max(v, self.min_value(state.generateSuccessor(0, action), 1, d, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def is_terminal(self, state, d):
        return state.isWin() or state.isLose() or self.depth == d


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state

        ###
        an adversary which chooses amongst their getLegalActions uniformly at random.
        ###
        """
        "*** YOUR CODE HERE ***"

        self.ghosts = numpy.arange(1, gameState.getNumAgents())
        # print("self.depth is {}".format(self.depth))
        if self.is_terminal(gameState, 0):
            return self.evaluationFunction(gameState)
        pac_actions = [action for action in gameState.getLegalActions(0)]
        v = float("-inf")
        result_action = pac_actions[0]
        for action in pac_actions:
            min_ghost = self.expected_value(gameState.generateSuccessor(0, action), 1, 0, "-inf", "inf")
            if min_ghost > v:
                result_action = action
                v = min_ghost
        return result_action

    def expected_value(self, state, ghostIndex, d, alpha, beta):
        """
        def min-value(state , α, β):
            initialize v = +∞.
            for each successor s of state:
                v = min(v, value(s, α, β))
                if v ≤ α return v
                β = min(β,v)
            return v
        """
        if self.is_terminal(state, d):
            return self.evaluationFunction(state)
        # expected v initialize to 0
        v = 0
        legal_actions = state.getLegalActions(ghostIndex)
        prob = 1/len(legal_actions)
        for action in legal_actions:
            if ghostIndex == self.ghosts[-1]:
                v += prob * self.max_value(state.generateSuccessor(ghostIndex, action), d + 1, alpha, beta)
            else:
                v += prob * self.expected_value(state.generateSuccessor(ghostIndex, action), ghostIndex + 1, d, alpha, beta)
        return v

    def max_value(self, state, d, alpha, beta):
        """
        alpha: max's best option on path to root
        beta: min's best option on path to root
        def max-value(state， alpha, beta):
            initialize v = −∞.
            for each successor s of state:
                v = max(v, min - value(successor, alpha, beta))
                # this layer is to choose max, if v already > beta, then
                if v > beta:
                    return v
                alpha = max(alpha, v)
        return v
        """
        if self.is_terminal(state, d):
            return self.evaluationFunction(state)
        v = float("-inf")
        for action in state.getLegalActions(0):
            v = max(v, self.expected_value(state.generateSuccessor(0, action), 1, d, alpha, beta))
        return v

    def is_terminal(self, state, d):
        return state.isWin() or state.isLose() or self.depth == d



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    v = 0
    #final_action = currentGameState.getLegalActions(0)[0]
    toFood = []
    toGhost = []
    toCapsule = []
    for action in currentGameState.getLegalActions(0):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentPos = currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #currentGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
        newFood = newFood.asList()
        newGhostPos = [G.getPosition() for G in newGhostStates]
        newCapsule = successorGameState.getCapsules()
        min_new_scared = min(newScaredTimes)
        #min_current_scared = min(currentScaredTimes)
        #capsurePos = currentGameState.getCapsules()
        #index_mean_scared = np.argmin(np.array(newScaredTimes))
        if min_new_scared <= 0 and newPos in newGhostPos:
            v -= 100
       
        else:
            if len(newFood) > 0:
                food_d_min = min([manhattanDistance(f, newPos) for f in newFood])
                toFood.append(1.0/food_d_min)
            # ghost distance
            if len(newGhostPos) > 0:
                ghost_d_min = min([manhattanDistance(g, newPos) for g in newGhostPos])
            #toFood.append(1.0/food_d_min)
                toGhost.append(1/ghost_d_min)
            if len(newCapsule) > 0:
                cap_d_min = min([manhattanDistance(i, newPos) for i in newGhostPos])
                toCapsule.append(10/cap_d_min)
            #v += (10.0/food_d_min - 10.0/ghost_d_min - 10*len(capsurePos))
        #if temp > v:
            #v = temp
            #final_action = action
        #if min_new_scared > 0:
        #new = -100
    food = currentGameState.getFood().asList()
    capsurePos = currentGameState.getCapsules()
    if len(toFood) == 0 or len(toGhost) == 0 or len(toCapsule) == 0:
        return v  - 10 * len(capsurePos) - 10 * len(food) + currentGameState.getScore()
    else:
        return v+ max(toFood) - min(toGhost) + max(toCapsule) - 10 * len(capsurePos) - 10 * len(food) + currentGameState.getScore()
 
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
