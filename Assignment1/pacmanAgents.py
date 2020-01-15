# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import *
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        returnAction = []
        queue = []
        depth = 0
        index = 0
        legal = state.getLegalPacmanActions()
        for action in legal:
            nextState = state.generatePacmanSuccessor(action)
            queue.append((depth, action, nextState, index))
            returnAction.append(action)
            index += 1
        isBreak = False
        while queue:
            depth, perAction, curState, index = queue.pop(0)
            if curState.isWin():
                return returnAction[index]
            if curState.isLose():
                continue
            depth += 1

            nextLegalAction = curState.getLegalPacmanActions()
            for perAction in nextLegalAction:
                nextSuccessor = curState.generatePacmanSuccessor(perAction)
                if nextSuccessor is not None:
                    queue.append((depth, perAction, nextSuccessor, index))
                else:
                    isBreak = True
                    break
            if isBreak:
                break

        cost = []
        for perDepth, perAction, perState, index in queue:
            cost.append((perDepth + admissibleHeuristic(perState), index))
        if not cost:
            return Directions.STOP
        else:
            cost.sort()
            minCost, resIndex = cost.pop(0)
            return returnAction[resIndex]


class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        returnAction = []
        stack = []
        depth = 0
        index = 0
        legal = state.getLegalPacmanActions()
        for action in legal:
            nextState = state.generatePacmanSuccessor(action)
            stack.append((depth, action, nextState, index))
            returnAction.append(action)
            index += 1
        isBreak = False
        while stack:
            depth, perAction, curState, index = stack.pop(-1)
            if curState.isWin():
                return returnAction[index]
            if curState.isLose():
                continue
            depth += 1

            nextLegalAction = curState.getLegalPacmanActions()
            for perAction in nextLegalAction:
                nextSuccessor = curState.generatePacmanSuccessor(perAction)
                if nextSuccessor is not None:
                    stack.append((depth, perAction, nextSuccessor, index))
                else:
                    isBreak = True
                    break
            if isBreak:
                break

        cost = []
        for perDepth, perAction, perState, index in stack:
            cost.append((perDepth + admissibleHeuristic(perState), index))
        if not cost:
            return Directions.STOP
        else:
            cost.sort()
            minCost, resIndex = cost.pop(0)
            return returnAction[resIndex]


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        returnAction = []
        priorityQueue = []
        visited = set()
        depth = 0
        legal = state.getLegalPacmanActions()
        index = 0
        for action in legal:
            nextState = state.generatePacmanSuccessor(action)
            priorityQueue.append((depth + admissibleHeuristic(nextState), action, nextState, depth, index))
            returnAction.append(action)
            index += 1
        priorityQueue.sort()
        isBreak = False
        while priorityQueue:
            cost, perAction, curState, depth, index = priorityQueue.pop(0)
            if curState.isWin():
                return returnAction[index]
            if curState.isLose():
                continue
            if curState in visited:
                continue
            visited.add(curState)
            depth += 1

            nextCurLegal = curState.getLegalPacmanActions()
            for nextAction in nextCurLegal:
                nextSuccessor = curState.generatePacmanSuccessor(nextAction)
                if nextSuccessor is not None:
                    priorityQueue.append((depth + admissibleHeuristic(nextSuccessor), nextAction, nextSuccessor, depth, index))
                    priorityQueue.sort()
                else:
                    isBreak = True
                    break
            if isBreak:
                break

        if not priorityQueue:
            return Directions.STOP
        else:
            cost, resAction, resState, resDepth, index = priorityQueue.pop(0)
            return returnAction[index]
