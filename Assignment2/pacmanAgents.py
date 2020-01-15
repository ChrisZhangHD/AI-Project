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
import math

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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        print 'possible = ', possible
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        print 'actionList = ', self.actionList
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];


class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        possible = state.getAllPossibleActions()
        actionList = []
        for i in range(0, 5):
            actionList.append(possible[random.randint(0, len(possible) - 1)])
        actionScoreMap = [Directions.STOP, gameEvaluation(state, state)]

        while True:
            nextActionSeq = []
            for i in range(0, 5):
                if random.randint(1, 2) == 1:
                    nextActionSeq.append(actionList[i])
                else:
                    nextActionSeq.append(possible[random.randint(0, len(possible) - 1)])
            curState = state
            for action in nextActionSeq:
                nextState = curState.generatePacmanSuccessor(action)
                if nextState is not None and (nextState.isWin() + nextState.isLose()) == 0:
                    curState = nextState
                else:
                    if nextState is None:
                        return actionScoreMap[0]
                    if nextState.isWin():
                        return nextActionSeq[0]
                    if nextState.isLose():
                        break

            curScore = gameEvaluation(state, curState)
            if curScore > actionScoreMap[1]:
                actionScoreMap[1] = curScore
                actionScoreMap[0] = nextActionSeq[0]
                actionList = nextActionSeq


class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP

        def rankSelection(randomNumber):
            if randomNumber == 1:
                return 0
            elif randomNumber <= 3:
                return 1
            elif randomNumber <= 6:
                return 2
            elif randomNumber <= 10:
                return 3
            elif randomNumber <= 15:
                return 4
            elif randomNumber <= 21:
                return 5
            elif randomNumber <= 28:
                return 6
            elif randomNumber <= 36:
                return 7

        possible = state.getAllPossibleActions()
        population = []
        chromosomeScoreMap = []
        for i in range(8):
            eachChromosome = []
            for j in range(5):
                eachChromosome.append(random.choice(possible))
            population.append(eachChromosome)
            chromosomeScoreMap.append([gameEvaluation(state, state), eachChromosome])

        while True:
            for i in range(8):
                curChromosome = population[i]
                curState = state
                for action in curChromosome:
                    nextState = curState.generatePacmanSuccessor(action)
                    if nextState is not None and (nextState.isWin() + nextState.isLose()) == 0:
                        curState = nextState
                    else:
                        if nextState is None:
                            return (chromosomeScoreMap[0][1])[0]
                        if nextState.isWin():
                            return curChromosome[0]
                        if nextState.isLose():
                            break

                curChromosomeScore = gameEvaluation(state, curState)
                chromosomeScoreMap[i][0] = curChromosomeScore

            chromosomeScoreMap.sort()

            newPopulation = []
            while len(newPopulation) < 8:
                p1 = rankSelection(random.randint(1, 36))
                p2 = rankSelection(random.randint(1, 36))
                while p1 == p2:
                    p2 = rankSelection(random.randint(1, 36))
                parent1 = chromosomeScoreMap[p1][1]
                parent2 = chromosomeScoreMap[p2][1]

                if random.randint(1, 10) <= 7:
                    newChromosome = []
                    for i in range(5):
                        if random.randint(0, 1) == 1:
                            newChromosome.append(parent1[i])
                        else:
                            newChromosome.append(parent2[i])
                    newPopulation.append(newChromosome)
                else:
                    newPopulation.append(parent1)
                    newPopulation.append(parent2)

            isMutate = False
            if random.randint(1, 10) == 1:
                isMutate = True
            if isMutate:
                for eachNewChromosome in newPopulation:
                    for i in range(8):
                        eachNewChromosome[random.randint(0, len(eachNewChromosome) - 1)] = random.choice(possible)
            population = newPopulation


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        def UCT(node, parentVisitTimes):
            return node[3] / float(node[4]) + math.sqrt(2 * math.log(float(parentVisitTimes)) / node[4])

        def defaultPolicy(state):
            curState = state
            if curState is None:
                return None
            if curState.isLose():
                return None

            for i in range(5):
                actions = curState.getLegalPacmanActions()
                if actions is None or len(actions) == 0:
                    return None
                nextAction = random.choice(actions)
                nextState = curState.generatePacmanSuccessor(nextAction)
                if nextState is None:
                    return None
                if nextState.isLose():
                    return None
                curState = nextState
            reward = gameEvaluation(rootState, curState)
            return reward

        def getReward(state, node):
            curState = state
            curNode = node
            expandNode = None
            reward = None
            while curState is not None:
                stateAction = curState.getLegalPacmanActions()
                if len(curNode[2]) != len(stateAction):
                    expandNode = expand(curState, curNode)
                    reward = defaultPolicy(curState)
                    break
                else:
                    curNode = bestChild(curNode)
                    nextAction = curNode[1]
                    curState = curState.generatePacmanSuccessor(nextAction)
            return expandNode, reward

        def expand(state, node):
            actions = state.getLegalPacmanActions()
            childNodes = node[2]
            actionToChildNode = []
            for childNode in childNodes:
                actionToChildNode.append(childNode[1])
            for action in actions:
                if action not in actionToChildNode:
                    newChild = [node, action, [], 0.0, 1]
                    node[2].append(newChild)
                    return newChild
            return None

        def bestChild(node):
            return max(node[2], key=lambda x: UCT(x, node[4]))

        def backup(node, reward):
            while node is not None and reward is not None:
                node[4] += 1
                node[3] += reward
                node = node[0]

        # Node = [parent, action, child, score, visited]
        root = [None, None, [], 0.0, 1]
        rootState = state
        while True:
            node, reward = getReward(rootState, root)
            if node is not None and reward is not None:
                backup(node, reward)
            else:
                break
        mostVisitedNode = max(root[2], key=lambda x: x[4])
        return mostVisitedNode[1]







