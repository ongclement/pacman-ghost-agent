# qlearningGhostAgents.py
# ------------------
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


from game import *
from learningGhostAgents import ReinforcementGhostAgent
from ghostfeatureExtractors import *
import sys
import random,util,math
import pickle
class QLearningGhostAgent(ReinforcementGhostAgent):
    """
    Q-Learning Ghost Agent

    Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

    Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

    Functions you should use
        - self.getLegalActions(state)
        which returns legal actions for a state
    """
    def __init__(self,epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0,agentIndex=1, extractor='GhostAdvancedExtractor', **args):
        "You can initialize Q-values here..."
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        args['agentIndex'] = agentIndex
        self.index = agentIndex 
        self.q_values = util.Counter()
        
        self.featExtractor = util.lookup(extractor, globals())()
        self.weights = util.Counter()
        ReinforcementGhostAgent.__init__(self, **args)
        
    
    def getQValue(self, state, action):
        """
        Normal Q learning
        """
        f = self.featExtractor
        features = f.getFeatures(state,action)
        qvalue = 0
        for feature in features.keys():
            qvalue += self.weights[feature] * features[feature]
        return qvalue  

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0.0
        maxqvalue = -999999
        for action in legalActions:
            if self.getQValue(state,action) > maxqvalue:
                maxqvalue = self.getQValue(state,action)
        return maxqvalue  

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        bestAction = [None]
        legalActions = self.getLegalActions(state)
        maxqvalue = -999999
        for action in legalActions:
            if self.getQValue(state,action) > maxqvalue:
                maxqvalue = self.getQValue(state,action)
                bestAction = [action]
            elif self.getQValue(state,action) == maxqvalue:
                bestAction.append(action)

        return random.choice(bestAction)
        #util.raiseNotDefined()
        
    def getWeights(self):
        return self.weights

    def update(self, state, action, nextState, reward):
        """
        normal q learning
        """    
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        key = state,action
        self.q_values[key] = (1.0 - self.alpha) * self.getQValue(state,action) + self.alpha * sample
        #util.raiseNotDefined()
        """
        actionsFromNextState = self.getLegalActions(nextState)
        maxqnext = -999999
        for act in actionsFromNextState:
            if self.getQValue(nextState,act) > maxqnext:
                maxqnext = self.getQValue(nextState,act)
        if maxqnext == -999999:
            maxqnext = 0
        diff = (reward + (self.discount * maxqnext)) - self.getQValue(state,action)
        features = self.featExtractor.getFeatures(state,action)
        self.q_values[(state,action)] += self.alpha * diff 
        for feature in features.keys():
            self.weights[feature] += self.alpha * diff * features[feature]
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementGhostAgent.final(self, state) 
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.weights)

    def getAction(self, state):
        #Uncomment the following if you want one of your agent to be a random agent.
        #if self.agentIndex == 1:
        #    return random.choice(self.getLegalActions(state))
        # if self.agentIndex == 1:
        #     # Make first ghost aStarSearch agent
        #     action = aStarSearch(state)
        #     self.doAction(state, action)
        #     return action
        if util.flipCoin(self.epsilon):
            action = random.choice(self.getLegalActions(state))
            self.doAction(state, action)
            return action
        else:
            action = self.computeActionFromQValues(state)
            self.doAction(state, action)
            return action
            

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def aStarSearch():
        pass
        # from game import Directions

        # #initialization
        # fringe = util.PriorityQueue() 
        # visitedList = []

        # #push the starting point into queue
        # fringe.push((problem.getStartState(),[],0),0 + heuristic(problem.getStartState(),problem)) # push starting point with priority num of 0
        # #pop out the point
        # (state,toDirection,toCost) = fringe.pop()
        # #add the point to visited list
        # visitedList.append((state,toCost + heuristic(problem.getStartState(),problem)))

        # while not problem.isGoalState(state): #while we do not find the goal point
        #     successors = problem.getSuccessors(state) #get the point's succesors
        #     for son in successors:
        #         visitedExist = False
        #         total_cost = toCost + son[2]
        #         for (visitedState,visitedToCost) in visitedList:
        #             # if the successor has not been visited, or has a lower cost than the previous one
        #             if (son[0] == visitedState) and (total_cost >= visitedToCost): 
        #                 visitedExist = True
        #                 break

        #         if not visitedExist:        
        #             # push the point with priority num of its total cost
        #             fringe.push((son[0],toDirection + [son[1]],toCost + son[2]),toCost + son[2] + heuristic(son[0],problem)) 
        #             visitedList.append((son[0],toCost + son[2])) # add this point to visited list

        #     (state,toDirection,toCost) = fringe.pop()

        # return toDirection