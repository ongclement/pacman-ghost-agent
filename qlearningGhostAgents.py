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
    def __init__(self,epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0,agentIndex=1, extractor='GhostIdentityExtractor', **args):
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
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
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
        "*** YOUR CODE HERE ***"
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
        "*** YOUR CODE HERE ***"
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
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        print(reward)
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
            "*** YOUR CODE HERE ***"
            #sys.exit(1)
            pass

    def getAction(self, state):
        #Uncomment the following if you want one of your agent to be a random agent.
        #if self.agentIndex == 1:
        #    return random.choice(self.getLegalActions(state))
        if self.agentIndex == 1:
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


    