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
import matplotlib.pyplot as plt

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
        self.weights_dict = {}
        self.features_dict = {}
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

            # Recording for graphing purposes
            if feature not in self.features_dict:
                self.features_dict[feature] = []
            else:
                self.features_dict[feature].append(features[feature])
            if feature not in self.weights_dict:
                self.weights_dict[feature] = []
            else:
                self.weights_dict[feature].append(self.weights[feature])

        # print(self.weights)
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementGhostAgent.final(self, state) 
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # for feature in self.features_dict.keys():
                # print(str(len(self.features_dict[feature])) + '=' + str(len(self.weights_dict[feature])))
                # steps = range(0, len(self.features_dict[feature]))
                # plt.plot(steps, self.features_dict[feature], 'Feature value')
                # plt.plot(steps, self.weights_dict[feature], 'Weight')
                # plt.xlabel('Step')
                # plt.title(feature)
                # plt.show()
            
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
        if self.agentIndex == 1 and state.getGhostPosition(1) == (1, 9):
            # if self.blueGhostAction == 'EAST' and 'WEST' in self.getLegalActions(state):
            self.doAction(state, 'East')
            return 'East'
            #     self.blueGhostAction = 'WEST'
            # elif 'EAST'in self.getLegalActions(state):
            #     self.doAction('EAST')
            #     self.blueGhostAction = 'EAST'
        if self.agentIndex == 1 and state.getGhostPosition(1) == (2, 9):
            self.doAction(state, 'West')
            return 'West'
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

# class ActorCriticModel(keras.Model):
#     def __init__(self, state_size, action_size):
#         super(ActorCriticModel, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#         self.dense1 = layers.Dense(100, activation='relu')
#         self.policy_logits = layers.Dense(action_size)
#         self.dense2 = layers.Dense(100, activation='relu')
#         self.values = layers.Dense(1)

#     def call(self, inputs):
#         # Forward pass
#         x = self.dense1(inputs)
#         logits = self.policy_logits(x)
#         v1 = self.dense2(inputs)
#         values = self.values(v1)
#         return logits, values

#     def record(episode,
#             episode_reward,
#             worker_idx,
#             global_ep_reward,
#             result_queue,
#             total_loss,
#             num_steps):
#         """
#         Arguments:
#             episode: Current episode
#             episode_reward: Reward accumulated over the current episode
#             worker_idx: Which thread (worker)
#             global_ep_reward: The moving average of the global reward
#             result_queue: Queue storing the moving average of the scores
#             total_loss: The total loss accumualted over the current episode
#             num_steps: The number of steps the episode took to complete
#         """
#         if global_ep_reward == 0:
#             global_ep_reward = episode_reward
#         else:
#             global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
#         return global_ep_reward


# class RandomAgent:
#     """Random Agent that will play the specified game
#         Arguments:
#         env_name: Name of the environment to be played
#         max_eps: Maximum number of episodes to run agent for.
#     """
#     def __init__(self, env_name, max_eps):
#         self.env = gym.make(env_name)
#         self.max_episodes = max_eps
#         self.global_moving_average_reward = 0
#         self.res_queue = Queue()

#     def run(self):
#         reward_avg = 0
#         for episode in range(self.max_episodes):
#         done = False
#         self.env.reset()
#         reward_sum = 0.0
#         steps = 0
#         while not done:
#             # Sample randomly from the action space and step
#             _, reward, done, _ = self.env.step(self.env.action_space.sample())
#             steps += 1
#             reward_sum += reward
#         # Record statistics
#         self.global_moving_average_reward = record(episode,
#                                                     reward_sum,
#                                                     0,
#                                                     self.global_moving_average_reward,
#                                                     self.res_queue, 0, steps)

#         reward_avg += reward_sum
#         final_avg = reward_avg / float(self.max_episodes)
#         print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
#         return final_avg

# class NeuralNetQAgent(QLearningGhostAgent):
#     def __init__(self, extractor='GhostAdvancedExtractor', *args, **kwargs):
#         self.nnet = None
#         QLearningGhostAgent.__init__(self, *args, **kwargs)

#     def getQValue(self, state, action):
#         if self.nnet is None:
#             self.nnet = NeuralNetwork(state)
#         prediction = self.nnet.predict(state, action)
#         return prediction

#     def update(self, state, action, nextState, reward):
#         if self.nnet is None:
#             self.nnet = NeuralNetwork(state)

#         maxQ = 0
#         for a in self.getLegalActions(nextState):
#             if self.getQValue(state, action) > maxQ:
#                 maxQ = self.getQValue(state, action)

#         y = reward + (self.discount * maxQ)

#         self.nnet.update(nextState, action, y)
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import RMSprop

# import numpy as np
# import random, util, math
# class NeuralNetwork:
#     def __init__(self, state):
#         walls = state.getWalls()
#         self.width = walls.width
#         self.height = walls.height
#         self.size = 5 * self.width * self.height

#         self.model = Sequential()
#         self.model.add(Dense(164, init='lecun_uniform', input_shape=(875,)))
#         self.model.add(Activation('relu'))

#         self.model.add(Dense(150, init='lecun_uniform'))
#         self.model.add(Activation('relu'))

#         self.model.add(Dense(1, init='lecun_uniform'))
#         self.model.add(Activation('linear'))

#         rms = RMSprop()
#         self.model.compile(loss='mse', optimizer=rms)

#     def predict(self, state, action):
#         reshaped_state = self.reshape(state, action)
#         return self.model.predict(reshaped_state, batch_size=1)[0][0]

#     def update(self, state, action, y):
#         reshaped_state = self.reshape(state, action)
#         y = [[y]]
#         self.model.fit(reshaped_state, y, batch_size=1, nb_epoch=1, verbose=1)

#     def reshape(self, state, action):
#         reshaped_state = np.empty((1, 2 * self.size))
#         food = state.getFood()
#         walls = state.getWalls()
#         for x in range(self.width):
#             for y in range(self.height):
#                 reshaped_state[0][x * self.width + y] = int(food[x][y])
#                 reshaped_state[0][self.size + x * self.width + y] = int(walls[x][y])
#         ghosts = state.getGhostPositions()
#         ghost_states = np.zeros((1, self.size))
#         for g in ghosts:
#             ghost_states[0][int(g[0] * self.width + g[1])] = int(1)
#         x, y = state.getPacmanPosition()
#         dx, dy = Actions.directionToVector(action)
#         next_x, next_y = int(x + dx), int(y + dy)
#         pacman_state = np.zeros((1, self.size))
#         pacman_state[0][int(x * self.width + y)] = 1
#         pacman_nextState = np.zeros((1, self.size))
#         pacman_nextState[0][int(next_x * self.width + next_y)] = 1
#         reshaped_state = np.concatenate((reshaped_state, ghost_states, pacman_state, pacman_nextState), axis=1)
#         return reshaped_state