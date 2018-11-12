# featureExtractors.py
# --------------------
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


"Feature extractors for Ghost Agent game states"

from game import Directions, Actions
import util
from math import sqrt

class GhostFeatureExtractor:
    def getFeatures(self, state, action):
        """
        Returns a dict from features to counts
        Usually, the count will just be 1.0 for
        indicator functions.
        """
        util.raiseNotDefined()

class GhostIdentityExtractor(GhostFeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

def aStarSearch():
    from game import Directions

    #initialization
    fringe = util.PriorityQueue() 
    visitedList = []

    #push the starting point into queue
    fringe.push((problem.getStartState(),[],0),0 + heuristic(problem.getStartState(),problem)) # push starting point with priority num of 0
    #pop out the point
    (state,toDirection,toCost) = fringe.pop()
    #add the point to visited list
    visitedList.append((state,toCost + heuristic(problem.getStartState(),problem)))

    while not problem.isGoalState(state): #while we do not find the goal point
        successors = problem.getSuccessors(state) #get the point's succesors
        for son in successors:
            visitedExist = False
            total_cost = toCost + son[2]
            for (visitedState,visitedToCost) in visitedList:
                # if the successor has not been visited, or has a lower cost than the previous one
                if (son[0] == visitedState) and (total_cost >= visitedToCost): 
                    visitedExist = True
                    break

            if not visitedExist:        
                # push the point with priority num of its total cost
                fringe.push((son[0],toDirection + [son[1]],toCost + son[2]),toCost + son[2] + heuristic(son[0],problem)) 
                visitedList.append((son[0],toCost + son[2])) # add this point to visited list

        (state,toDirection,toCost) = fringe.pop()

    return toDirection

class GhostAdvancedExtractor(GhostFeatureExtractor):
    def getFeatures(self, state, action):
        features = util.Counter()

        ## Feature 1: Distance from pacman
        
        # Get their positions
        ghost_a_pos = state.getGhostPosition(1)
        ghost_b_pos = state.getGhostPosition(2)
        pacman_pos = state.getPacmanPosition()

        features['ghost_a_pacman_proximity'] = util.manhattanDistance(ghost_a_pos, pacman_pos)
        features['ghost_b_pacman_proximity'] = util.manhattanDistance(ghost_b_pos, pacman_pos)

        ## Feature 2: Pacman's next position

        

        # Return all features
        features.divideAll(10)
        return features


        # # extract the grid of food and wall locations and get the ghost locations
        # food = state.getFood()
        # walls = state.getWalls()
        # ghosts = state.getGhostPositions()

        # features = util.Counter()

        # features["bias"] = 1.0

        # # compute the location of pacman after he takes the action
        # x, y = state.getPacmanPosition()
        # dx, dy = Actions.directionToVector(action)
        # next_x, next_y = int(x + dx), int(y + dy)

        # # count the number of ghosts 1-step away
        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # # if there is no danger of ghosts then add the food feature
        # if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
        #     features["eats-food"] = 1.0

        # dist = closestFood((next_x, next_y), food, walls)
        # if dist is not None:
        #     # make the distance a number less than one otherwise the update
        #     # will diverge wildly
        #     features["closest-food"] = float(dist) / (walls.width * walls.height)
        # features.divideAll(10.0)
        # return features

