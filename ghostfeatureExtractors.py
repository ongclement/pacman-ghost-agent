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
import numpy as np


def closestCapsule(pos, capsules, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) in capsules:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None


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
        feats[(state, action)] = 1.0
        return feats


class GhostAdvancedExtractor(GhostFeatureExtractor):
    def getFeatures(self, state, action):
        ## Boiler plate
        # Declare features
        features = util.Counter()

        # Get current positions of all agents
        ghost_a_pos = state.getGhostPosition(1)
        ghost_b_pos = state.getGhostPosition(2)
        pacman_pos = state.getPacmanPosition()

        # Get pacman's next position
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(pacman_pos[0] + dx), int(pacman_pos[1] + dy)
        pacman_next_pos = (next_x, next_y)

        # Get capsules left in state
        capsules = state.getCapsules()

        # Get walls in state
        walls = state.getWalls()

        
        # ## Feature 1: Distance from pacman
        # features['ghost_a_pacman_proximity'] = util.manhattanDistance(
        #     ghost_a_pos, pacman_pos)
        # features['ghost_b_pacman_proximity'] = util.manhattanDistance(
        #     ghost_b_pos, pacman_pos)


        ## Feature 2: Find distance to pacman's next position
        features['ghost_a_pacman_real_distance'] = pacmanDistanceBFS(ghost_a_pos, pacman_next_pos, walls)
        features['ghost_b_pacman_real_distance'] = pacmanDistanceBFS(ghost_b_pos, pacman_next_pos, walls)

        ## Feature: Make real distance negative if ghosts are scared
        negate = 1
        if state.getGhostState(1).scaredTimer > 0:
            negate = -1
        features['ghost_a_scared_pacman_real_distance'] = pacmanDistanceBFS(ghost_a_pos, pacman_next_pos, walls) * negate
        features['ghost_b_scared_pacman_real_distance'] = pacmanDistanceBFS(ghost_b_pos, pacman_next_pos, walls) * negate


        # ## Feature 3: If pacman dist is near, additional feature to give chase
        # if(features['ghost_a_pacman_proximity'] >= 0.4):
        #     features['ghost_a_chase'] = 10
        # if(features['ghost_b_pacman_proximity'] >= 0.4):
        #     features['ghost_b_chase'] = 10

        
        # ## Feature 4: Run away when Pacman is near capsule
        # cap_dist = closestCapsule((next_x, next_y), capsules, walls)
        # if cap_dist is not None:
        #     # Make the distance a number less than one otherwise the update will diverge wildly
        #     features["run_away"] = float(cap_dist) / \
        #         (walls.width * walls.height) * -1

        
        ## Return all features
        features.divideAll(10)
        return features


####################
# Helper functions #
####################

def pacmanDistanceBFS(ghost_pos, pacman_pos, walls):
    breadth = [(ghost_pos[0], ghost_pos[1], 0)]
    expanded = set()
    
    while breadth:
        pos_x, pos_y, dist = breadth.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))

        # Exit if pacman is found
        if (pos_x, pos_y) == (pacman_pos[0], pacman_pos[1]):
            return dist
        
        # Otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            breadth.append((nbr_x, nbr_y, dist+1))
    
    # Pacman not found
    return -1