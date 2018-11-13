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
# import numpy as np


def ghostDistance(pacman_pos, ghost_pos, walls):
    fringe = [(pacman_pos[0], pacman_pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) == (int(ghost_pos[0]), int(ghost_pos[1])):
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
        # Boiler plate
        # Declare features
        features = util.Counter()

        # Get current positions of all agents
        ghost_a_pos = state.getGhostPosition(1)
        ghost_b_pos = state.getGhostPosition(2)
        pacman_pos = state.getPacmanPosition()
        ghost_a_x, ghost_a_y = state.getGhostPosition(1)
        ghost_b_x, ghost_b_y = state.getGhostPosition(2)
        pacman_x, pacman_y = state.getPacmanPosition()

        # Get pacman's next position
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(pacman_pos[0] + dx), int(pacman_pos[1] + dy)
        pacman_next_pos = (next_x, next_y)

        # Get capsules left in state
        capsules = state.getCapsules()

        # Get walls in state
        walls = state.getWalls()

        # if state.getGhostPosition(1) == (1, 9) or state.getGhostPosition(1) == (2, 9):
        if state.getGhostScore() <= -30:
            x, y = state.getGhostPosition(2)
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)

            ghost_b_dist = ghostDistance((next_x, next_y), pacman_pos, walls)
            if ghost_b_dist is not None:
                features["ghost_b_dist"] = float(ghost_b_dist) / \
                    (walls.width * walls.height) * 40
        else:
            x, y = state.getGhostPosition(1)
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)

            ghost_a_dist = closestCapsule((next_x, next_y), [(1, 9)], walls)
            # # ghost_b_dist = ghostDistance(ghost_b_pos, capsules[1], walls)
            if ghost_a_dist is not None:
                features["ghost_dist"] = float(ghost_a_dist) / \
                    (walls.width * walls.height)

        # # if blue makes it to (1,9)
        # if state.getGhostPosition(1) == (1, 9) or state.getGhostPosition(1) == (2, 9):
        #     # features['ghost_b_pacman_real_distance'] = 1 / \
        #     #     (pacmanDistanceBFS(ghost_b_pos, pacman_next_pos, walls) + 10) * 10
        #     # print(features)
        #     # features['ghost_b_pacman_proximity'] = util.manhattanDistance(
        #     #     ghost_b_pos, pacman_pos)
        #     # if(features['ghost_b_pacman_proximity'] >= 0.4):
        #     #     features['ghost_b_chase'] = 10
        #     # if len(state.getCapsules()) > 1 or state.getGhostState(1).scaredTimer > 0:
        #     #     features["ghost_b_dist"] = 0

        #     # x, y = state.getGhostPosition(2)
        #     # dx, dy = Actions.directionToVector(action)
        #     # next_x, next_y = int(x + dx), int(y + dy)

        #     # ghost_b_dist = ghostDistance((next_x, next_y), pacman_pos, walls)
        #     # if ghost_b_dist is not None:
        #     #     features["ghost_b_dist"] = float(ghost_b_dist) / \
        #     #         (walls.width * walls.height) * 10
        #     if len(state.getCapsules()) == 2:
        #         x, y = state.getGhostPosition(2)
        #         dx, dy = Actions.directionToVector(action)
        #         next_x, next_y = int(x + dx), int(y + dy)

        #         ghost_b_dist = closestCapsule(
        #             (next_x, next_y), [(1, 9)], walls)
        #         # # ghost_b_dist = ghostDistance(ghost_b_pos, capsules[1], walls)
        #         if ghost_b_dist is not None:
        #             features["ghost_dist"] = float(ghost_b_dist) / \
        #                 (walls.width * walls.height)

        #     # if len(state.getCapsules()) > 1 or state.getGhostState(1).scaredTimer > 0:
        #     #     features["ghost_b_dist"] = 0
        #     # else:
        #     #     features["ghost_dist"] = 0
        #     #     features["ghost_b_dist"] = 1 / util.manhattanDistance(
        #     #         ghost_b_pos, pacman_pos) * 10
        #     if len(state.getCapsules()) == 1 and state.getGhostState(1).scaredTimer > 0:
        #         features["ghost_dist"] = 0
        #         x, y = state.getGhostPosition(2)
        #         dx, dy = Actions.directionToVector(action)
        #         next_x, next_y = int(x + dx), int(y + dy)

        #         ghost_b_dist = closestCapsule(
        #             (next_x, next_y), [(1, 1)], walls)
        #         if ghost_b_dist is not None:
        #             features["ghost_1_1_dist"] = float(ghost_b_dist) / \
        #                 (walls.width * walls.height)
        #     if len(state.getCapsules()) == 1 and state.getGhostState(1).scaredTimer == 0:
        #         features["ghost_dist"] = 0
        #         features["ghost_1_1_dist"] = 0
        #         features["ghost_start_chase"] = util.manhattanDistance(
        #             ghost_b_pos, pacman_pos) * 3
        # else:
        #     x, y = state.getGhostPosition(1)
        #     dx, dy = Actions.directionToVector(action)
        #     next_x, next_y = int(x + dx), int(y + dy)

        #     ghost_a_dist = closestCapsule((next_x, next_y), [(1, 9)], walls)
        #     # # ghost_b_dist = ghostDistance(ghost_b_pos, capsules[1], walls)
        #     if ghost_a_dist is not None:
        #         features["ghost_dist"] = float(ghost_a_dist) / \
        #             (walls.width * walls.height)

        # elif len(state.getCapsules()) <= 1:

        # ## Feature: Distance from pacman
        # features['ghost_a_pacman_proximity'] = util.manhattanDistance(
        #     ghost_a_pos, pacman_pos)
        # features['ghost_b_pacman_proximity'] = util.manhattanDistance(
        #     ghost_b_pos, pacman_pos)

        # ## Feature: Find distance to pacman's next position
        # features['ghost_a_pacman_real_distance'] = 1 / (pacmanDistanceBFS(ghost_a_pos, pacman_next_pos, walls) + 10)
        # features['ghost_b_pacman_real_distance'] = 1 / (pacmanDistanceBFS(ghost_b_pos, pacman_next_pos, walls) + 10)

        # ## Feature: Scared Ghost
        # features['ghost_is_scared'] = 0
        # if state.getGhostState(1).scaredTimer > 0:
        #     features['ghost_is_scared'] = 12
        #     features['ghost_a_pacman_real_distance'] = 0
        #     features['ghost_b_pacman_real_distance'] = 0

        # Feature: Scared Ghost
        # features['ghost_is_scared'] = 0
        # if state.getGhostState(1).scaredTimer > 0:
        #     features['ghost_is_scared'] = 12
        #     features['ghost_a_pacman_real_distance'] *= -1
        #     features['ghost_b_pacman_real_distance'] *= -1

        # Feature: Get num of capsules
        # features['1_capsule_left'] = 0
        # if capsules == 1:
        #     features['1_capsule_left'] = 5

        # ## Feature: If pacman dist is near, additional feature to give chase
        # if(features['ghost_a_pacman_proximity'] >= 0.4):
        #     features['ghost_a_chase'] = 10
        # if(features['ghost_b_pacman_proximity'] >= 0.4):
        #     features['ghost_b_chase'] = 10

        # Feature: Run away when Pacman is near capsule
        # cap_dist = closestCapsule((next_x, next_y), capsules, walls)
        # if cap_dist is not None:
        #     features["run_away"] = float(cap_dist) / (walls.width * walls.height) * -1

        # Return all features
        # print('features = ' + str(features))
        features.divideAll(10)
        return features


####################
# Helper functions #
####################

def pacmanDistanceBFS(ghost_pos, pacman_pos, walls):
    breadth = [(ghost_pos[0], ghost_pos[1], 0)]
    expanded = set()
    distances = []

    while breadth:
        pos_x, pos_y, dist = breadth.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))

        # If pacman is found
        if (pos_x, pos_y) == (pacman_pos[0], pacman_pos[1]):
            distances.append(dist)

        # Otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            breadth.append((nbr_x, nbr_y, dist+1))

    # Return min distance
    if distances:
        return 1 / (min(distances) + 1)
    return 0


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
