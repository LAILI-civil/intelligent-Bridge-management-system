"""
This code is used to design the optimal maintenance plan in the component-level, (from whole bridge to superstructure,
then from superstructure to cable or girder). The Chinese standard JTGT H21-2011 Standards for assessing the technical
condition of highway bridges is used to evaluate the condition of bridge,
the CJJ 99-2017 Urban Bridge Maintenance Technical Standards is used to construct the maintenance.

Four necessary databases are constructed for Inspection & maintenance planning, which are inspection database, including
visual inspection and bridge testing operation vehicle (change the POMDP to MDP, uncertainty reduce to 0), and monitoring.
To reduce the action space and be consistent with reality, inspection action is for whole bridge.

Monitoring system is considered to affect statistical law for deterioration.
Second database is the deterioration database, which gives the deteriorated rate of each component in Weibull function or
Markov chain.

Third database is the maintenance database, which is divided the repair and replacement. This database give the
maintenance utility.

Last one is the Cost database, which gives the fee of inspection and maintenance.

Notice the analytic hierarchy evaluation process and corresponding weight is used to construct a decision tree to
reasonable simplify the action space from 4^16 = 4294967296  to 3^3+1 + 16 * 3^3 = 460.

The deterioration is not comform to the Markov process or semi-Markov process, it can follow the two parameters Weibull
or three parameters Weibull or mixed Markov chain

The author list the parameter of which can be adjusted for specific bridge. This framework is versatile for any type of
actual bridge once those parameters are updated for a specific bridge.

Copyright@2023: Li Lai (PhD in the Hong Kong Polytechnic University), Aijun Wang, senior programmer
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers
from tensorflow import keras
import numpy as np
import math
import matplotlib.pyplot as plt
import action_necessary

# Inspection database construction
# visual inspection
# bridge testing operation vehicle change the POMDP to MDP accuracy = 1
def observation(accuracy, state_number, Matrix_type = True):
    """
    the observation matrix is manually designed, based on the state space
    Args:
        accuracy: the accurate level of visual inspections(such as 70% =input 0.7)
        state_number: the state number of a component
        matrix_type: the discrete distributions of observation matrix
    Returns:
    observation_matrix
     """
    if Matrix_type:
        observation_matrix = np.zeros((state_number, state_number))
        observation_matrix[0, 0] = accuracy
        observation_matrix[0, 1] = 1 - accuracy
        observation_matrix[state_number - 1, state_number - 2] = 1 - accuracy
        observation_matrix[state_number - 1, state_number - 1] = accuracy
        for step in range(state_number - 2):
            observation_matrix[step + 1, step] = (1 - accuracy) / 2
            observation_matrix[step + 1, step + 1] = accuracy
            observation_matrix[step + 1, step + 2] = (1 - accuracy) / 2

        return observation_matrix
    else:
        # observation_matrix = np.array([[0.6, 0.3, 0.1],
        #                                [0.5, 0.35, 0.15],
        #                                [0.25, 0.5, 0.25],
        #                                [0.15, 0.35, 0.5],
        #                                [0.1, 0.3, 0.6]])
        observation_matrix = np.zeros((state_number, state_number))
        observation_matrix[0, 0] = accuracy
        observation_matrix[0, 1] = (1 - accuracy) * 2 / 3
        observation_matrix[0, 2] = (1 - accuracy) * 1 / 3

        observation_matrix[1, 0] = (1 - accuracy) * 2 / 5
        observation_matrix[1, 1] = accuracy
        observation_matrix[1, 2] = (1 - accuracy) * 2 / 5
        observation_matrix[1, 3] = (1 - accuracy) * 1 / 5

        observation_matrix[state_number - 2, state_number - 4] = (1 - accuracy) * 1 / 5
        observation_matrix[state_number - 2, state_number - 3] = (1 - accuracy) * 2 / 5
        observation_matrix[state_number - 2, state_number - 2] = accuracy
        observation_matrix[state_number - 2, state_number - 1] = (1 - accuracy) * 2 / 5

        observation_matrix[state_number - 1, state_number - 3] = (1 - accuracy) * 1 / 3
        observation_matrix[state_number - 1, state_number - 2] = (1 - accuracy) * 2 / 3
        observation_matrix[state_number - 1, state_number - 1] = accuracy
        for step in range(state_number - 4):
            observation_matrix[step + 2, step] = (1 - accuracy) / 6
            observation_matrix[step + 2, step + 1] = (1 - accuracy) / 3
            observation_matrix[step + 2, step + 2] = accuracy
            observation_matrix[step + 2, step + 3] = (1 - accuracy) / 3
            observation_matrix[step + 2, step + 4] = (1 - accuracy) / 6
        return observation_matrix


# component deterioration database construction

def Markov_state_transition_matrix(state_number, duration):
    """
        :param state_number: 5
        :param duration: a vector size is state_number - 1 to consider the duration of state i deteriorate to i+1
        :return: transition_matrix:
        """
    # assumption is the one-jump state can happen in a defined period
    transition_matrix = np.zeros((state_number, state_number))
    transition_matrix[state_number - 1, state_number - 1] = 1

    for step, ri in enumerate(duration):
        Pi = ri / (1 + ri)
        Qi = 1 / (1 + ri)
        transition_matrix[step, step] = Pi
        transition_matrix[step, step + 1] = Qi
    return transition_matrix


"""
state transition in Markov chain
"""
def state_evolution(state, time, old_hidden, state_T, state_T_D, normalized_time):
    """
    :param state: a size of number * 5 state vector
    :param time: a size of number * 1 vector
    :param old_hidden: a size of number * 1 vector
    :param state_T: state transition in normal service. size 5 * 5
    :param state_T_D: state transition in corrosion condition, size 5 * 5
    :param normalized_time: scale value
    :return: new_state, new_time, new_hidden
    """
    new_hidden = np.zeros((len(old_hidden)), dtype = int)
    new_time = np.zeros((len(time)))
    new_state = np.zeros((len(state)))
    for i in range(len(time)):
        if time[i] <= 0:
            new_state[i * 5: i * 5 + 5] = state[i * 5: i * 5 + 5] @ state_T_D
            new_state[i * 5: i * 5 + 5] = new_state[i * 5: i * 5 + 5] / np.sum(new_state[i * 5: i * 5 + 5])
            new_time[i] = 0

            Random_number = np.random.uniform(0, 1)
            state_mark = 0.
            for j in range(round(len(state)/len(time))):
                state_mark = state_mark + state_T_D[old_hidden[i], j]
                if Random_number <= state_mark:
                    new_hidden[i] = j
                    break

        elif time[i] > 0:
            new_state[i * 5: i * 5 + 5] = state[i * 5: i * 5 + 5] @ state_T
            new_state[i * 5: i * 5 + 5] = new_state[i * 5: i * 5 + 5] / np.sum(new_state[i * 5: i * 5 + 5])
            new_time[i] = time[i] - 1 / normalized_time
            if np.abs(new_time[i]) < 1e-10:
                new_time[i] = 0

            Random_number = np.random.uniform(0, 1)
            state_mark = 0.
            for j in range(round(len(state)/len(time))):
                state_mark = state_mark + state_T[old_hidden[i], j]
                if Random_number <= state_mark:
                    new_hidden[i] = j
                    break
    return new_state, new_time, new_hidden

"""
Three parameter Weibull function for 0-1 component state, such as lighting, welded details.
"""


def Three_parameter_Weibull_transition(component_num, parameters, time_duration):
    """
    drainage 0 0-5% >5%
    Safety barriers length 0 0-3% 3-10% >10%
    sidewalk 0 0-10% 10-20% >20%
    :param component_num: the current working number
    :param parameters: a vector size is 1*3, [scale parameter, shape parameter, location parameters]
    :param time_duration: a value refers to the normal working time
    :return: survive
    """
    a = parameters[0]
    b = parameters[1]
    mu = parameters[2]

    if time_duration <= mu:
        Cum_failure_rate = 0
        Reliability_t1 = 1 - Cum_failure_rate
    else:
        Cum_failure_rate = 1 - math.exp(-((time_duration - mu) / a) ** b)
        Reliability_t1 = 1 - Cum_failure_rate

    if time_duration + 1 <= mu:
        Cum_failure_rate = 0
        Reliability_t2 = 1 - Cum_failure_rate
    else:
        Cum_failure_rate = 1 - math.exp(-((time_duration + 1 - mu) / a) ** b)
        Reliability_t2 = 1 - Cum_failure_rate
    failure_rate = (Reliability_t1 - Reliability_t2) / (Reliability_t1 + 1e-20)
    Random_number = np.random.uniform(0, 1, round(component_num))
    survive = round(np.sum(Random_number > failure_rate))

    return survive

"""
action defining, for individual part, action is inspection, preventive maintenance, repair or replacement, 
the total action number is 3 * 64 * 29 = 5568
"""


def action_ecoding(action_index):
    """
    :param action_index: 0 means Do nothing, 1 means manual inspection, 2 means detail inspection (MDP)
    2^6 = 64, deck, arch and transverse, girder and slab, hanger, support, column and foundation
    0 means spending 0% percent annual budget, 1 means 5%, 31 means 155%
    Note: the 35-40 cost not include the inspection and preventive maintenance.
    Here, we list the possible action combinations. [0 1] [0 ~ 63] [0 - 37]
    Action index: 2 * 32 * 38 = 2432
    for instance Do nothing and preventive maintenance, action is [0, 1, 1, 1, 1,1, 0]
    :return: state_transition matrix
    """
    action_0 = action_index // 1216

    action_1 = (action_index % 1216) // 608
    action_2 = ((action_index % 1216) % 608) // 304
    action_3 = (((action_index % 1216) % 608) % 304) // 152
    action_4 = ((((action_index % 1216) % 608) % 304) % 152) // 76
    action_5 = (((((action_index % 1216) % 608) % 304) % 152) % 76) // 38

    action_6 = (((((action_index % 1216) % 608) % 304) % 152) % 76) % 38
    action_vector = np.array([action_0, action_1, action_2, action_3, action_4, action_5, action_6], dtype=int)
    return action_vector

def action_decoding(action_vector):
    """
    :param action_vector: a vector [0-2, 0-63, 0-28], int
    :return: action_value, [1,0,0,1,0,1,24], int
    """
    action_value = (action_vector[0] * 1216 + action_vector[1] * 608 + action_vector[2] * 304 + action_vector[3] * 152
                    + action_vector[4] * 76 + action_vector[5] * 38 + action_vector[6])
    return action_value

def state_ecoding(state):
    """
    :param state: state is from the tensorflow.numpy()
    pavement: 20 * 132; expansion joint: 20 * 2; sidewalk: 132 * 2; barrier: 132 * 4; drainage: 130/5*2 = 52;
    lighting: 120
    arch: 3 concrete-steel tube (Markov chain);  Transverse: 20 (Markov chain); hanger: 21 * 3 (Markov chain);
    support: 6 (Markov chain)
    Column: 6 (Markov chain); foundation: 6 (Markov chain)
    :return: the standard-based state
    """
    # pavement maximum life-span 15 years [1,2,3,...,15], minimum 3 years, and [0,0,0,1] 4 years
    # 15 + 15 * 4
    pavement_state = state[0:15*5]
    pavement_damage_percentage = 1 - np.sum(pavement_state[0:15])
    if pavement_damage_percentage == 0:
        DMCI_pavement = 100
    elif pavement_damage_percentage > 0 and pavement_damage_percentage <= 0.1:
        DMCI_pavement = 75
    elif pavement_damage_percentage > 0.1 and pavement_damage_percentage <= 0.2:
        DMCI_pavement = 60
    else:
        DMCI_pavement = 50

    # expansion joint maximum life-span 10 years, minimum 2 years, and two for deterioration [0,0,1] 3 years
    # 10 + 10 * 3
    expansion_state = state[75:75+10*4]
    expansion_damage_percentage = 1 - np.sum(expansion_state[0:10])
    if expansion_damage_percentage == 0:
        DMCI_expansion = 100
    elif expansion_damage_percentage > 0 and expansion_damage_percentage <= 0.1:
        DMCI_expansion = 75
    elif expansion_damage_percentage > 0.1 and expansion_damage_percentage <= 0.2:
        DMCI_expansion = 60
    else:
        DMCI_expansion = 50

    # sidewalk maximum life-span 32 years, minimum 16 years and two for deterioration [0,0,0,0,1] 5 years
    # 32 + 32 * 5
    sidewalk_state = state[115:115+32*6]
    sidewalk_damage_percentage = 1 - np.sum(sidewalk_state[0:32])
    if sidewalk_damage_percentage == 0:
        DMCI_sidewalk = 100
    elif sidewalk_damage_percentage > 0 and sidewalk_damage_percentage <= 0.1:
        DMCI_sidewalk = 75
    elif sidewalk_damage_percentage > 0.1 and sidewalk_damage_percentage <= 0.2:
        DMCI_sidewalk = 60
    else:
        DMCI_sidewalk = 50

    # Barrier maximum life-span 50 years, minimum 20 years, and two for deterioration [0,0,0,0,1] 5 years
    # 50 + 50 * 5
    barrier_state = state[307:307+50*6]
    barrier_damage_percentage = 1 - np.sum(barrier_state[0:50])
    if barrier_damage_percentage == 0:
        DMCI_barrier = 100
    elif barrier_damage_percentage > 0 and barrier_damage_percentage <= 0.1:
        DMCI_barrier = 75
    elif barrier_damage_percentage > 0.1 and barrier_damage_percentage <= 0.2:
        DMCI_barrier = 60
    else:
        DMCI_barrier = 50

    # drainage maximum life-span 50 years, and minimum life-span 15 years, no deterioration
    # 50
    drainage_state = state[607:657]
    drainage_damage_percentage = 1 - np.sum(drainage_state[0:50])
    if drainage_damage_percentage == 0:
        DMCI_drainage = 100
    elif drainage_damage_percentage > 0 and drainage_damage_percentage <= 0.1:
        DMCI_drainage = 80
    else:
        DMCI_drainage = 65

    # lighting maximum life-span 8 years (32850 hours / 365 / 12 best), (1 years), no deterioration
    lighting_state = state[657:665]
    lighting_damage_percentage = 1 - np.sum(lighting_state)
    if lighting_damage_percentage == 0:
        DMCI_lighting = 100
    elif lighting_damage_percentage > 0 and lighting_damage_percentage <= 0.1:
        DMCI_lighting = 75
    elif lighting_damage_percentage > 0.1 and lighting_damage_percentage <= 0.2:
        DMCI_lighting = 60
    else:
        DMCI_lighting = 50

    # arch depends on the Markov chain, and 3 node for time, normalized 0-12
    arch_state = state[665:680]
    # arch_time = state[680:683]
    arch_expected_score = np.zeros((round(len(arch_state)/5)))
    for i in range(round(len(arch_state)/5)):
        arch_expected_score[i] = arch_state[5*i:5*(i+1)] @ np.array([100, 65, 55, 40, 0])

    # Transverse depends on the Markov chain, 20 nodes for time, normalized 0-12
    transverse_state = state[683:783]
    # transverse_time = state[783:803]
    transverse_expected_score = np.zeros((round(len(transverse_state)/5)))
    for i in range(round(len(transverse_state) / 5)):
        transverse_expected_score[i] = transverse_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    # Primary girder depends on the Markov chain, 4 nodes for time, normalized 0-15
    pri_girder_state = state[803:823]
    # pri_girder_time = state[823:827]
    pri_girder_expected_score = np.zeros((round(len(pri_girder_state) / 5)))
    for i in range(round(len(pri_girder_state) / 5)):
        pri_girder_expected_score[i] = pri_girder_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    # Secondary girder depends on the Markov chain, 42 nodes for time, normalized 0-15
    sec_girder_state = state[827:1037]
    # sec_girder_time = state[1037:1079]
    sec_girder_expected_score = np.zeros((round(len(sec_girder_state) / 5)))
    for i in range(round(len(sec_girder_state) / 5)):
        sec_girder_expected_score[i] = sec_girder_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    # bridge deck slab depends on the Markov chain, 44 nodes for time, normalized 0-15
    slab_state = state[1079:1299]
    # slab_time = state[1299:1343]
    slab_expected_score = np.zeros((round(len(slab_state) / 5)))
    for i in range(round(len(slab_state) / 5)):
        slab_expected_score[i] = slab_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    # hanger depends on the Markov chain, 63 nodes for time, normalized 0-5
    hanger_state = state[1343:1658]
    # hanger_time = state[1658:1721]
    hanger_expected_score = np.zeros((round(len(hanger_state)/5)))
    for i in range(round(len(hanger_state) / 5)):
        hanger_expected_score[i] = hanger_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    # support depends on the Markov chain, 8 nodes for time, normalized 0-5
    support_state = state[1721:1761]
    # support_time = state[1761:1769]
    support_expected_score = np.zeros((round(len(support_state)/5)))
    for i in range(round(len(support_state) / 5)):
        support_expected_score[i] = support_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    # column depends on the Markov chain, 6 nodes for time, normalized 0-15
    column_state = state[1769:1799]
    # column_time = state[1799:1805]
    column_expected_score = np.zeros((round(len(column_state)/5)))
    for i in range(round(len(column_state) / 5)):
        column_expected_score[i] = column_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    # foundation depends on the Markov chain, 6 nodes for time, normalized 0-10
    foundation_state = state[1805:1835]
    # foundation_time = state[1835:1841]
    foundation_expected_score = np.zeros((round(len(foundation_state)/5)))
    for i in range(round(len(foundation_state) / 5)):
        foundation_expected_score[i] = foundation_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    # standard t value
    t = np.array(
        [0, 10, 9.7, 9.5, 9.2, 8.9, 8.7, 8.5, 8.3, 8.1, 7.9, 7.7, 7.5, 7.3, 7.2, 7.08, 6.96, 6.84, 6.72, 6.6, 6.48,
         6.36, 6.24, 6.12, 6, 5.88, 5.76, 5.64, 5.52, 5.4, 5.35, 5.3, 5.25, 5.2, 5.15, 5.1, 5.05, 5, 4.95, 4.9, 4.85,
         4.8, 4.75, 4.7, 4.65, 4.6, 4.55, 4.5, 4.45, 4.4, 4.36, 4.32, 4.28, 4.24, 4.2, 4.16, 4.12, 4.08, 4.04])
    t = np.concatenate((t, np.arange(4, 2.8, -0.04)))

    if np.all(arch_expected_score >= 60):
        PCCI_arch = np.mean(arch_expected_score) - (100 - np.min(arch_expected_score)) / t[
            (round(len(arch_state) / 5 - 1))]
    else:
        PCCI_arch = np.min(arch_expected_score)
    if np.all(transverse_expected_score >= 60):
        PCCI_transverse = np.mean(transverse_expected_score) - (100 - np.min(transverse_expected_score)) / t[
            (round(len(transverse_state) / 5 - 1))]
    else:
        PCCI_transverse = np.min(transverse_expected_score)
    if np.all(pri_girder_expected_score >= 60):
        PCCI_Pri_girder = np.mean(pri_girder_expected_score) - (100 - np.min(pri_girder_expected_score)) / t[
            (round(len(pri_girder_state) / 5 - 1))]
    else:
        PCCI_Pri_girder = np.min(pri_girder_expected_score)
    if np.all(sec_girder_expected_score >= 60):
        PCCI_Sec_girder = np.mean(sec_girder_expected_score) - (100 - np.min(sec_girder_expected_score)) / t[
            (round(len(sec_girder_state) / 5 - 1))]
    else:
        PCCI_Sec_girder = np.min(sec_girder_expected_score)
    if np.all(slab_expected_score >= 60):
        PCCI_slab = np.mean(slab_expected_score) - (100 - np.min(slab_expected_score)) / t[
            (round(len(slab_state) / 5 - 1))]
    else:
        PCCI_slab = np.min(slab_expected_score)
    if np.all(hanger_expected_score >= 60):
        PCCI_hanger = np.mean(hanger_expected_score) - (100 - np.min(hanger_expected_score)) / t[
            (round(len(hanger_state) / 5 - 1))]
    else:
        PCCI_hanger = np.min(hanger_expected_score)
    if np.all(support_expected_score >= 60):
        PCCI_support = np.mean(support_expected_score) - (100 - np.min(support_expected_score)) / t[
            (round(len(support_state) / 5 - 1))]
    else:
        PCCI_support = np.min(support_expected_score)

    if np.all(column_expected_score >= 60):
        BCCI_column = np.mean(column_expected_score) - (100 - np.min(column_expected_score)) / t[
            (round(len(column_state) / 5 - 1))]
    else:
        BCCI_column = np.min(column_expected_score)
    if np.all(foundation_expected_score >= 60):
        BCCI_foundation = np.mean(foundation_expected_score) - (100 - np.min(foundation_expected_score)) / t[
            (round(len(foundation_state) / 5 - 1))]
    else:
        BCCI_foundation = np.min(foundation_expected_score)

    DMCI = (0.4 * DMCI_pavement + 0.25 * DMCI_expansion + 0.1 * DMCI_sidewalk + 0.1 * DMCI_barrier + 0.1 * DMCI_drainage
            + 0.05 * DMCI_lighting)
    PCCI = 0.322 * PCCI_arch + 0.057 * PCCI_transverse + 0.161 * PCCI_Pri_girder + 0.161 * PCCI_Sec_girder + 0.092 * PCCI_slab + 0.15 * PCCI_hanger + 0.057 * PCCI_support
    BCCI = 0.517 * BCCI_column + 0.483 * BCCI_foundation

    Bridge_state = DMCI * 0.2 + PCCI * 0.4 + BCCI * 0.4

    DMCI = np.array([DMCI_pavement, DMCI_expansion, DMCI_sidewalk, DMCI_barrier, DMCI_drainage, DMCI_lighting])
    PCCI = np.array([PCCI_arch, PCCI_transverse, PCCI_Pri_girder, PCCI_Sec_girder, PCCI_slab, PCCI_hanger, PCCI_support])
    BCCI = np.array([BCCI_column, BCCI_foundation])

    return DMCI, PCCI, BCCI, Bridge_state

# estimate the utility of preventive maintenance action
def preventive_utility(preventive_time, discounting_factor, base_cost, equipment_cost):
    """
    :param preventive_time: a vector represents the component number on resistant time
    for instance, pavement has 4 years resistant time, [40, 20, 10, 3], 40 means 40 units resistent time is 0
    20 units resistent time is 1 ...
    :param discounting_factor: consider the cost decrease with maintenance number increasing
    :param base_cost: unit preventive cost
    :param equipment_cost: rent of construction equipment
    :return: cost, utility, and preventive_number
    """
    utility = np.zeros((len(preventive_time)))
    cost = np.zeros((len(preventive_time)))
    utility_vector = np.arange(len(preventive_time), 0, -1)
    for i in range(len(preventive_time)):
        cost[i] = np.sum(preventive_time[0:i + 1]) * ((equipment_cost / base_cost) / math.exp(
            np.sum(preventive_time[0:i + 1]) / discounting_factor) + 1) * base_cost
        utility[i] = cost[i] / (np.sum(preventive_time[0:i + 1] * utility_vector[0:i + 1]) + 1e-10)
    preventive_number = np.argmin(utility)
    return cost[preventive_number], utility[preventive_number], preventive_number

def maintenance_cost(number, discounting_factor, base_cost, equipment_cost):
    """
    :param number: a scale value represents the damage component number
    :param discounting_factor: consider the cost decrease with maintenance number increasing
    :param base_cost: unit replacement cost
    :param equipment_cost: rent of construction equipment
    :return: utility value
    """
    Cost = ((equipment_cost / base_cost) / math.exp(number / discounting_factor) + 1) * base_cost
    return Cost

def large_component_maintenance(state, PCCI, unit_cost, size, percentage, weight, discounting_factor, rent_fee):
    """
    :param state: a vector of state of component
    :param PCCI: score of component
    :param unit_cost: minimum-partition of maintenance unit
    :param size: size of component, related to unit area cost
    :param percentage: the percentage damage area of component, a vector such as [0, 0.05, 0.15, 0.3, 0.5]
    :param weight: weighting factor
    :param discounting_factor: maintenace fee discounting with maintenance area increasing
    :param rent_fee: the construction machinery fee
    :return: maintenance_strategy, utility
    """

    # standard t value
    t = np.array(
        [0, 10, 9.7, 9.5, 9.2, 8.9, 8.7, 8.5, 8.3, 8.1, 7.9, 7.7, 7.5, 7.3, 7.2, 7.08, 6.96, 6.84, 6.72, 6.6, 6.48,
         6.36, 6.24, 6.12, 6, 5.88, 5.76, 5.64, 5.52, 5.4, 5.35, 5.3, 5.25, 5.2, 5.15, 5.1, 5.05, 5, 4.95, 4.9, 4.85,
         4.8, 4.75, 4.7, 4.65, 4.6, 4.55, 4.5, 4.45, 4.4, 4.36, 4.32, 4.28, 4.24, 4.2, 4.16, 4.12, 4.08, 4.04])
    t = np.concatenate((t, np.arange(4, 2.8, -0.04)))

    expected_state = np.zeros((round(len(state) / 5)))
    expected_score = np.zeros((round(len(state) / 5)))
    for i in range(round(len(state) / 5)):
        expected_state[i] = state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        expected_score[i] = state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])
    component_damage_percentage = np.interp(expected_state, np.array([1, 2, 3, 4, 5]), percentage)

    # necessary cost, state worst than 4.
    maintenance_necessary = 0
    Low_performance_component = np.where(expected_state >= 4)[0]
    number = np.sum(size * component_damage_percentage[Low_performance_component])

    for i in range(len(Low_performance_component)):
        maintenance_necessary += unit_cost * size * component_damage_percentage[Low_performance_component[i]] * (
                    (rent_fee / unit_cost) / math.exp(number / discounting_factor) + 1)
    expected_score[Low_performance_component] = 100
    if np.all(expected_score >= 60):
        PCCI_new = np.mean(expected_score) - (100 - np.min(expected_score)) / t[(round(len(state) / 5 - 1))]
    else:
        PCCI_new = np.min(expected_score)
    utility_low = np.round(maintenance_necessary / ((PCCI_new - PCCI) * weight + 1e-10), 4)

    # optional maintenance, state at 3-4
    maintenance_optional_1 = 0
    poor_performance_component = np.where((expected_state < 4) & (expected_state >= 3))[0]
    number_1 = np.sum(size * component_damage_percentage[Low_performance_component]) + np.sum(
        size * component_damage_percentage[poor_performance_component])
    for i in range(len(Low_performance_component)):
        maintenance_optional_1 += unit_cost * size * component_damage_percentage[Low_performance_component[i]] * (
                (rent_fee / unit_cost) / math.exp(number_1 / discounting_factor) + 1)
    for i in range(len(poor_performance_component)):
        maintenance_optional_1 += unit_cost * size * component_damage_percentage[poor_performance_component[i]] * (
                    (rent_fee / unit_cost) / math.exp(number_1 / discounting_factor) + 1)
    expected_score[poor_performance_component] = 100
    if np.all(expected_score >= 60):
        PCCI_optional_1 = np.mean(expected_score) - (100 - np.min(expected_score)) / t[(round(len(state) / 5 - 1))]
    else:
        PCCI_optional_1 = np.min(expected_score)
    utility_optional_1 = np.round(maintenance_optional_1 / ((PCCI_optional_1 - PCCI) * weight + 1e-10), 4)

    # optional maintenance, state as 2-3
    maintenance_optional_2 = 0
    moderate_performance_component = np.where((expected_state < 3) & (expected_state > 2))[0]
    number_2 = np.sum(size * component_damage_percentage[Low_performance_component]) + np.sum(
        size * component_damage_percentage[poor_performance_component]) + np.sum(
        size * component_damage_percentage[moderate_performance_component])
    for i in range(len(Low_performance_component)):
        maintenance_optional_2 += unit_cost * size * component_damage_percentage[Low_performance_component[i]] * (
                (rent_fee / unit_cost) / math.exp(number_2 / discounting_factor) + 1)
    for i in range(len(poor_performance_component)):
        maintenance_optional_2 += unit_cost * size * component_damage_percentage[poor_performance_component[i]] * (
                (rent_fee / unit_cost) / math.exp(number_2 / discounting_factor) + 1)
    for i in range(len(moderate_performance_component)):
        maintenance_optional_2 += unit_cost * size * component_damage_percentage[moderate_performance_component[i]] * (
                (rent_fee / unit_cost) / math.exp(number_2 / discounting_factor) + 1)
    expected_score[moderate_performance_component] = 100
    if np.all(expected_score >= 60):
        PCCI_optional_2 = np.mean(expected_score) - (100 - np.min(expected_score)) / t[(round(len(state) / 5 - 1))]
    else:
        PCCI_optional_2 = np.min(expected_score)
    utility_optional_2 = np.round(maintenance_optional_2 / ((PCCI_optional_2 - PCCI) * weight + 1e-10), 4)

    return utility_low, maintenance_necessary, utility_optional_1, maintenance_optional_1, utility_optional_2, maintenance_optional_2

def cost_cal(state, bridge, DMCI, PCCI, BCCI):
    """
    Cost need to consider the traffic interruption (Hours of work), depend on maximum interrupted action
    maintenance cost discounting with the damage area
    :param state: based on the state return the cost of conduct maintenance
    Notice the cost for superstructure and substructure contain repair and replacement action
    Notice the repair and replacement include the traffic interruption cost.
    unit: 10000￥
    inspection_car_fee = 0.4 / day
    manual_inspection_equipment = 0.1 / day
    insepction_workforce_fee = 0.1 / day
    --------------------------------------------------------------------------------------------------------------------
    Preventive maintenance:
    Pavement_wearing_fee = 25 / 10000 per m^2
    expansion_joint_clearing = 49.4 / 10000 per m
    sidewalk_surface_mend = 36.63/10000 per m^2 concrete surface protection
    Concrete_barrier_fee = 29.52 / 10000 per m^2
    drainage_clear = 4.4/10000 per number
    lighting_no

    arch_corrosion = (13.5 + 28.8 (bottom painting) + 15.4 (mid) + 28.9(surface))  descaling and painting per m^2
    Transverse_corrosion = (13.5 + 28.8 (bottom painting) + 15.4 (mid) + 28.9(surface))  descaling and painting per m^2
    girder_prevention = (5.2 + 11.3+ 45.5) clearing, roughen, and mortar per m^2
    slab_prevention = (7.8 + 11.3 + 15.3) clearing, roughen, and mortar per m^2
    hanger_corrosion = 16269 / 10000 per ton, ton is the mass of steel wire
    support_prevention = (46+46.5+255)/10000 per number

    column_concrete_flaw_insite = 29.36 / 10000 per m^2
    Foundation_concrete_flaw = 29.36 / 10000 per m^2
    --------------------------------------------------------------------------------------------------------------------
    repair and replacement:
    Pavement = 879/10000 per m^3 (replacement) asphalt，for this bridge, road thickness is 16cm
    expansion_joint = (140.1 + 638)/10000 per m (replacement) rubber expansion joint
    sidewalk = (606.1 + 202.4) / 10000 per m^3 precast + installation, 30 * 99 cm thickness 242.55 per m^2
    barrier = 228 / 10000 per m
    drainage = (104 + 1000) / 10000 per number
    lighting = 500/10000 per/number

    arch = 1320/10000 per m (repair cannot replacement, mainly on crack)
    Transverse = 949 / 10000 per m (repair)
    girder = 1264 / 10000 per m^2 (repair) 1.75 * 1.1 and 1.75 * 0.95
    slab = 1265.5 / 10000 per m^3 (repair) 10 width 6 length, assume 10cm, 126.55 per m^2
    hanger = 35342 / 10000 per ton, ton is the mass of steel wire
    support = (1037 * number)+ 12399 / 10000, 12399 is the lifting fee

    column = 949 / 10000 per m^2 12 mm (steel plate reinforcement)
    Foundation = 396.2 / 10000 per m^3 (concrete)
    :return: the cost
    """
    """
    get structural information from the state vector--------------------------------------------------------------------
    """
    pavement_state = state[0:15]
    expansion_state = state[75:75 + 10]
    sidewalk_state = state[115:115 + 32]
    barrier_state = state[307:307 + 50]
    drainage_state = state[607:607 + 50]
    lighting_state = state[657:665]
    arch_state = state[665:680]
    transverse_state = state[683:783]
    pri_girder_state = state[803:823]
    sec_girder_state = state[827:1037]
    slab_state = state[1079:1299]
    hanger_state = state[1343:1658]
    support_state = state[1721:1761]
    column_state = state[1769:1799]
    foundation_state = state[1805:1835]

    """
    repair and replacement cost, it should be noted that the replacement do not need the preventive maintenance, since 
    the component is new. Repair needs the preventive maintenance.
    In addition, the damage component don't need preventive maintenance, because it is broken. 
    ------------------------------------------------------
    """
    # because the budget is limited, the maintenance action utility will be calculated
    # bridge deck system
    pavement_number = round(bridge[0] * (1 - np.sum(pavement_state)))
    if np.sum(pavement_state) < 0.8:
        unit_N_pavement_cost = maintenance_cost(pavement_number, 30, 90.64, 10548)
        pavement_necessary_cost = pavement_number * unit_N_pavement_cost
        pavement_necessary_utility = pavement_necessary_cost / ((100 - DMCI[0]) * 0.4 * 0.2)
    else:
        pavement_necessary_cost = 0
        pavement_necessary_utility = 0
    unit_O_pavement_cost = maintenance_cost(bridge[0], 30, 90.64, 10548)
    pavement_optional_cost = bridge[0] * unit_O_pavement_cost
    # pavement_optional_utility < 1e10 is judge condition
    pavement_optional_utility = pavement_optional_cost / ((100 - DMCI[0]) * 0.4 * 0.2 + 1e-10)

    expansion_number = round(bridge[1] * (1 - np.sum(expansion_state)))
    if np.sum(expansion_state) < 0.8:
        unit_N_expansion_cost = maintenance_cost(expansion_number, 2.5, 778.1, 3890.5)
        expansion_necessary_cost = expansion_number * unit_N_expansion_cost
        expansion_necessary_utility = expansion_necessary_cost / ((100 - DMCI[1]) * 0.25 * 0.2)
    else:
        expansion_necessary_cost = 0
        expansion_necessary_utility = 0
    unit_O_expansion_cost = maintenance_cost(bridge[1], 2.5, 778.1, 3890.5)
    expansion_optional_cost = bridge[1] * unit_O_expansion_cost
    expansion_optional_utility = expansion_optional_cost / ((100 - DMCI[1]) * 0.25 * 0.2 + 1e-10)

    sidewalk_number = round(bridge[2] * (1 - np.sum(sidewalk_state)))
    if np.sum(sidewalk_state) < 0.8:
        unit_N_sidewalk_cost = maintenance_cost(sidewalk_number, 10, 242.55, 8085)
        sidewalk_necessary_cost = sidewalk_number * unit_N_sidewalk_cost
        sidewalk_necessary_utility = sidewalk_necessary_cost / ((100 - DMCI[2]) * 0.1 * 0.2)
    else:
        sidewalk_necessary_cost = 0
        sidewalk_necessary_utility = 0
    unit_O_sidewalk_cost = maintenance_cost(bridge[2], 10, 242.55, 8085)
    sidewalk_optional_cost = bridge[2] * unit_O_sidewalk_cost
    sidewalk_optional_utility = sidewalk_optional_cost / ((100 - DMCI[2]) * 0.1 * 0.2 + 1e-10)

    barrier_number = round(bridge[3] * (1 - np.sum(barrier_state)))
    if np.sum(barrier_state) < 0.8:
        unit_N_barrier_cost = maintenance_cost(barrier_number, 7, 228, 2280)
        barrier_necessary_cost = barrier_number * unit_N_barrier_cost
        barrier_necessary_utility = barrier_necessary_cost / ((100 - DMCI[3]) * 0.1 * 0.2)
    else:
        barrier_necessary_cost = 0
        barrier_necessary_utility = 0
    unit_O_barrier_cost = maintenance_cost(bridge[3], 7, 228, 2280)
    barrier_optional_cost = bridge[3] * unit_O_barrier_cost
    barrier_optional_utility = barrier_optional_cost / ((100 - DMCI[3]) * 0.1 * 0.2 + 1e-10)

    drainage_number = round(bridge[4] * (1 - np.sum(drainage_state)))
    unit_drainage_cost = maintenance_cost(drainage_number, 1.5, 604, 6832)
    drainage_replacement_cost = drainage_number * unit_drainage_cost
    drainage_maintenance_utility = drainage_replacement_cost / ((100 - DMCI[4]) * 0.1 * 0.2 + 1e-10)

    lighting_number = round(bridge[5] * (1 - np.sum(lighting_state)))
    unit_lighting_cost = maintenance_cost(lighting_number, 2, 400, 4000)
    lighting_replacement_cost = lighting_number * unit_lighting_cost
    lighting_maintenance_utility = lighting_replacement_cost / ((100 - DMCI[5]) * 0.05 * 0.2 + 1e-10)

    # the arch and Transverse uses repair action not replacement!
    # two possible maintenance strategies, one is repair the low score components, other repair all components
    # similar, we find the maximum utility of maintenance strategy
    # arch (repair)
    arch_utility_necessary, arch_cost_necessary, arch_utility_optional_1, arch_cost_optional_1, arch_utility_optional_2, arch_cost_optional_2 = large_component_maintenance(
        arch_state, PCCI[0], 721, bridge[6],
        np.array([0, 0.05, 0.15, 0.3, 0.5]),
        0.1288, 80,
        12300)
    # transverse (repair)
    transverse_utility_necessary, transverse_cost_necessary, transverse_utility_optional_1, transverse_cost_optional_1, transverse_utility_optional_2, transverse_cost_optional_2 = large_component_maintenance(
        transverse_state, PCCI[1],
        721, bridge[7],
        np.array([0, 0.05, 0.15, 0.3, 0.5]),
        0.0228, 13,
        14420)
    # pri_girder (repair)
    pri_girder_utility_necessary, pri_girder_cost_necessary, pri_girder_utility_optional_1, pri_girder_cost_optional_1, pri_girder_utility_optional_2, pri_girder_cost_optional_2 = large_component_maintenance(
        pri_girder_state, PCCI[2],
        764, bridge[8],
        np.array([0, 0.25, 0.5, 0.75, 1]),
        0.0644, 40,
        18960)
    # sec_girder (repair)
    sec_girder_utility_necessary, sec_girder_cost_necessary, sec_girder_utility_optional_1, sec_girder_cost_optional_1, sec_girder_utility_optional_2, sec_girder_cost_optional_2 = large_component_maintenance(
        sec_girder_state, PCCI[3],
        764, bridge[9],
        np.array([0, 0.25, 0.5, 0.75, 1]),
        0.0644, 30,
        18960)
    # slab (repair)
    slab_utility_necessary, slab_cost_necessary, slab_utility_optional_1, slab_cost_optional_1, slab_utility_optional_2, slab_cost_optional_2 = large_component_maintenance(
        slab_state, PCCI[4],
        546.55, bridge[10],
        np.array([0, 0.05, 0.1, 0.15, 0.3]),
        0.0368, 32,
        8198.25)
    # cost depends on the length of hanger (replacement) 13344.6 + 48.6 (day) * 400 (force) per ton + equipment 30000
    # hanger length weight mid: 14-15.2mm 15.414kg/m  side: 12-15.2mm 13.101kg/m  (1.101kg/m)
    # the function is different from the idential component
    hanger_expected_state = np.zeros((round(len(hanger_state) / 5)))
    hanger_expected_score = np.zeros((round(len(hanger_state) / 5)))
    hanger_length = np.concatenate((bridge[11:32], bridge[11:32], bridge[11:32]))
    for i in range(round(len(hanger_state) / 5)):
        hanger_expected_state[i] = hanger_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        hanger_expected_score[i] = hanger_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    hanger_cost_necessary = 0
    Low_performance_hanger = np.where(hanger_expected_state >= 4)[0]
    if len(Low_performance_hanger) > 0:
        hanger_cost_necessary += 18835.5
        for i in range(len(Low_performance_hanger)):
            if Low_performance_hanger[i] <= 21 or Low_performance_hanger[i] > 42:
                hanger_cost_necessary += hanger_length[Low_performance_hanger[i]] * 13.101 / 1000 * 12784.6
            else:
                hanger_cost_necessary += hanger_length[Low_performance_hanger[i]] * 15.414 / 1000 * 12784.6
    hanger_expected_score[Low_performance_hanger] = 100
    if np.all(hanger_expected_score >= 60):
        PCCI_new_hanger = np.mean(hanger_expected_score) - (100 - np.min(hanger_expected_score)) / 3.88
    else:
        PCCI_new_hanger = np.min(hanger_expected_score)
    hanger_utility_necessary = hanger_cost_necessary / ((PCCI_new_hanger - PCCI[5]) * 0.06 + 1e-10)

    good_hanger_number = np.sum(hanger_expected_state < 2)
    if len(hanger_expected_state) > len(Low_performance_hanger) and len(hanger_expected_state) - good_hanger_number - len(Low_performance_hanger)>0:
        hanger_ranking = np.argsort(-hanger_expected_state)
        hanger_utility_optional_1 = np.zeros((len(hanger_expected_state) - len(Low_performance_hanger) - good_hanger_number))
        hanger_maintenance_fee = np.zeros((len(hanger_expected_state) - len(Low_performance_hanger) - good_hanger_number))
        if len(Low_performance_hanger) == 0:
            hanger_cost_optional_1 = 18835.5
        else:
            hanger_cost_optional_1 = hanger_cost_necessary
        for i in range(len(hanger_expected_state) - len(Low_performance_hanger) - good_hanger_number):
            if hanger_ranking[i + len(Low_performance_hanger)] <= 21 or hanger_ranking[
                i + len(Low_performance_hanger)] > 42:
                hanger_cost_optional_1 += hanger_length[
                                            hanger_ranking[i + len(Low_performance_hanger)]] * 13.101 / 1000 * 12784.6
            else:
                hanger_cost_optional_1 += hanger_length[
                                            hanger_ranking[i + len(Low_performance_hanger)]] * 15.414 / 1000 * 12784.6
            hanger_expected_score[hanger_ranking[i + len(Low_performance_hanger)]] = 100
            if np.all(hanger_expected_score >= 60):
                PCCI_new_hanger = np.mean(hanger_expected_score) - (100 - np.min(hanger_expected_score)) / 3.88
            else:
                PCCI_new_hanger = np.min(hanger_expected_score)
            hanger_maintenance_fee[i] = hanger_cost_optional_1
            hanger_utility_optional_1[i] = hanger_cost_optional_1 / ((PCCI_new_hanger - PCCI[5]) * 0.06 + 1e-10)
        hanger_utility_less = np.argmin(hanger_utility_optional_1)
        optional_maintenance_hanger = hanger_ranking[0:hanger_utility_less + len(Low_performance_hanger) + 1]
    else:
        # all hangers are damage, the optional maintenance is identical with necessary maintenance
        hanger_utility_optional_1 = np.zeros((1))
        hanger_maintenance_fee = np.zeros((1))
        hanger_utility_less = 0
        hanger_utility_optional_1[hanger_utility_less] = hanger_utility_necessary
        hanger_maintenance_fee[hanger_utility_less] = hanger_cost_necessary
        optional_maintenance_hanger = np.where(hanger_expected_state >= 4)[0]

    # optional maintenance is worse than necessary maintenance
    if hanger_utility_optional_1[hanger_utility_less] > hanger_utility_necessary and hanger_utility_necessary > 0:
        hanger_utility_optional_1[hanger_utility_less] = hanger_utility_necessary
        hanger_maintenance_fee[hanger_utility_less] = hanger_cost_necessary
        optional_maintenance_hanger = np.where(hanger_expected_state >= 4)[0]

    for i in range(round(len(hanger_state) / 5)):
        hanger_expected_score[i] = hanger_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])
    hanger_cost_optional_2 = 0
    moderate_performance_hanger = np.where(hanger_expected_state >= 2)[0]
    if len(moderate_performance_hanger) > 0:
        hanger_cost_optional_2 += 18835.5
        for i in range(len(moderate_performance_hanger)):
            if moderate_performance_hanger[i] <= 21 or moderate_performance_hanger[i] > 42:
                hanger_cost_optional_2 += hanger_length[moderate_performance_hanger[i]] * 13.101 / 1000 * 12784.6
            else:
                hanger_cost_optional_2 += hanger_length[moderate_performance_hanger[i]] * 15.414 / 1000 * 12784.6
    hanger_expected_score[moderate_performance_hanger] = 100
    if np.all(hanger_expected_score >= 60):
        PCCI_new_hanger = np.mean(hanger_expected_score) - (100 - np.min(hanger_expected_score)) / 3.88
    else:
        PCCI_new_hanger = np.min(hanger_expected_score)
    hanger_utility_optional_2 = hanger_cost_optional_2 / ((PCCI_new_hanger - PCCI[5]) * 0.06 + 1e-10)

    # support (replacement), lifting fee is much larger than support itself. (1037 * number)+ 12399
    support_expected_state = np.zeros((round(len(support_state) / 5)))
    support_expected_score = np.zeros((round(len(support_state) / 5)))
    for i in range(round(len(support_state) / 5)):
        support_expected_state[i] = support_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        support_expected_score[i] = support_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    support_cost_necessary = 0
    Low_performance_support = np.where(support_expected_state >= 4)[0]
    if len(Low_performance_support) > 0:
        support_cost_necessary += len(Low_performance_support) * 2037 + 32399
    support_expected_score[Low_performance_support] = 100
    if np.all(support_expected_score >= 60):
        PCCI_new_support = np.mean(support_expected_score) - (100 - np.min(support_expected_score)) / 8.5
    else:
        PCCI_new_support = np.min(support_expected_score)
    support_utility_necessary = support_cost_necessary / ((PCCI_new_support - PCCI[6]) * 0.0228 + 1e-10)

    good_support_number = np.sum(support_expected_state <= 2)
    if len(support_expected_state) > len(Low_performance_support) and len(support_expected_state) - len(Low_performance_support) - good_support_number > 0:
        support_ranking = np.argsort(-support_expected_state)
        support_utility_optional_1 = np.zeros((len(support_expected_state) - len(Low_performance_support) - good_support_number))
        support_maintenance_fee = np.zeros((len(support_expected_state) - len(Low_performance_support) - good_support_number))
        if len(Low_performance_support) == 0:
            support_cost_optional_1 = 32399
        else:
            support_cost_optional_1 = support_cost_necessary
        for i in range(len(support_ranking) - len(Low_performance_support) - good_support_number):
            support_expected_score[support_ranking[i + len(Low_performance_support)]] = 100
            support_cost_optional_1 += 2037
            if np.all(support_expected_score >= 60):
                PCCI_new_support = np.mean(support_expected_score) - (100 - np.min(support_expected_score)) / 8.5
            else:
                PCCI_new_support = np.min(support_expected_score)
            support_maintenance_fee[i] = support_cost_optional_1
            support_utility_optional_1[i] = support_cost_optional_1 / ((PCCI_new_support - PCCI[6]) * 0.0228 + 1e-10)
        support_utility_less = np.argmin(support_utility_optional_1)
        optional_maintenance_support = support_ranking[0: support_utility_less + len(Low_performance_support) + 1]
    else:
        support_utility_optional_1 = np.zeros((1))
        support_maintenance_fee = np.zeros((1))
        support_utility_less = 0
        support_utility_optional_1[support_utility_less] = support_utility_necessary
        support_maintenance_fee[support_utility_less] = support_cost_necessary
        optional_maintenance_support = np.where(support_expected_state >= 4)[0]
    if support_maintenance_fee[support_utility_less] > support_utility_necessary and support_utility_necessary > 0:
        support_utility_optional_1[support_utility_less] = support_utility_necessary
        support_maintenance_fee[support_utility_less] = support_cost_necessary
        optional_maintenance_support = np.where(support_expected_state >= 4)[0]

    for i in range(round(len(support_state) / 5)):
        support_expected_score[i] = support_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])
    support_cost_optional_2 = 0
    moderate_performance_support = np.where(support_expected_state > 2)[0]
    if len(moderate_performance_support) > 0:
        support_cost_optional_2 += len(moderate_performance_support) * 2037 + 32399
    support_expected_score[moderate_performance_support] = 100
    if np.all(support_expected_score >= 60):
        PCCI_new_support = np.mean(support_expected_score) - (100 - np.min(support_expected_score)) / 8.5
    else:
        PCCI_new_support = np.min(support_expected_score)
    support_utility_optional_2 = support_cost_optional_2 / ((PCCI_new_support - PCCI[6]) * 0.0228 + 1e-10)

    # column (repair) BCCI_column
    column_utility_necessary, column_cost_necessary, column_utility_optional_1, column_cost_optional_1, column_utility_optional_2, column_cost_optional_2 = large_component_maintenance(
        column_state, BCCI[0],
        2949, bridge[33],
        np.array([0, 0.015, 0.065, 0.15, 0.3]),
        0.2068, 20,
        58980)

    # foundation (repair) BCCI_foundation
    foundation_utility_necessary, foundation_cost_necessary, foundation_utility_optional_1, foundation_cost_optional_1, foundation_utility_optional_2, foundation_cost_optional_2 = large_component_maintenance(
        foundation_state, BCCI[1],
        2996.2, bridge[34],
        np.array([0, 0.015, 0.065, 0.15, 0.3]),
        0.1932, 18,
        59924)

    """
    preventive maintenance cost-----------------------------------------------------------------------------------------
    """
    # preventive maintenance action, define a factor which maintenance fee decrease with the area increase
    # [0,0,0,1] --> (0,1,2,3), 4 years  15 years life-span
    # broken component not needs preventive maintenance action
    # state[15*(i+1):15*(i+2)])] or pavement_time_vector means the percentage of state number belong to preventive time i
    # for instenace, 5 years service time component [0,1,2,3,4] --> [0.2,0.3,0.1,0.1,0.1]
    # time vector [0.8, 0.7, 0.5, 0.8 ,0.1] means 0 year service component 80% preventive time is 0
    # 1 year service component 70% preventive time is 0
    pavement_time = np.zeros((4))
    for i in range(len(pavement_time)):
        pavement_time[i] = np.round(np.sum(bridge[0] * state[0:15] * state[15*(i+1):15*(i+2)]), 0)
    # choose the cost utility maximum scheme
    pavement_P_cost, pavement_P_utility, pavement_P_num = preventive_utility(pavement_time, 30, 25, 300)

    # [0,0,1] --> (0,1,2), 3 years   7 year life-span
    expansion_time = np.zeros((3))
    for i in range(len(expansion_time)):
        expansion_time[i] = np.round(np.sum(bridge[1] * state[75:85] * state[75+10*(i+1):75+10*(i+2)]), 0)
    expansion_P_cost, expansion_P_utility, expansion_P_num = preventive_utility(expansion_time, 2.5, 49.4, 247)

    # [0,0,0,0,1] --> (0,1,2,3,4), 5 years  32 years life-span
    sidewalk_time = np.zeros((5))
    for i in range(len(sidewalk_time)):
        sidewalk_time[i] = np.round(np.sum(bridge[2] * state[115:147] * state[115+32*(i+1):115+32*(i+2)]), 0)
    sidewalk_P_cost, sidewalk_P_utility, sidewalk_P_num = preventive_utility(sidewalk_time, 7, 36.63, 366.3)

    # [0,0,0,0,1] --> (0,1,2,3,4), 5 years  50 years life-span
    barrier_time = np.zeros((5))
    for i in range(len(barrier_time)):
        barrier_time[i] = np.round(np.sum(bridge[3] * state[307:357] * state[307+50*(i+1):307+50*(i+2)]), 0)
    barrier_P_cost, barrier_P_utility, barrier_P_num = preventive_utility(barrier_time, 7, 29.52, 295.2)

    # arch, transverse, girder, slab, hanger, support
    superstructure_P_number = np.empty(8, dtype=object)

    # area 145.16 * 0.55 * 2pi * 2= 6.911 = 1003.3 m^2
    # Transverse area 10.7 * 0.5 * 2pi = 33.6 m^2
    # arch bridge depends on the Markov chain, and 3 node for time, normalized 0-12
    # it should consider the rent of equipment and utility of maintenance years
    # coating state 1 intact, state 2 <=5%, state 3 5~10%, state 4 > 10%
    arch_time = state[680:683]
    transverse_time = state[783:803]
    arch_transverse_P_cost = 0
    arch_expected_state = np.zeros((len(arch_time)))
    for i in range(len(arch_time)):
        arch_expected_state[i] = arch_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
    transverse_expected_state = np.zeros((len(transverse_time)))
    for i in range(len(transverse_time)):
        transverse_expected_state[i] = transverse_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])

    arch_P_number = np.where(arch_time == 0)[0]
    transverse_P_number = np.where(transverse_time == 0)[0]
    superstructure_P_number[0] = arch_P_number
    superstructure_P_number[1] = transverse_P_number
    if len(arch_P_number) > 0 or len(transverse_P_number) > 0:
        arch_transverse_P_cost += (np.sum(1003.3 * 86.6 * (arch_expected_state[arch_P_number] - 1) * 0.1/3) +
                                   np.sum(33.62 * 86.6 * (transverse_expected_state[transverse_P_number] - 1) * 0.1/3) +
                                   4000)
    else:
        superstructure_P_number[0] = np.array([1000])
        superstructure_P_number[1] = np.array([1000])

    # girder area 132 * 1.75 * 1, cost 62 per m^2  132*1.75*2 + 132 = 594
    # slab area 10 * 6, cost 34.4 per m^2
    # it should consider the rent of equipment and utility of maintenance years
    # normalized 15 years, based on the standard
    pri_girder_time = state[823:827]
    sec_girder_time = state[1037:1079]
    slab_time = state[1299:1343]
    girder_slab_P_cost = 0
    pri_girder_expected_state = np.zeros((len(pri_girder_time)))
    for i in range(len(pri_girder_time)):
        pri_girder_expected_state[i] = pri_girder_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
    sec_girder_expected_state = np.zeros((len(sec_girder_time)))
    for i in range(len(sec_girder_time)):
        sec_girder_expected_state[i] = sec_girder_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
    slab_expected_state = np.zeros((len(slab_time)))
    for i in range(len(slab_time)):
        slab_expected_state[i] = slab_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])

    pri_girder_P_number = np.where(pri_girder_time == 0)[0]
    sec_girder_P_number = np.where(sec_girder_time == 0)[0]
    slab_P_number = np.where(slab_time == 0)[0]
    superstructure_P_number[2] = pri_girder_P_number
    superstructure_P_number[3] = sec_girder_P_number
    superstructure_P_number[4] = slab_P_number
    if len(pri_girder_P_number) > 0 or len(sec_girder_P_number) > 0 or len(slab_P_number) > 0:
        girder_slab_P_cost += (np.sum(132 * 4.5 * 62 * (pri_girder_expected_state[pri_girder_P_number] - 1) * 0.1 / 3) +
                               np.sum(3.98 * 10 * 62 * (sec_girder_expected_state[sec_girder_P_number] - 1) * 0.1 / 3) +
                               np.sum(60 * 34.4 * (slab_expected_state[slab_P_number] - 1) * 0.1 / 3) + 4000)
    else:
        superstructure_P_number[2] = np.array([1000])
        superstructure_P_number[3] = np.array([1000])
        superstructure_P_number[4] = np.array([1000])

    # hanger length weight mid: 14-15.2mm 15.414kg/m  side: 12-15.2mm 13.101kg/m  (1.101kg/m)
    # 63 nodes for time, normalized 0-5 years
    # note the replacement don't need preventive maintenance action
    hanger_time = state[1658:1721].copy()
    hanger_time[Low_performance_hanger] = 1
    hanger_P_cost = 0
    hanger_P_number = np.where(hanger_time == 0)[0]
    superstructure_P_number[5] = hanger_P_number
    if len(hanger_P_number) > 0:
        hanger_P_cost += 2000 # equipment cost
        for i in range(len(hanger_P_number)):
            if hanger_P_number[i] <= 21 or hanger_P_number[i] > 42:
                hanger_P_cost += hanger_length[hanger_P_number[i]] * 13.101 / 1000 * 1626.9
            else:
                hanger_P_cost += hanger_length[hanger_P_number[i]] * 15.414 / 1000 * 1626.9
    else:
        superstructure_P_number[5] = np.array([1000])

    hanger_time[optional_maintenance_hanger] = 1
    opt1_P_hanger_cost = 0
    hanger_P_opt1_number = np.where(hanger_time == 0)[0]
    superstructure_P_number[6] = hanger_P_opt1_number
    if len(hanger_P_opt1_number) > 0:
        opt1_P_hanger_cost += 2000
        for i in range(len(hanger_P_opt1_number)):
            if hanger_P_opt1_number[i] <= 21 or hanger_P_opt1_number[i] > 42:
                opt1_P_hanger_cost += hanger_length[hanger_P_opt1_number[i]] * 13.101 / 1000 * 1626.9
            else:
                opt1_P_hanger_cost += hanger_length[hanger_P_opt1_number[i]] * 15.414 / 1000 * 1626.9
    else:
        superstructure_P_number[6] = np.array([1000])

    hanger_time[moderate_performance_hanger] = 1
    opt2_P_hanger_cost = 0
    hanger_P_opt2_number = np.where(hanger_time == 0)[0]
    superstructure_P_number[7] = hanger_P_opt2_number
    if len(hanger_P_opt2_number) > 0:
        opt2_P_hanger_cost += 2000
        for i in range(len(hanger_P_opt2_number)):
            if hanger_P_opt2_number[i] <= 21 or hanger_P_opt2_number[i] > 42:
                opt2_P_hanger_cost += hanger_length[hanger_P_opt2_number[i]] * 13.101 / 1000 * 1626.9
            else:
                opt2_P_hanger_cost += hanger_length[hanger_P_opt2_number[i]] * 15.414 / 1000 * 1626.9
    else:
        superstructure_P_number[7] = np.array([1000])

    # column, foundation
    substructure_P_number = np.empty(2, dtype = object)
    # state 1 intact, state 2 <=3%, state 3 3~10%, state 4 >10%,
    # column_concrete_flaw_insite = 1328/10000 per m^2
    # area h:10 * w:3 * t:3: total area = 10 * 3 * 4 = 120 m^2
    column_time = state[1799:1805]
    foundation_time = state[1835:1841]
    substructure_P_cost = 0
    column_P_number = np.where(column_time == 0)[0]
    foundation_P_number = np.where(foundation_time == 0)[0]
    substructure_P_number[0] = column_P_number
    substructure_P_number[1] = foundation_P_number
    if len(column_P_number) > 0 or len(foundation_P_number) > 0:
        substructure_P_cost += len(column_P_number) * bridge[30] * 29.36 + len(foundation_P_number) * bridge[31] * 29.36 + 3000
    else:
        substructure_P_number[0] = np.array([1000])
        substructure_P_number[1] = np.array([1000])

    deck_M_utility = np.concatenate(([pavement_necessary_utility], [pavement_optional_utility],
                                     [expansion_necessary_utility], [expansion_optional_utility],
                                     [sidewalk_necessary_utility], [sidewalk_optional_utility],
                                     [barrier_necessary_utility], [barrier_optional_utility],
                                     [drainage_maintenance_utility], [lighting_maintenance_utility]))
    deck_M_utility = np.round(deck_M_utility, 7)

    superstructure_M_utility = np.concatenate(([arch_utility_necessary], [arch_utility_optional_1], [arch_utility_optional_2],
                                            [transverse_utility_necessary], [transverse_utility_optional_1], [transverse_utility_optional_2],
                                            [pri_girder_utility_necessary], [pri_girder_utility_optional_1], [pri_girder_utility_optional_2],
                                            [sec_girder_utility_necessary], [sec_girder_utility_optional_1], [sec_girder_utility_optional_2],
                                            [slab_utility_necessary], [slab_utility_optional_1], [slab_utility_optional_2],
                                            [hanger_utility_necessary], [hanger_utility_optional_1[hanger_utility_less]], [hanger_utility_optional_2],
                                            [support_utility_necessary], [support_utility_optional_1[support_utility_less]], [support_utility_optional_2]))
    superstructure_M_utility = np.round(superstructure_M_utility, 7)
    substructure_M_utility = np.concatenate(([column_utility_necessary], [column_utility_optional_1], [column_utility_optional_2],
                                          [foundation_utility_necessary], [foundation_utility_optional_1], [foundation_utility_optional_2]))
    substructure_M_utility = np.round(substructure_M_utility, 7)

    deck_M_cost = np.concatenate(([pavement_necessary_cost], [pavement_optional_cost], [expansion_necessary_cost],
                                  [expansion_optional_cost], [sidewalk_necessary_cost], [sidewalk_optional_cost],
                                  [barrier_necessary_cost], [barrier_optional_cost],
                                  [drainage_replacement_cost], [lighting_replacement_cost]))

    superstructure_M_cost = np.concatenate(([arch_cost_necessary], [arch_cost_optional_1], [arch_cost_optional_2],
                                            [transverse_cost_necessary], [transverse_cost_optional_1], [transverse_cost_optional_2],
                                            [pri_girder_cost_necessary], [pri_girder_cost_optional_1], [pri_girder_cost_optional_2],
                                            [sec_girder_cost_necessary], [sec_girder_cost_optional_1], [sec_girder_cost_optional_2],
                                            [slab_cost_necessary], [slab_cost_optional_1], [slab_cost_optional_2],
                                            [hanger_cost_necessary], [hanger_maintenance_fee[hanger_utility_less]], [hanger_cost_optional_2],
                                            [support_cost_necessary], [support_maintenance_fee[support_utility_less]], [support_cost_optional_2]))
    substructure_M_cost = np.concatenate(([column_cost_necessary], [column_cost_optional_1], [column_cost_optional_2],
                                          [foundation_cost_necessary], [foundation_cost_optional_1], [foundation_cost_optional_2]))

    deck_P_number = np.concatenate(
        ([pavement_P_num], [expansion_P_num], [sidewalk_P_num], [barrier_P_num]))

    deck_P_cost = np.concatenate(
        ([pavement_P_cost], [expansion_P_cost], [sidewalk_P_cost], [barrier_P_cost]))
    superstructure_P_cost = np.concatenate(([arch_transverse_P_cost], [girder_slab_P_cost], [hanger_P_cost],
                                            [opt1_P_hanger_cost], [opt2_P_hanger_cost]))

    return (deck_M_utility, superstructure_M_utility, substructure_M_utility,
            deck_M_cost, superstructure_M_cost, substructure_M_cost,
            optional_maintenance_hanger, optional_maintenance_support,
            deck_P_cost, superstructure_P_cost, substructure_P_cost,
            deck_P_number, superstructure_P_number, substructure_P_number)

"""
Maintenance planing ranking---------------------------------------------------------------------------------------------
"""


def Maintenance_list(deck_M_utility, superstructure_M_utility, substructure_M_utility,
                     deck_M_cost, superstructure_M_cost, substructure_M_cost,
                     deck_P_cost, superstructure_P_cost, substructure_P_cost, action):
    """
    :param deck_M_utility:
    :param superstructure_M_utility:
    :param substructure_M_utility:
    :param deck_M_cost:
    :param superstructure_M_cost:
    :param substructure_M_cost:
    :param optional_maintenance_hanger:
    :param optional_maintenance_support:
    :param deck_P_cost:
    :param superstructure_P_cost:
    :param substructure_P_cost:
    :param action: a vector of action index
    :return: the final cost and reward of action
    """

    action_index = action_ecoding(action)

    """
    necessary maintenance list and cost---------------------------------------------------------------------------------
    """
    # judge which component repair is necessary
    if deck_M_utility[0] == 0:
        pavement_necessary_booling = 0
    else:
        pavement_necessary_booling = 1
    if deck_M_utility[2] == 0:
        expansion_necessary_booling = 0
    else:
        expansion_necessary_booling = 1
    if deck_M_utility[4] == 0:
        sidewalk_necessary_booling = 0
    else:
        sidewalk_necessary_booling = 1
    if deck_M_utility[6] == 0:
        barrier_necessary_booling = 0
    else:
        barrier_necessary_booling = 1
    if superstructure_M_utility[0] == 0:
        arch_necessary_booling = 0
    else:
        arch_necessary_booling = 1
    if superstructure_M_utility[3] == 0:
        transverse_necessary_booling = 0
    else:
        transverse_necessary_booling = 1
    if superstructure_M_utility[6] == 0:
        pri_girder_necessary_booling = 0
    else:
        pri_girder_necessary_booling = 1
    if superstructure_M_utility[9] == 0:
        sec_girder_necessary_booling = 0
    else:
        sec_girder_necessary_booling = 1
    if superstructure_M_utility[12] == 0:
        slab_necessary_booling = 0
    else:
        slab_necessary_booling = 1
    if superstructure_M_utility[15] == 0:
        hanger_necessary_booling = 0
    else:
        hanger_necessary_booling = 1
    if superstructure_M_utility[18] == 0:
        support_necessary_booling = 0
    else:
        support_necessary_booling = 1
    if substructure_M_utility[0] == 0:
        column_necessary_booling = 0
    else:
        column_necessary_booling = 1
    if substructure_M_utility[3] == 0:
        foundation_necessary_booling = 0
    else:
        foundation_necessary_booling = 1

    necessary_booling = np.array(
        [pavement_necessary_booling, expansion_necessary_booling, sidewalk_necessary_booling, barrier_necessary_booling,
         0, 0, arch_necessary_booling,
         transverse_necessary_booling, pri_girder_necessary_booling, sec_girder_necessary_booling,
         slab_necessary_booling, hanger_necessary_booling, support_necessary_booling,
         column_necessary_booling, foundation_necessary_booling])

    necessary_cost = np.array([deck_M_cost[0], deck_M_cost[2], deck_M_cost[4], deck_M_cost[6], 0, 0, superstructure_M_cost[0],
                               superstructure_M_cost[3], superstructure_M_cost[6], superstructure_M_cost[9],
                               superstructure_M_cost[12], superstructure_M_cost[15], superstructure_M_cost[18],
                               substructure_M_cost[0], substructure_M_cost[3]])

    necessary_utility = np.array(
        [deck_M_utility[0], deck_M_utility[2], deck_M_utility[4], deck_M_utility[6], deck_M_utility[8],
         deck_M_utility[9], superstructure_M_utility[0], superstructure_M_utility[3],
         superstructure_M_utility[6], superstructure_M_utility[9],
         superstructure_M_utility[12], superstructure_M_utility[15], superstructure_M_utility[18],
         substructure_M_utility[0], substructure_M_utility[3]])

    sum_necessary_cost = np.sum(necessary_booling * necessary_cost)
    necessary_utility_list = necessary_utility * necessary_booling
    necessary_utility_index = np.nonzero(necessary_utility_list)[0]

    if action_index[-1] < len(necessary_utility_index):
        effective_list = necessary_utility_list[necessary_utility_index]
        ranking_necessary_maintenance = effective_list.argsort()

        maintenance_necessary_list = necessary_utility_index[ranking_necessary_maintenance[0: action_index[-1]]]
        maintenance_fee = np.sum(necessary_cost[maintenance_necessary_list])
        maintenance_optional_list = []
        reward = - 1
    else:
        maintenance_necessary_list = necessary_utility_index.copy()
        # judge which component can be repaired
        if deck_M_utility[1] > 1e10:
            pavement_optional_booling = 1
        else:
            pavement_optional_booling = 0
        if deck_M_utility[3] > 1e10:
            expansion_optional_booling = 1
        else:
            expansion_optional_booling = 0
        if deck_M_utility[5] > 1e10:
            sidewalk_optional_booling = 1
        else:
            sidewalk_optional_booling = 0
        if deck_M_utility[7] > 1e10:
            barrier_optional_booling = 1
        else:
            barrier_optional_booling = 0
        if deck_M_utility[8] == 0:
            drainage_optional_booling = 1
        else:
            drainage_optional_booling = 0
        if deck_M_utility[9] == 0:
            lighting_optional_booling = 1
        else:
            lighting_optional_booling = 0

        if superstructure_M_utility[1] == 0 or superstructure_M_utility[0] == superstructure_M_utility[1]:
            arch_optional_booling_1 = 1
        else:
            arch_optional_booling_1 = 0
        if superstructure_M_utility[2] == 0 or superstructure_M_utility[0] == superstructure_M_utility[2] or superstructure_M_utility[1] == superstructure_M_utility[2]:
            arch_optional_booling_2 = 1
        else:
            arch_optional_booling_2 = 0

        if superstructure_M_utility[4] == 0 or superstructure_M_utility[3] == superstructure_M_utility[4]:
            transverse_optional_booling_1 = 1
        else:
            transverse_optional_booling_1 = 0
        if superstructure_M_utility[5] == 0 or superstructure_M_utility[3] == superstructure_M_utility[5] or superstructure_M_utility[4] == superstructure_M_utility[5]:
            transverse_optional_booling_2 = 1
        else:
            transverse_optional_booling_2 = 0

        if superstructure_M_utility[7] == 0 or superstructure_M_utility[6] == superstructure_M_utility[7]:
            pri_girder_optional_booling_1 = 1
        else:
            pri_girder_optional_booling_1 = 0
        if superstructure_M_utility[8] == 0 or superstructure_M_utility[6] == superstructure_M_utility[8] or superstructure_M_utility[7] == superstructure_M_utility[8]:
            pri_girder_optional_booling_2 = 1
        else:
            pri_girder_optional_booling_2 = 0

        if superstructure_M_utility[10] == 0 or superstructure_M_utility[9] == superstructure_M_utility[10]:
            sec_girder_optional_booling_1 = 1
        else:
            sec_girder_optional_booling_1 = 0
        if superstructure_M_utility[11] == 0 or superstructure_M_utility[9] == superstructure_M_utility[11] or superstructure_M_utility[10] == superstructure_M_utility[11]:
            sec_girder_optional_booling_2 = 1
        else:
            sec_girder_optional_booling_2 = 0

        if superstructure_M_utility[13] == 0 or superstructure_M_utility[12] == superstructure_M_utility[13]:
            slab_optional_booling_1 = 1
        else:
            slab_optional_booling_1 = 0
        if superstructure_M_utility[14] == 0 or superstructure_M_utility[12] == superstructure_M_utility[14] or superstructure_M_utility[13] == superstructure_M_utility[14]:
            slab_optional_booling_2 = 1
        else:
            slab_optional_booling_2 = 0

        if superstructure_M_utility[16] > 1e10 or superstructure_M_utility[15] == superstructure_M_utility[16]:
            hanger_optional_booling_1 = 1
        else:
            hanger_optional_booling_1 = 0
        if superstructure_M_utility[17] == 0 or superstructure_M_utility[15] == superstructure_M_utility[17] or superstructure_M_utility[16] == superstructure_M_utility[17]:
            hanger_optional_booling_2 = 1
        else:
            hanger_optional_booling_2 = 0

        if superstructure_M_utility[19] > 1e10 or superstructure_M_utility[18] == superstructure_M_utility[19]:
            support_optional_booling_1 = 1
        else:
            support_optional_booling_1 = 0
        if superstructure_M_utility[20] == 0 or superstructure_M_utility[18] == superstructure_M_utility[20] or superstructure_M_utility[19] == superstructure_M_utility[20]:
            support_optional_booling_2 = 1
        else:
            support_optional_booling_2 = 0

        if substructure_M_utility[1] == 0 or substructure_M_utility[0] == substructure_M_utility[1]:
            column_optional_booling_1 = 1
        else:
            column_optional_booling_1 = 0
        if substructure_M_utility[2] == 0 or substructure_M_utility[0] == substructure_M_utility[2] or substructure_M_utility[1] == substructure_M_utility[2]:
            column_optional_booling_2 = 1
        else:
            column_optional_booling_2 = 0

        if substructure_M_utility[4] == 0 or substructure_M_utility[3] == substructure_M_utility[4]:
            foundation_optional_booling_1 = 1
        else:
            foundation_optional_booling_1 = 0
        if substructure_M_utility[5] == 0 or substructure_M_utility[3] == substructure_M_utility[5] or substructure_M_utility[4] == substructure_M_utility[5]:
            foundation_optional_booling_2 = 1
        else:
            foundation_optional_booling_2 = 0

        # the budget cover the necessary maintenance fee, the situation change to optional maintenance
        optional_booling = np.array(
            [pavement_optional_booling, expansion_optional_booling, sidewalk_optional_booling, barrier_optional_booling,
             drainage_optional_booling, lighting_optional_booling, arch_optional_booling_1, transverse_optional_booling_1,
             pri_girder_optional_booling_1, sec_girder_optional_booling_1, slab_optional_booling_1,
             hanger_optional_booling_1, support_optional_booling_1,
             column_optional_booling_1, foundation_optional_booling_1,
             arch_optional_booling_2, transverse_optional_booling_2,
             pri_girder_optional_booling_2, sec_girder_optional_booling_2, slab_optional_booling_2, hanger_optional_booling_2,
             support_optional_booling_2,
             column_optional_booling_2, foundation_optional_booling_2])

        optional_cost = np.array(
            [deck_M_cost[1] - deck_M_cost[0], deck_M_cost[3] - deck_M_cost[2], deck_M_cost[5] - deck_M_cost[4],
             deck_M_cost[7] - deck_M_cost[6], deck_M_cost[8], deck_M_cost[9],
             superstructure_M_cost[1] - superstructure_M_cost[0], superstructure_M_cost[4] - superstructure_M_cost[3],
             superstructure_M_cost[7] - superstructure_M_cost[6], superstructure_M_cost[10] - superstructure_M_cost[9],
             superstructure_M_cost[13] - superstructure_M_cost[12], superstructure_M_cost[16] - superstructure_M_cost[15],
             superstructure_M_cost[19] - superstructure_M_cost[18],
             substructure_M_cost[1] - substructure_M_cost[0], substructure_M_cost[4] - substructure_M_cost[3],
             superstructure_M_cost[2] - superstructure_M_cost[0], superstructure_M_cost[5] - superstructure_M_cost[3],
             superstructure_M_cost[8] - superstructure_M_cost[6], superstructure_M_cost[11] - superstructure_M_cost[9],
             superstructure_M_cost[14] - superstructure_M_cost[12], superstructure_M_cost[17] - superstructure_M_cost[15],
             superstructure_M_cost[20] - superstructure_M_cost[18],
             substructure_M_cost[2] - substructure_M_cost[0], substructure_M_cost[5] - substructure_M_cost[3]])

        optional_utility = np.array(
            [deck_M_utility[1], deck_M_utility[3], deck_M_utility[5], deck_M_utility[7], deck_M_utility[8],
             deck_M_utility[9], superstructure_M_utility[1], superstructure_M_utility[4], superstructure_M_utility[7],
             superstructure_M_utility[10], superstructure_M_utility[13], superstructure_M_utility[16],
             superstructure_M_utility[19],
             substructure_M_utility[1], substructure_M_utility[4],
             superstructure_M_utility[2], superstructure_M_utility[5], superstructure_M_utility[8],
             superstructure_M_utility[11], superstructure_M_utility[14], superstructure_M_utility[17],
             superstructure_M_utility[20],
             substructure_M_utility[2], substructure_M_utility[5]])

        optional_utility_index = np.where(optional_booling == 0)[0]
        optional_effective_list = optional_utility[optional_utility_index]
        ranking_optional_maintenance = optional_effective_list.argsort()

        if action_index[-1] - len(necessary_utility_index) <= len(optional_utility_index):
            maintenance_optional_list = optional_utility_index[
                ranking_optional_maintenance[0: action_index[-1] - len(necessary_utility_index)]]
            maintenance_fee = sum_necessary_cost
            for i in range(action_index[-1] - len(necessary_utility_index)):
                if optional_utility_index[ranking_optional_maintenance[i]] > 14 and optional_utility_index[ranking_optional_maintenance[i]] - 9 in maintenance_optional_list:
                    maintenance_fee += optional_cost[optional_utility_index[ranking_optional_maintenance[i]]] - \
                                       optional_cost[optional_utility_index[ranking_optional_maintenance[i]] - 9]
                else:
                    maintenance_fee += optional_cost[optional_utility_index[ranking_optional_maintenance[i]]]
            reward = 0
        else:
            maintenance_optional_list = optional_utility_index.copy()
            maintenance_fee = sum_necessary_cost
            for i in range(len(optional_utility_index)):
                if optional_utility_index[ranking_optional_maintenance[i]] > 14 and optional_utility_index[ranking_optional_maintenance[i]] - 9 in maintenance_optional_list:
                    maintenance_fee += optional_cost[optional_utility_index[ranking_optional_maintenance[i]]] - \
                                       optional_cost[optional_utility_index[ranking_optional_maintenance[i]] - 9]
                else:
                    maintenance_fee += optional_cost[optional_utility_index[ranking_optional_maintenance[i]]]
            reward = -1

    """
    preventive maintenance cost-----------------------------------------------------------------------------------------
    """
    if 0 in maintenance_optional_list:
        deck_P_cost[0] = 0
    if 1 in maintenance_optional_list:
        deck_P_cost[1] = 0
    if 2 in maintenance_optional_list:
        deck_P_cost[2] = 0
    if 3 in maintenance_optional_list:
        deck_P_cost[3] = 0
    P_cost_list = np.array(
        [np.sum(deck_P_cost), superstructure_P_cost[0], superstructure_P_cost[1], superstructure_P_cost[2],
         substructure_P_cost])
    if 20 in maintenance_optional_list:
        P_cost_list[3] = superstructure_P_cost[4]
    elif 11 in maintenance_optional_list:
        P_cost_list[3] = superstructure_P_cost[3]

    preventive_action = action_index[1:6]
    preventive_cost = np.sum(preventive_action * P_cost_list)

    """
    Inspection cost-----------------------------------------------------------------------------------------------------
    """
    inspection_cost = 0
    if action_index[0] == 1:
        inspection_cost += 30000

    return maintenance_necessary_list, maintenance_optional_list, maintenance_fee, preventive_cost, inspection_cost, reward

"""
environment-------------------------------------------------------------------------------------------------------------
"""


class environment():
    """this part define the bridge degradation process"""
    def __init__(self):
        """fundamental parameters of structure or bridges"""

        self.state_number = 5

        # Weibull model for pavement system
        self.pavement_nor = np.array([8.8436, 3.4663, 5.0489])
        self.pavement_deter = np.array([8.0323, 3.1486, 3.8192])

        # Weilbull model for expansion system
        self.expansion_nor = np.array([8.5684, 3.5452, 1.2834])
        self.expansion_deter = np.array([6.3232, 3.8387, 1.3725])

        # Weilbull model for sidewalk
        self.sidewalk_nor = np.array([24.2879,	2.9871, 11.0581])
        self.sidewalk_deter = np.array([17.5529, 3.5067, 11.2052])

        # Weilbull model for barrier
        self.barrier_nor = np.array([21.1525, 3.1065, 16.6534])
        self.barrier_deter = np.array([16.3822, 3.4021, 14.9368])

        # Weibull model for drainage
        self.drainage_nor = np.array([19.2600, 2.5389, 10.9171])

        # Weibull model for lighting
        self.lighting_nor = np.array([5.5492, 2.9615, 2.0569])

        # Markov model for steel arch, no support material, 40-70 years, expected 60 years
        # 5 states, (0, 0, 0, 0, 0, 0), (5, 0.5, 2.5, 0, 0, 0), (15, 1.5, 7.5, 1, 15, 0), (25, 2.5, 15, 3, 45, 0),
        # (25, 2.5, 15, 3, 45, 1)
        # [5.67, 18, 26.33, 10]
        self.arch = np.array([5.67, 18, 26.33, 10])
        self.arch_det = 3
        self.arch_matrix = Markov_state_transition_matrix(self.state_number, self.arch)
        self.arch_matrix_D = Markov_state_transition_matrix(self.state_number, self.arch / self.arch_det)

        # Markov model for transverse, no support material, 30-60 years, expected 40 years
        # [3.78, 12, 17.55, 6.67]
        self.transverse = np.array([3.78, 12, 17.55, 6.67])
        self.transverse_det = 2.8
        self.transverse_matrix = Markov_state_transition_matrix(self.state_number, self.transverse)
        self.transverse_matrix_D = Markov_state_transition_matrix(self.state_number,
                                                                  self.transverse / self.transverse_det)

        # Markov model for primary girder and secondary girder, 40-60 years, expected 50, 5 states
        # (0,0,0,0,0), (10, 0.5, 1/6, 0, 0), (30, 1.5, 1/2, 0, 0), (30, 1.5, 5/6, 1, 0), (30, 1.5, 5/6, 2, 1)
        # [8.67, 17.33, 9, 15]
        self.girder = np.array([8.67, 17.33, 15, 9])
        self.girder_det = 2
        self.girder_matrix = Markov_state_transition_matrix(self.state_number, self.girder)
        self.girder_matrix_D = Markov_state_transition_matrix(self.state_number, self.girder / self.girder_det)

        # Markov model for slab, 20-40 years, expected 30, 5 states,
        # (0,0,0,0,0), (10, 0.5, 1/6, 0, 0), (30, 1.5, 1/2, 0, 0), (30, 1.5, 5/6, 1, 0), (30, 1.5, 5/6, 2, 1)
        # [5.2, 10.4, 5.4, 9]
        self.slab = np.array([5.2, 10.4, 9, 5.4])
        self.slab_det = 1.9
        self.slab_matrix = Markov_state_transition_matrix(self.state_number, self.slab)
        self.slab_matrix_D = Markov_state_transition_matrix(self.state_number, self.slab / self.slab_det)

        # Markov model for hanger, middle hanger and edge hanger should consider an increase factor
        # related to traffic, in this case, 10-30 years, expected 25
        # 5 states (0, 0), (1.5, 0), (6.5, 0), (15, 1), (15, 3), [0.75, 2.5, 6.75, 5]
        self.hanger = np.array([4.58, 7.5, 7.92, 5])
        self.hanger_det = 3.4
        self.hanger_matrix = Markov_state_transition_matrix(self.state_number, self.hanger)
        self.hanger_matrix_D = Markov_state_transition_matrix(self.state_number, self.hanger / self.hanger_det)

        # Markov model for support 10-40 years, in this case, 25 years, 5 states,
        # (0, 0), (0.5, 10), (1.5, 25), (2.5, 25), (2,5, 50), [5, 6, 4, 5]
        self.support = np.array([6.25, 7.5, 6.25, 5])
        self.support_det = 1
        self.support_matrix = Markov_state_transition_matrix(self.state_number, self.support)
        self.support_matrix_D = Markov_state_transition_matrix(self.state_number, self.support / self.support_det)

        # Markov model for column, 30-60 years, expected 50 years, 5 states, 0, 1/16, 5/16, 9/16, [5.125, 12.5, 16.5, 15.875]
        self.column = np.array([5.125, 12.5, 16.5, 15.875])
        self.column_det = 1.9
        self.column_matrix = Markov_state_transition_matrix(self.state_number, self.column)
        self.column_matrix_D = Markov_state_transition_matrix(self.state_number, self.column / self.column_det)

        # Markov model for foundation, 40-80 years, expected 50 years, 5 states, 0, 1.5, 6.5, 15, 25 [1.5, 5, 8.5, 10]
        self.foundation = np.array([6.33, 13.33, 17, 13.34])
        self.foundation_det = 2.1
        self.foundation_matrix = Markov_state_transition_matrix(self.state_number, self.foundation)
        self.foundation_matrix_D = Markov_state_transition_matrix(self.state_number,
                                                                  self.foundation / self.foundation_det)

        # observation matrix
        self.accuracy_visual = 0.6
        self.accuracy_NDT = 0.99

        # defined repair or replace action state transition matrix
        self.repair_matrix = np.zeros((self.state_number, self.state_number))
        self.repair_matrix[:, 0] = 1

    def step(self, bridge_infor, states, actions, hidden_state):
        """
        :param bridge_infor: the constitution of bridge component
        :param states: last item is time with one-hot
        :param actions:
        :param hidden_state: 0-4, [3 + 20 + 4 + 44 + 63 + 6 + 6 + 6]
        :return: new_state, reward, new_hidden_state
        """
        # obtain the state information from inputting vector
        state = states.flatten()
        pavement_state = state[0:15]
        pavement_time = state[15:75]

        expansion_state = state[75:85]
        expansion_time = state[85:115]

        sidewalk_state = state[115:147]
        sidewalk_time = state[147:307]

        barrier_state = state[307:357]
        barrier_time = state[357:607]

        drainage_state = state[607:657]

        lighting_state = state[657:665]

        arch_state = state[665:680]
        arch_expected_state = np.zeros((round(len(arch_state) / 5)))
        for i in range(round(len(arch_state) / 5)):
            arch_expected_state[i] = arch_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        arch_time = state[680:683]
        arch_hidden = hidden_state[0:3]

        transverse_state = state[683:783]
        transverse_expected_state = np.zeros((round(len(transverse_state) / 5)))
        for i in range(round(len(transverse_state) / 5)):
            transverse_expected_state[i] = transverse_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        transverse_time = state[783:803]
        transverse_hidden = hidden_state[3:23]

        pri_girder_state = state[803:823]
        pri_girder_expected_state = np.zeros((round(len(pri_girder_state) / 5)))
        for i in range(round(len(pri_girder_state) / 5)):
            pri_girder_expected_state[i] = pri_girder_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        pri_girder_time = state[823:827]
        pri_girder_hidden = hidden_state[23:27]

        sec_girder_state = state[827:1037]
        sec_girder_expected_state = np.zeros((round(len(sec_girder_state) / 5)))
        for i in range(round(len(sec_girder_state) / 5)):
            sec_girder_expected_state[i] = sec_girder_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        sec_girder_time = state[1037:1079]
        sec_girder_hidden = hidden_state[27:69]

        slab_state = state[1079:1299]
        slab_expected_state = np.zeros((round(len(slab_state) / 5)))
        for i in range(round(len(slab_state) / 5)):
            slab_expected_state[i] = slab_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        slab_time = state[1299:1343]
        slab_hidden = hidden_state[69:113]

        hanger_state = state[1343:1658]
        hanger_expected_state = np.zeros((round(len(hanger_state) / 5)))
        for i in range(round(len(hanger_state) / 5)):
            hanger_expected_state[i] = hanger_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        hanger_time = state[1658:1721]
        hanger_hidden = hidden_state[113:176]

        support_state = state[1721:1761]
        support_expected_state = np.zeros((round(len(support_state) / 5)))
        for i in range(round(len(support_state) / 5)):
            support_expected_state[i] = support_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        support_time = state[1761:1769]
        support_hidden = hidden_state[176:184]

        column_state = state[1769:1799]
        column_expected_state = np.zeros((round(len(column_state) / 5)))
        for i in range(round(len(column_state) / 5)):
            column_expected_state[i] = column_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        column_time = state[1799:1805]
        column_hidden = hidden_state[184:190]

        foundation_state = state[1805:1835]
        foundation_expected_state = np.zeros((round(len(foundation_state) / 5)))
        for i in range(round(len(foundation_state) / 5)):
            foundation_expected_state[i] = foundation_state[5 * i:5 * (i + 1)] @ np.array([1, 2, 3, 4, 5])
        foundation_time = state[1835:1841]
        foundation_hidden = hidden_state[190:196]

        time = tf.argmax(state[1841:1941]).numpy()

        DMCI, PCCI, BCCI, Bridge_state = state_ecoding(state)

        (deck_M_utility, superstructure_M_utility, substructure_M_utility,
         deck_M_cost, superstructure_M_cost, substructure_M_cost,
         optional_maintenance_hanger, optional_maintenance_support,
         deck_P_cost, superstructure_P_cost, substructure_P_cost,
         deck_P_number, superstructure_P_number, substructure_P_number) = cost_cal(state, bridge_infor, DMCI, PCCI, BCCI)

        maintenance_necessary_list, maintenance_optional_list, maintenance_fee, preventive_cost, inspection_cost, reward = Maintenance_list(
            deck_M_utility, superstructure_M_utility, substructure_M_utility,
            deck_M_cost, superstructure_M_cost, substructure_M_cost,
            deck_P_cost, superstructure_P_cost, substructure_P_cost, actions)

        pavement_infor = np.zeros((round(len(pavement_time) / len(pavement_state) + 1), len(pavement_state)))
        for i in range(round(len(pavement_time) / len(pavement_state))):
            pavement_infor[i, :] = np.round(bridge_infor[0] * pavement_state * pavement_time[
                                                               i * len(pavement_state): (i + 1) * len(pavement_state)])

        expansion_infor = np.zeros((round(len(expansion_time) / len(expansion_state) + 1), len(expansion_state)))
        for i in range(round(len(expansion_time) / len(expansion_state))):
            expansion_infor[i, :] = np.round(bridge_infor[1] * expansion_state * expansion_time[
                                                               i * len(expansion_state): (i + 1) * len(expansion_state)])

        sidewalk_infor = np.zeros((round(len(sidewalk_time) / len(sidewalk_state) + 1), len(sidewalk_state)))
        for i in range(round(len(sidewalk_time) / len(sidewalk_state))):
            sidewalk_infor[i, :] = np.round(bridge_infor[2] * sidewalk_state * sidewalk_time[
                                                               i * len(sidewalk_state): (i + 1) * len(sidewalk_state)])

        barrier_infor = np.zeros((round(len(barrier_time) / len(barrier_state) + 1), len(barrier_state)))
        for i in range(round(len(barrier_time) / len(barrier_state))):
            barrier_infor[i, :] = np.round(bridge_infor[3] * barrier_state * barrier_time[
                                                               i * len(barrier_state): (i + 1) * len(barrier_state)])

        # maintenance_necessary_list and maintenance_optional_list is the index which components are repaired or replanced
        # it should notice that the component first experience repair, Then, working a year
        # the repair is transient compared a year
        """
        Repair or replace component based on the action-----------------------------------------------------------------
        """
        # 0 is pavement, 1 is expansion, 2 is sidewalk, 3 is barrier, 4 is drainage, 5 is lighting
        # 6 is arch, 7 is transverse, 8 is primary girder, 9 is sec girder, 10 is slab, 11 is hanger, 12 is support
        # 13 is column, 14 is foundation
        # pavement replacement the damage and reset the preventive time to 0; [0:15]
        # state start with the service year 1, not 0
        # for instance, life-span is 5, means 0->1, 1->2, 2->3, 3->4, 4->5, state at least 1
        # print('maintenance_optional_list',maintenance_optional_list)
        # print('maintenance_necessary_list',maintenance_necessary_list)
        if 0 in maintenance_optional_list:
            pavement_rep = bridge_infor[0]
            pavement_infor = np.zeros((round(len(pavement_time) / len(pavement_state) + 1), len(pavement_state)))
        elif 0 in maintenance_necessary_list:
            pavement_rep = round(bridge_infor[0] * (1 - np.sum(pavement_state)))
        else:
            pavement_rep = 0
        # expansion repair the damage and reset the preventive time to 0; [0:7]
        if 1 in maintenance_optional_list:
            expansion_rep = bridge_infor[1]
            expansion_infor = np.zeros((round(len(expansion_time) / len(expansion_state) + 1), len(expansion_state)))
        elif 1 in maintenance_necessary_list:
            expansion_rep = round(bridge_infor[1] * (1 - np.sum(expansion_state)))
        else:
            expansion_rep = 0
        # sidewalk; [0:32]
        if 2 in maintenance_optional_list:
            sidewalk_rep = bridge_infor[2]
            sidewalk_infor = np.zeros((round(len(sidewalk_time) / len(sidewalk_state) + 1), len(sidewalk_state)))
        elif 2 in maintenance_necessary_list:
            sidewalk_rep = round(bridge_infor[2] * (1 - np.sum(sidewalk_state)))
        else:
            sidewalk_rep = 0
        # barrier; [0:50]
        if 3 in maintenance_optional_list:
            barrier_rep = bridge_infor[3]
            barrier_infor = np.zeros((round(len(barrier_time) / len(barrier_state) + 1), len(barrier_state)))
        elif 3 in maintenance_necessary_list:
            barrier_rep = round(bridge_infor[3] * (1 - np.sum(barrier_state)))
        else:
            barrier_rep = 0
        # drainage; [0:50]
        if 4 in maintenance_optional_list or 4 in maintenance_necessary_list:
            drainage_rep = round(bridge_infor[4] * (1 - np.sum(drainage_state)))
        else:
            drainage_rep = 0
        # lighting; [0:8]
        if 5 in maintenance_optional_list or 5 in maintenance_necessary_list:
            lighting_rep = round(bridge_infor[5] * (1 - np.sum(lighting_state)))
        else:
            lighting_rep = 0

        # arch; [0:15], 3
        if 15 in maintenance_optional_list:
            arch_repair_index = np.where(arch_expected_state > 2)[0]
            for i in arch_repair_index:
                arch_state[i*5: i*5 + 5] = arch_state[i*5 : i*5 + 5] @ self.repair_matrix
                arch_hidden[i] = 0
        elif 6 in maintenance_optional_list:
            arch_repair_index = np.where(arch_expected_state >= 3)[0]
            for i in arch_repair_index:
                arch_state[i * 5: i * 5 + 5] = arch_state[i * 5: i * 5 + 5] @ self.repair_matrix
                arch_hidden[i] = 0
        elif 6 in maintenance_necessary_list and 6 not in maintenance_optional_list:
            arch_repair_index = np.where(arch_expected_state >= 4)[0]
            for i in arch_repair_index:
                arch_state[i * 5: i * 5 + 5] = arch_state[i * 5: i * 5 + 5] @ self.repair_matrix
                arch_hidden[i] = 0

        # transverse; [0:100], 20
        if 16 in maintenance_optional_list:
            transverse_repair_index = np.where(transverse_expected_state > 2)[0]
            for i in transverse_repair_index:
                transverse_state[i*5: i*5 + 5] = transverse_state[i*5 : i*5 + 5] @ self.repair_matrix
                transverse_hidden[i] = 0
        elif 7 in maintenance_optional_list:
            transverse_repair_index = np.where(transverse_expected_state >= 3)[0]
            for i in transverse_repair_index:
                transverse_state[i * 5: i * 5 + 5] = transverse_state[i * 5: i * 5 + 5] @ self.repair_matrix
                transverse_hidden[i] = 0
        elif 7 not in maintenance_optional_list and 7 in maintenance_necessary_list:
            transverse_repair_index = np.where(transverse_expected_state >= 4)[0]
            for i in transverse_repair_index:
                transverse_state[i * 5: i * 5 + 5] = transverse_state[i * 5: i * 5 + 5] @ self.repair_matrix
                transverse_hidden[i] = 0

        # primary girder; [0:20], 4
        if 17 in maintenance_optional_list:
            pri_girder_repair_index = np.where(pri_girder_expected_state > 2)[0]
            for i in pri_girder_repair_index:
                pri_girder_state[i * 5: i * 5 + 5] = pri_girder_state[i * 5: i * 5 + 5] @ self.repair_matrix
                pri_girder_hidden[i] = 0
        elif 8 in maintenance_optional_list:
            pri_girder_repair_index = np.where(pri_girder_expected_state >= 3)[0]
            for i in pri_girder_repair_index:
                pri_girder_state[i * 5: i * 5 + 5] = pri_girder_state[i * 5: i * 5 + 5] @ self.repair_matrix
                pri_girder_hidden[i] = 0
        elif 8 not in maintenance_optional_list and 8 in maintenance_necessary_list:
            pri_girder_repair_index = np.where(pri_girder_expected_state >= 4)[0]
            for i in pri_girder_repair_index:
                pri_girder_state[i * 5: i * 5 + 5] = pri_girder_state[i * 5: i * 5 + 5] @ self.repair_matrix
                pri_girder_hidden[i] = 0

        # secondary grider; [0:210], 42
        if 18 in maintenance_optional_list:
            sec_girder_repair_index = np.where(sec_girder_expected_state > 2)[0]
            for i in sec_girder_repair_index:
                sec_girder_state[i * 5: i * 5 + 5] = sec_girder_state[i * 5: i * 5 + 5] @ self.repair_matrix
                sec_girder_hidden[i] = 0
        elif 9 in maintenance_optional_list:
            sec_girder_repair_index = np.where(sec_girder_expected_state >= 3)[0]
            for i in sec_girder_repair_index:
                sec_girder_state[i * 5: i * 5 + 5] = sec_girder_state[i * 5: i * 5 + 5] @ self.repair_matrix
                sec_girder_hidden[i] = 0
        elif 9 not in maintenance_optional_list and 9 in maintenance_necessary_list:
            sec_girder_repair_index = np.where(sec_girder_expected_state >= 4)[0]
            for i in sec_girder_repair_index:
                sec_girder_state[i * 5: i * 5 + 5] = sec_girder_state[i * 5: i * 5 + 5] @ self.repair_matrix
                sec_girder_hidden[i] = 0

        # slab; [0:220], 44
        if 19 in maintenance_optional_list:
            slab_repair_index = np.where(slab_expected_state > 2)[0]
            for i in slab_repair_index:
                slab_state[i * 5: i * 5 + 5] = slab_state[i * 5: i * 5 + 5] @ self.repair_matrix
                slab_hidden[i] = 0
        elif 10 in maintenance_optional_list:
            slab_repair_index = np.where(slab_expected_state >= 3)[0]
            for i in slab_repair_index:
                slab_state[i * 5: i * 5 + 5] = slab_state[i * 5: i * 5 + 5] @ self.repair_matrix
                slab_hidden[i] = 0
        elif 10 not in maintenance_optional_list and 10 in maintenance_necessary_list:
            slab_repair_index = np.where(slab_expected_state >= 4)[0]
            for i in slab_repair_index:
                slab_state[i * 5: i * 5 + 5] = slab_state[i * 5: i * 5 + 5] @ self.repair_matrix
                slab_hidden[i] = 0

        # hanger; [0:315], 63
        if 20 in maintenance_optional_list:
            hanger_repair_index = np.where(hanger_expected_state > 2)[0]
            for i in hanger_repair_index:
                hanger_state[i * 5: i * 5 + 5] = hanger_state[i * 5: i * 5 + 5] @ self.repair_matrix
                hanger_hidden[i] = 0
                hanger_time[i] = 1
        elif 11 in maintenance_optional_list:
            for i in optional_maintenance_hanger:
                hanger_state[i * 5: i * 5 + 5] = hanger_state[i * 5: i * 5 + 5] @ self.repair_matrix
                hanger_hidden[i] = 0
                hanger_time[i] = 1
        elif 11 not in maintenance_optional_list and 11 in maintenance_necessary_list:
            hanger_repair_index = np.where(hanger_expected_state >= 4)[0]
            for i in hanger_repair_index:
                hanger_state[i * 5: i * 5 + 5] = hanger_state[i * 5: i * 5 + 5] @ self.repair_matrix
                hanger_hidden[i] = 0
                hanger_time[i] = 1

        # support; [0:40], 8
        if 21 in maintenance_optional_list:
            support_repair_index = np.where(support_expected_state > 2)[0]
            for i in support_repair_index:
                support_state[i * 5: i * 5 + 5] = support_state[i * 5: i * 5 + 5] @ self.repair_matrix
                support_hidden[i] = 0
                support_time[i] = 1
        elif 12 in maintenance_optional_list:
            for i in optional_maintenance_support:
                support_state[i * 5: i * 5 + 5] = support_state[i * 5: i * 5 + 5] @ self.repair_matrix
                support_hidden[i] = 0
                support_time[i] = 1
        elif 12 not in maintenance_optional_list and 12 in maintenance_necessary_list:
            support_repair_index = np.where(support_expected_state >= 4)[0]
            for i in support_repair_index:
                support_state[i * 5: i * 5 + 5] = support_state[i * 5: i * 5 + 5] @ self.repair_matrix
                support_hidden[i] = 0
                support_time[i] = 1

        # column; [0:30], 6
        if 22 in maintenance_optional_list:
            column_repair_index = np.where(column_expected_state > 2)[0]
            for i in column_repair_index:
                column_state[i * 5: i * 5 + 5] = column_state[i * 5: i * 5 + 5] @ self.repair_matrix
                column_hidden[i] = 0
        elif 13 in maintenance_optional_list:
            column_repair_index = np.where(column_expected_state >= 3)[0]
            for i in column_repair_index:
                column_state[i * 5: i * 5 + 5] = column_state[i * 5: i * 5 + 5] @ self.repair_matrix
                column_hidden[i] = 0
        elif 13 not in maintenance_optional_list and 13 in maintenance_necessary_list:
            column_repair_index = np.where(column_expected_state >= 4)[0]
            for i in column_repair_index:
                column_state[i * 5: i * 5 + 5] = column_state[i * 5: i * 5 + 5] @ self.repair_matrix
                column_hidden[i] = 0

        # foundation; [0:30], 6
        if 23 in maintenance_optional_list:
            foundation_repair_index = np.where(foundation_expected_state > 2)[0]
            for i in foundation_repair_index:
                foundation_state[i * 5: i * 5 + 5] = foundation_state[i * 5: i * 5 + 5] @ self.repair_matrix
                foundation_hidden[i] = 0
        elif 14 in maintenance_optional_list:
            foundation_repair_index = np.where(foundation_expected_state >= 3)[0]
            for i in foundation_repair_index:
                foundation_state[i * 5: i * 5 + 5] = foundation_state[i * 5: i * 5 + 5] @ self.repair_matrix
                foundation_hidden[i] = 0
        elif 14 not in maintenance_optional_list and 14 in maintenance_necessary_list:
            foundation_repair_index = np.where(foundation_expected_state >= 4)[0]
            for i in foundation_repair_index:
                foundation_state[i * 5: i * 5 + 5] = foundation_state[i * 5: i * 5 + 5] @ self.repair_matrix
                foundation_hidden[i] = 0

        """
        preventive maintenace state evolution---------------------------------------------------------------------------
        """
        # action_index is a vector [0, 1, 1, 1, 1, 1, 1, 0],
        action_index = action_ecoding(actions)
        # action_index[1] means preventive maintenace in deck system
        # pavement_P_num is the number of preventive years be prevented
        # for instance, pavement_P_num = 1 , means 0 and 1 preventive years component be prevented

        # deck system preventive mainetance
        if action_index[1] == 1:
            # pavement; 4 years; [0:75], [15:30]-> 0; [30:45]->1; [45:60]->2; [60:75]->3;
            for i in range(deck_P_number[0] + 1):
                pavement_infor[-1, :] += pavement_infor[i, :]
                pavement_infor[i, :] = np.zeros((1, len(pavement_state)))
            # expansion; 3 years; [75:103], [82:89]-> 0; [89:96]->1; [96:103]->2;
            for i in range(deck_P_number[1] + 1):
                expansion_infor[-1, :] += expansion_infor[i, :]
                expansion_infor[i, :] = np.zeros((1, len(expansion_state)))
            # sidewalk; 5 years; [103:295]; [135:167]-> 0; [167:199]->1; [199:231]->2; [231:263]->3; [263:295]->4;
            for i in range(deck_P_number[2] + 1):
                sidewalk_infor[-1, :] += sidewalk_infor[i, :]
                sidewalk_infor[i, :] = np.zeros((1, len(sidewalk_state)))
            # barrier; 5 years; [295:595]; [345:395]-> 0; [395:445]->1; [445:495]->2; [495:545]->3; [545:595]->4;
            for i in range(deck_P_number[3] + 1):
                barrier_infor[-1, :] += barrier_infor[i, :]
                barrier_infor[i, :] = np.zeros((1, len(barrier_state)))
        if action_index[1] == 1 and 0 in maintenance_optional_list and 1 in maintenance_optional_list and 2 in maintenance_optional_list and 3 in maintenance_optional_list:
            reward_P_deck = -1
        else:
            reward_P_deck = 0

        # arch and transverse
        if action_index[2] == 1 and 1000 in superstructure_P_number[0] and 1000 in superstructure_P_number[1]:
            reward_P_arch_transverse = -1
        elif action_index[2] == 1:
            arch_P_number = np.where(arch_time == 0)[0]
            transverse_P_number = np.where(transverse_time == 0)[0]
            arch_time[arch_P_number] = 1
            transverse_time[transverse_P_number] = 1
            reward_P_arch_transverse = 0
        else:
            reward_P_arch_transverse = 0
        # girder and slab
        if action_index[3] == 1 and 1000 in superstructure_P_number[2] and 1000 in superstructure_P_number[3] and 1000 in superstructure_P_number[4]:
            reward_P_girder_slab = -1
        elif action_index[3] == 1:
            pri_girder_P_number = np.where(pri_girder_time == 0)[0]
            sec_girder_P_number = np.where(sec_girder_time == 0)[0]
            slab_P_number = np.where(slab_time == 0)[0]
            pri_girder_time[pri_girder_P_number] = 1
            sec_girder_time[sec_girder_P_number] = 1
            slab_time[slab_P_number] = 1
            reward_P_girder_slab = 0
        else:
            reward_P_girder_slab = 0
        # hanger, the replaced hanger time is reset to 0 in replace action part code
        if action_index[4] == 1 and 20 in maintenance_optional_list and 1000 in superstructure_P_number[7]:
            reward_P_hanger = -1
        elif action_index[4] == 1 and 20 in maintenance_optional_list:
            hanger_time[superstructure_P_number[7]] = 1
            reward_P_hanger = 0
        elif action_index[4] == 1 and 11 in maintenance_optional_list and 1000 in superstructure_P_number[6]:
            reward_P_hanger = -1
        elif action_index[4] == 1 and 11 in maintenance_optional_list:
            hanger_time[superstructure_P_number[6]] = 1
            reward_P_hanger = 0
        elif action_index[4] == 1 and 1000 in superstructure_P_number[5]:
            reward_P_hanger = -1
        elif action_index[4] == 1:
            hanger_time[superstructure_P_number[5]] = 1
            reward_P_hanger = 0
        else:
            reward_P_hanger = 0
        # column and foundation
        if action_index[5] == 1 and 1000 in substructure_P_number[0] and 1000 in substructure_P_number[1]:
            reward_P_substructure = -1
        elif action_index[5] == 1:
            column_P_number = np.where(column_time == 0)[0]
            foundation_P_number = np.where(foundation_time == 0)[0]
            column_time[column_P_number] = 1
            foundation_time[foundation_P_number] = 1
            reward_P_substructure = 0
        else:
            reward_P_substructure = 0

        """
        state transition in fixed period--------------------------------------------------------------------------------
        """
        # deck system
        # pavement
        pavement = np.hstack((np.zeros((round(len(pavement_time) / len(pavement_state) + 1), 1)), pavement_infor))
        pavement[-1, 0] = pavement_rep
        pavement_new_1 = np.zeros((np.size(pavement, 0), np.size(pavement, 1)))
        pavement_new_2 = np.zeros((np.size(pavement, 0), np.size(pavement, 1)))
        # preventive time = 0
        for i in range(np.size(pavement, 1) - 1):
            pavement_new_1[0, i+1] = Three_parameter_Weibull_transition(pavement[0, i], self.pavement_deter, i)
        # preventive time > 0
        for i in range(np.size(pavement, 1) - 1):
            for j in range(1, np.size(pavement, 0)):
                pavement_new_2[j-1, i+1] = Three_parameter_Weibull_transition(pavement[j, i], self.pavement_nor, i)
        pavement_new = pavement_new_1 + pavement_new_2
        pavement_new_state = pavement_new[0:np.size(pavement, 0) - 1, 1:np.size(pavement, 1)] / bridge_infor[0]
        pavement_state_new = np.sum(pavement_new_state, axis=0)
        pavement_time_new = pavement_new_state / (pavement_state_new + 1e-10)
        pavement_time_new = np.reshape(pavement_time_new, [-1])

        # expansion
        expansion = np.hstack((np.zeros((round(len(expansion_time) / len(expansion_state) + 1), 1)), expansion_infor))
        expansion[-1, 0] = expansion_rep
        expansion_new_1 = np.zeros((np.size(expansion, 0), np.size(expansion, 1)))
        expansion_new_2 = np.zeros((np.size(expansion, 0), np.size(expansion, 1)))
        # preventive time = 0
        for i in range(np.size(expansion, 1) - 1):
            expansion_new_1[0, i+1] = Three_parameter_Weibull_transition(expansion[0, i], self.expansion_deter, i)
        # preventive time > 0
        for i in range(np.size(expansion, 1) - 1):
            for j in range(1, np.size(expansion, 0)):
                expansion_new_2[j-1, i+1] = Three_parameter_Weibull_transition(expansion[j, i], self.expansion_nor, i)
        expansion_new = expansion_new_1 + expansion_new_2
        expansion_new_state = expansion_new[0:np.size(expansion, 0) - 1, 1:np.size(expansion, 1)] / bridge_infor[1]
        expansion_state_new = np.sum(expansion_new_state, axis=0)
        expansion_time_new = expansion_new_state / (expansion_state_new + 1e-10)
        expansion_time_new = np.reshape(expansion_time_new, [-1])

        # sidewalk
        sidewalk = np.hstack((np.zeros((round(len(sidewalk_time) / len(sidewalk_state) + 1), 1)), sidewalk_infor))
        sidewalk[-1, 0] = sidewalk_rep
        sidewalk_new_1 = np.zeros((np.size(sidewalk, 0), np.size(sidewalk, 1)))
        sidewalk_new_2 = np.zeros((np.size(sidewalk, 0), np.size(sidewalk, 1)))
        # preventive time = 0
        for i in range(np.size(sidewalk, 1) - 1):
            sidewalk_new_1[0, i+1] = Three_parameter_Weibull_transition(sidewalk[0, i], self.sidewalk_deter, i)
        # preventive time > 0
        for i in range(np.size(sidewalk, 1) - 1):
            for j in range(1, np.size(sidewalk, 0)):
                sidewalk_new_2[j-1, i+1] = Three_parameter_Weibull_transition(sidewalk[j, i], self.sidewalk_nor, i)
        sidewalk_new = sidewalk_new_1 + sidewalk_new_2
        sidewalk_new_state = sidewalk_new[0:np.size(sidewalk, 0) - 1, 1:np.size(sidewalk, 1)] / bridge_infor[2]
        sidewalk_state_new = np.sum(sidewalk_new_state, axis=0)
        sidewalk_time_new = sidewalk_new_state / (sidewalk_state_new + 1e-10)
        sidewalk_time_new = np.reshape(sidewalk_time_new, [-1])

        # barrier
        barrier = np.hstack((np.zeros((round(len(barrier_time) / len(barrier_state) + 1), 1)), barrier_infor))
        barrier[-1, 0] = barrier_rep
        barrier_new_1 = np.zeros((np.size(barrier, 0), np.size(barrier, 1)))
        barrier_new_2 = np.zeros((np.size(barrier, 0), np.size(barrier, 1)))
        # preventive time = 0
        for i in range(np.size(barrier, 1) - 1):
            barrier_new_1[0, i + 1] = Three_parameter_Weibull_transition(barrier[0, i], self.barrier_deter, i)
        # preventive time > 0
        for i in range(np.size(barrier, 1) - 1):
            for j in range(1, np.size(barrier, 0)):
                barrier_new_2[j-1, i+1] = Three_parameter_Weibull_transition(barrier[j, i], self.barrier_nor, i)
        barrier_new = barrier_new_1 + barrier_new_2
        barrier_new_state = barrier_new[0:np.size(barrier, 0) - 1, 1:np.size(barrier, 1)] / bridge_infor[3]
        barrier_state_new = np.sum(barrier_new_state, axis=0)
        barrier_time_new = barrier_new_state / (barrier_state_new + 1e-10)
        barrier_time_new = np.reshape(barrier_time_new, [-1])

        # drainage
        drainage = np.concatenate(([drainage_rep], np.round(bridge_infor[4] * drainage_state)))
        drainage_new = np.zeros((len(drainage)))
        for i in range(len(drainage) - 1):
            drainage_new[i + 1] = Three_parameter_Weibull_transition(drainage[i], self.drainage_nor, i)
        drainage_state_new = drainage_new[1:len(drainage)] / bridge_infor[4]

        # lighting
        lighting = np.concatenate(([lighting_rep], np.round(bridge_infor[5] * lighting_state)))
        lighting_new = np.zeros((len(lighting)))
        for i in range(len(lighting) - 1):
            lighting_new[i + 1] = Three_parameter_Weibull_transition(lighting[i], self.lighting_nor, i)
        lighting_state_new = lighting_new[1:len(lighting)] / bridge_infor[5]

        # arch
        arch_new_state, arch_new_time, arch_new_hidden = state_evolution(arch_state, arch_time, arch_hidden,
                                                                         self.arch_matrix, self.arch_matrix_D, 10)
        # transverse
        transverse_new_state, transverse_new_time, transverse_new_hidden = state_evolution(transverse_state,
                                                                                           transverse_time,
                                                                                           transverse_hidden,
                                                                                           self.transverse_matrix,
                                                                                           self.transverse_matrix_D, 10)
        # primary girder
        pri_girder_new_state, pri_girder_new_time, pri_girder_new_hidden = state_evolution(pri_girder_state, pri_girder_time, pri_girder_hidden,
                                                                               self.girder_matrix, self.girder_matrix_D,
                                                                               15)
        # secondary girder
        sec_girder_new_state, sec_girder_new_time, sec_girder_new_hidden = state_evolution(sec_girder_state, sec_girder_time, sec_girder_hidden,
                                                                               self.girder_matrix, self.girder_matrix_D,
                                                                               15)
        # slab
        slab_new_state, slab_new_time, slab_new_hidden = state_evolution(slab_state, slab_time, slab_hidden,
                                                                               self.slab_matrix, self.slab_matrix_D,
                                                                               15)
        # hanger
        hanger_new_state, hanger_new_time, hanger_new_hidden = state_evolution(hanger_state, hanger_time, hanger_hidden,
                                                                               self.hanger_matrix, self.hanger_matrix_D,
                                                                               5)
        # support
        support_new_state, support_new_time, support_new_hidden = state_evolution(support_state, support_time,
                                                                                  support_hidden,
                                                                                  self.support_matrix,
                                                                                  self.support_matrix_D,
                                                                                  5)
        # column
        column_new_state, column_new_time, column_new_hidden = state_evolution(column_state, column_time, column_hidden,
                                                                               self.column_matrix, self.column_matrix_D,
                                                                               15)
        # foundation
        foundation_new_state, foundation_new_time, foundation_new_hidden = state_evolution(foundation_state,
                                                                                           foundation_time,
                                                                                           foundation_hidden,
                                                                                           self.foundation_matrix,
                                                                                           self.foundation_matrix_D,
                                                                                           15)

        new_hidden_state = np.concatenate((arch_new_hidden, transverse_new_hidden, pri_girder_new_hidden,
                                           sec_girder_new_hidden,slab_new_hidden,hanger_new_hidden,
                                           support_new_hidden, column_new_hidden, foundation_new_hidden))


        # observation part
        observation_value = np.ones((len(new_hidden_state)), dtype = int) * 10
        super_sub_structure = np.concatenate((arch_new_state, transverse_new_state, pri_girder_new_state,
                                              sec_girder_new_state, slab_new_state, hanger_new_state,
                                              support_new_state, column_new_state, foundation_new_state))
        del (arch_new_state, transverse_new_state, pri_girder_new_state,
             sec_girder_new_state, slab_new_state, hanger_new_state,
             support_new_state, column_new_state, foundation_new_state)
        if action_index[0] == 0:
            observation_matrix = observation(self.accuracy_visual, self.state_number, Matrix_type=False)
        elif action_index[0] == 1:
            observation_matrix = observation(self.accuracy_NDT, self.state_number, Matrix_type=True)

        for i in range(len(new_hidden_state)):
            obser_mark = 0.
            random_number = np.random.uniform(0, 1)
            for j in range(self.state_number):
                obser_mark = obser_mark + observation_matrix[new_hidden_state[i], j]
                if random_number <= obser_mark:
                    observation_value[i] = j
                    break

            super_sub_structure[i * 5: i * 5 + 5] = super_sub_structure[i * 5: i * 5 + 5] * observation_matrix[:,
                                                                                            observation_value[i]]
            super_sub_structure[i * 5: i * 5 + 5] = super_sub_structure[i * 5: i * 5 + 5] / np.sum(
                super_sub_structure[i * 5: i * 5 + 5])

        arch_new_state = super_sub_structure[0:15]
        transverse_new_state = super_sub_structure[15:115]
        pri_girder_new_state = super_sub_structure[115:135]
        sec_girder_new_state = super_sub_structure[135:345]
        slab_new_state = super_sub_structure[345:565]
        hanger_new_state = super_sub_structure[565:880]
        support_new_state = super_sub_structure[880:920]
        column_new_state = super_sub_structure[920:950]
        foundation_new_state = super_sub_structure[950:980]

        time += 1
        new_time = tf.one_hot(time, 100).numpy()
        # return part
        new_state = np.concatenate((pavement_state_new, pavement_time_new, expansion_state_new, expansion_time_new,
                                    sidewalk_state_new, sidewalk_time_new, barrier_state_new, barrier_time_new,
                                    drainage_state_new, lighting_state_new, arch_new_state,
                                    arch_new_time, transverse_new_state, transverse_new_time, pri_girder_new_state,
                                    pri_girder_new_time, sec_girder_new_state, sec_girder_new_time,
                                    slab_new_state, slab_new_time, hanger_new_state, hanger_new_time,
                                    support_new_state, support_new_time, column_new_state, column_new_time,
                                    foundation_new_state, foundation_new_time, new_time))
        new_state = new_state.reshape(1, len(new_state))

        """
        reward part-----------------------------------------------------------------------------------------------------
        """

        # cost item
        new_budget = 3e5 - (maintenance_fee + preventive_cost + inspection_cost)
        # ancillary facilities conditions
        # bridge_ancillary_condition = - 0.02 * (DMCI[4] < 66) - 0.01 * (DMCI[5] < 51)
        # entire bridge risk
        bridge_risk_1 = np.sum(arch_state[3::5]) * 0.1288 + np.sum(
            transverse_state[3::5]) * 0.0228 + np.sum(pri_girder_state[3::5]) * 0.0644 + np.sum(
            sec_girder_state[3::5]) * 0.0644 + np.sum(slab_state[3::5]) * 0.0368 + np.sum(
            hanger_state[3::5]) * 0.06 + np.sum(support_state[3::5]) * 0.0228 + np.sum(
            column_state[3::5]) * 0.2068 + np.sum(foundation_state[3::5]) * 0.1932
        bridge_risk_2 = - 0.1288 * np.sum(arch_state[4::5]) - np.sum(transverse_state[4::5]) * 0.0228 - np.sum(
            pri_girder_state[4::5]) * 0.0644 - np.sum(sec_girder_state[4::5]) * 0.0644 - np.sum(
            slab_state[4::5]) * 0.0368 - np.sum(hanger_state[4::5]) * 0.06 - np.sum(
            support_state[4::5]) * 0.0228 - np.sum(column_state[4::5]) * 0.2068 - np.sum(
            foundation_state[4::5]) * 0.1932
        bridge_risk = -1.5 * bridge_risk_1 + 15 * bridge_risk_2
        # bridge condition
        bridge_condition = 0
        if Bridge_state < 60:
            bridge_condition += -1
        # check whether the maintenance action is effective
        action_reward = 0.5 * reward + 0.1 * (
                    reward_P_deck + reward_P_arch_transverse + reward_P_girder_slab + reward_P_hanger + reward_P_substructure)
        # three parts
        final_reward = new_budget / 3e5 + bridge_risk + action_reward + bridge_condition

        cost = np.array([final_reward, maintenance_fee, preventive_cost, inspection_cost])
        return new_state, cost, new_hidden_state, maintenance_necessary_list, maintenance_optional_list

class Actor_build(keras.Model):
    def __init__(self, state_size, action_size, batchnorm=True, hidden=[2048, 2048, 2048]):
        super(Actor_build, self).__init__()
        self.fc1 = layers.Dense(hidden[0], input_shape=[None, state_size], kernel_regularizer=regularizers.l2(0.001))
        self.fc2 = layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(0.001))
        self.fc3 = layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(0.001))
        self.fc4 = layers.Dense(action_size, kernel_regularizer=regularizers.l2(0.001))
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.batchnorm = batchnorm

    def call(self, x):
        if self.batchnorm:
            x = tf.nn.relu(self.bn1(self.fc1(x)))
            # 2048 -> 2048
            res1 = x
            x = tf.nn.relu(self.bn2(self.fc2(x)))
            x = x + res1
            # 2048 -> 2048
            res2 = x
            x = tf.nn.relu(self.bn3(self.fc3(x)))
            x = x + res2
            x = self.fc4(x)
        else:
            x = tf.nn.relu(self.fc1(x))
            res1 = x
            x = tf.nn.relu(self.fc2(x))
            x = x + res1
            res2 = x
            x = tf.nn.relu(self.fc3(x))
            x = x + res2
            x = self.fc4(x)
        return x

class Critic_build(keras.Model):
    def __init__(self, state_size, batchnorm=True, hidden=[2048, 2048, 2048]):
        super(Critic_build, self).__init__()
        self.fc1 = layers.Dense(hidden[0], input_shape=[None, state_size], kernel_regularizer=regularizers.l2(0.001))
        self.fc2 = layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(0.001))
        self.fc3 = layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(0.001))
        self.fc4 = layers.Dense(1, kernel_regularizer=regularizers.l2(0.001))
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.batchnorm = batchnorm

    def call(self, x):
        if self.batchnorm:
            x = tf.nn.relu(self.bn1(self.fc1(x)))
            # 2048 -> 2048
            res1 = x
            x = tf.nn.relu(self.bn2(self.fc2(x)))
            x = x + res1
            # 2048 -> 2048
            res2 = x
            x = tf.nn.relu(self.bn3(self.fc3(x)))
            x = x + res2
            x = self.fc4(x)
        else:
            x = tf.nn.relu(self.fc1(x))
            res1 = x
            x = tf.nn.relu(self.fc2(x))
            x = x + res1
            res2 = x
            x = tf.nn.relu(self.fc3(x))
            x = x + res2
            x = self.fc4(x)
        return x

def get_action(Actor_network, state, greedy=False):
    logit = Actor_network(state)
    prob = tf.nn.softmax(logit).numpy()
    if greedy:
        return np.argmax(prob.ravel())
    action = np.random.choice(logit.shape[1], p=prob.ravel())
    return action

class Agent():
    def __init__(self,
                 Actor,
                 Critic,
                 Environment_set,
                 n_actions=2432,
                 input_shape=1941,
                 ):
        # state vector and action number
        self.n_actions = n_actions
        self.input_shape = input_shape

        # define the Actor-Critic networks
        self.Actor = Actor
        self.Critic = Critic
        self.Environment = Environment_set

        self.Critic_optimizer = tf.optimizers.Adam(5e-5)
        self.Actor_optimizer = tf.optimizers.Adam(5e-6)

    def save(self, folder_name, **kwargs):
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save Main_DQN and Target_DQN
        self.Actor.save_weights(folder_name + '/Actor.ckpt')
        self.Critic.save_weights(folder_name + '/Critic.ckpt')

    def load(self, folder_name, **kwargs):
        self.Actor.load_weights(folder_name + '/Actor.ckpt')
        self.Critic.load_weights(folder_name + '/Critic.ckpt')

    def Critic_learn(self, state_old, reward, state_new, gamma):
        reward = tf.reshape(tf.constant([reward], dtype=float), [-1, 1])
        with tf.GradientTape() as tape:
            v_old = self.Critic(state_old)
            v_new = self.Critic(state_new)
            TD_error = reward + gamma * v_new - v_old
            loss_critic = tf.square(TD_error)
        model_gradients = tape.gradient(loss_critic, self.Critic.trainable_variables)
        clip_Critic_value = [tf.clip_by_norm(grad, 80.0) for grad in model_gradients]
        self.Critic_optimizer.apply_gradients(zip(clip_Critic_value, self.Critic.trainable_variables))
        return TD_error

    def Actor_learn(self, state, action, TD_error):
        one_action = tf.cast(tf.one_hot(action, 2432), dtype=tf.float32)
        with tf.GradientTape() as tape:
            logit = self.Actor(state)
            cross_entropy = - tf.multiply(tf.math.log(tf.nn.softmax(logit) + 1e-20), one_action)
            loss_actor = tf.reduce_sum(tf.multiply(cross_entropy, TD_error))
            entropy = tf.reduce_sum(-tf.nn.softmax(logit) * tf.math.log(tf.nn.softmax(logit) + 1e-20))
            loss = loss_actor - 0.0018 * entropy
        grads = tape.gradient(loss, self.Actor.trainable_variables)
        clip_gradient_value = [tf.clip_by_norm(grad, 800.0) for grad in grads]
        self.Actor_optimizer.apply_gradients(zip(clip_gradient_value, self.Actor.trainable_variables))

    def estimation(self):
        # estimate whether performance become better
        # initial the parameters
        max_over_step = 100
        Reward = []

        Bridge_information = np.array(
            [132 * 8 * 2, 10 * 4, 132 * 2 * 3, 132 * 4, (132 / 12 + 1) * 2, 8 * 4, 145.16 * 7, 10.7 * 3.17, 132, 10, 6 * 10,
             4.79, 9.04, 12.79, 16.05, 18.84, 21.18, 23.07, 24.53, 25.57, 26.19, 26.4,
             26.19, 25.57, 24.53, 23.07, 21.18, 18.84, 16.05, 12.79, 9.04, 4.79,
             6, 20 * 3 * 4, 8.6 * 3 * 2 + 13.6 * 3 * 2 + 8.6 * 13.6])

        for li in range(500):
            # define initial state
            # pavement: 15 service years, and 4 preventive years
            pavement_initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            pavement_intial_time = np.zeros((4 * 15,))
            pavement_intial_time[3 * 15] = 1
            # expansion: 7 service years, and 3 preventive years
            expansion_initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            expansion_intial_time = np.zeros((3 * 10,))
            expansion_intial_time[2 * 10] = 1
            # sidewalk: 32 service years, and 5 preventive years
            sidewalk_initial_state = np.array(
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            sidewalk_intial_time = np.zeros((5 * 32,))
            sidewalk_intial_time[4 * 32] = 1
            # barrier: 50 service years, and 5 preventive years
            barrier_initial_state = np.array(
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            barrier_intial_time = np.zeros((5 * 50,))
            barrier_intial_time[4 * 50] = 1
            # drainage: 50 service years
            drainage_initial_state = np.array(
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            # lighting: 8 years
            lighting_initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
            # arch, 3 compoents, 12 preventive years
            arch_initial_state = np.zeros((15))
            for i in range(3):
                arch_initial_state[i * 5] = 1
            arch_initial_time = np.array([1, 1, 1])
            # transverse, 20 components, 12 preventive years
            transverse_initial_state = np.zeros((20 * 5))
            for i in range(20):
                transverse_initial_state[i * 5] = 1
            transverse_initial_time = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            # primary girder, 4 components, 15 preventive years
            pri_girder_initial_state = np.zeros((4 * 5))
            for i in range(4):
                pri_girder_initial_state[i * 5] = 1
            pri_girder_initial_time = np.array([1, 1, 1, 1])
            # secondary girder, 42 components, 15 preventive years
            sec_girder_initial_state = np.zeros((42 * 5))
            for i in range(42):
                sec_girder_initial_state[i * 5] = 1
            sec_girder_initial_time = np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1])
            # slab, 44 components, 15 preventive years
            slab_initial_state = np.zeros((44 * 5))
            for i in range(44):
                slab_initial_state[i * 5] = 1
            slab_initial_time = np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            # hanger, 63 components, 5 preventive years
            hanger_initial_state = np.zeros((63 * 5))
            for i in range(63):
                hanger_initial_state[i * 5] = 1
            hanger_initial_time = np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            # support, 6 component, 5 preventive years
            support_initial_state = np.zeros((8 * 5))
            for i in range(8):
                support_initial_state[i * 5] = 1
            support_initial_time = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            # column, 6 component, 15 preventive years
            column_initial_state = np.zeros((6 * 5))
            for i in range(6):
                column_initial_state[i * 5] = 1
            column_initial_time = np.array([1, 1, 1, 1, 1, 1])
            # foundation, 6 component, 15 preventive years
            foundation_initial_state = np.zeros((6 * 5))
            for i in range(6):
                foundation_initial_state[i * 5] = 1
            foundation_initial_time = np.array([1, 1, 1, 1, 1, 1])

            initial_time = tf.one_hot(0, 100).numpy()

            initial_states = np.concatenate(
                (pavement_initial_state, pavement_intial_time, expansion_initial_state, expansion_intial_time,
                 sidewalk_initial_state, sidewalk_intial_time, barrier_initial_state, barrier_intial_time,
                 drainage_initial_state, lighting_initial_state,
                 arch_initial_state, arch_initial_time, transverse_initial_state, transverse_initial_time,
                 pri_girder_initial_state, pri_girder_initial_time, sec_girder_initial_state, sec_girder_initial_time,
                 slab_initial_state, slab_initial_time, hanger_initial_state, hanger_initial_time,
                 support_initial_state, support_initial_time, column_initial_state, column_initial_time,
                 foundation_initial_state, foundation_initial_time, initial_time))
            initial_states = initial_states.reshape(1, len(initial_states))
            initial_hidden_state = np.zeros((3 + 20 + 4 + 42 + 44 + 63 + 8 + 6 + 6), dtype=int)

            t = 0
            states = initial_states.copy()
            hidden_state = initial_hidden_state.copy()
            budget = 0
            reward_sum = 0
            while t < max_over_step:
                action = get_action(self.Actor, states, greedy=True)

                New_belief_state, reward, hidden_state, maintenance_necessary_list, maintenance_optional_list = self.Environment.step(
                    Bridge_information, states, action, hidden_state)

                states = New_belief_state.copy()
                reward_sum += reward[0]

                t += 1

            Reward.append(reward_sum)
            print("\r", end="")
            print("进度: {}%: ".format(li / 10), "▓" * (li // 10), end="")

        Reward_sum = sum(Reward) / 500
        print(Reward_sum)
        return Reward_sum

    def get_trajectory(self, initial_state, hidden_state, exploration_step):
        """
        :param Actor_network:
        :param Critic_network:
        :param initial_state:
        :param exploration_step:
        :return: trajectory[states, actions, rewards], terminal(boolean)
        """
        Bridge_information = np.array(
            [132 * 8 * 2, 10 * 4, 132 * 2 * 3, 132 * 4, (132 / 12 + 1) * 2, 8 * 4, 145.16 * 7, 10.7 * 3.17, 132, 10, 6 * 10,
             4.79, 9.04, 12.79, 16.05, 18.84, 21.18, 23.07, 24.53, 25.57, 26.19, 26.4,
             26.19, 25.57, 24.53, 23.07, 21.18, 18.84, 16.05, 12.79, 9.04, 4.79,
             6, 20 * 3 * 4, 8.6 * 3 * 2 + 13.6 * 3 * 2 + 8.6 * 13.6])
        # memory
        memory_states = np.zeros((exploration_step, 1941))
        memory_actions = np.zeros((exploration_step))
        memory_reward = np.zeros((exploration_step, 4))
        memory_new_states = np.zeros((exploration_step, 1941))
        memory_hidden_state = np.zeros((exploration_step, 196))
        maintenance_list = np.empty((exploration_step, 2), dtype=object)

        # get MINI_STEP trajectory
        for i in range(exploration_step):
            memory_states[i, :] = initial_state.copy()
            action = get_action(self.Actor, initial_state, greedy = False)
            # action = action_necessary.action_necessary_select(initial_state, np.argmax(initial_state[0,1841:1941]))
            New_belief_state, reward, new_hidden_state, maintenance_necessary_list, maintenance_optional_list = self.Environment.step(
                Bridge_information, initial_state, action, hidden_state)

            memory_actions[i] = action
            memory_reward[i, :] = reward
            memory_hidden_state[i, :] = hidden_state
            memory_new_states[i, :] = New_belief_state.copy()
            maintenance_list[i, 0] = maintenance_necessary_list
            maintenance_list[i, 1] = maintenance_optional_list

            initial_state = New_belief_state.copy()
            hidden_state = new_hidden_state
            reward_sum = np.sum(memory_reward[:, 0])

        return (memory_states, memory_actions, memory_reward, memory_new_states, New_belief_state, reward_sum,
                new_hidden_state, memory_hidden_state, maintenance_list)

def main():
    # define initial state
    # pavement: 15 service years, and 4 preventive years
    pavement_initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    pavement_intial_time = np.zeros((4 * 15,))
    pavement_intial_time[3 * 15] = 1
    # expansion: 7 service years, and 3 preventive years
    expansion_initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expansion_intial_time = np.zeros((3 * 10,))
    expansion_intial_time[2 * 10] = 1
    # sidewalk: 32 service years, and 5 preventive years
    sidewalk_initial_state = np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    sidewalk_intial_time = np.zeros((5 * 32,))
    sidewalk_intial_time[4 * 32] = 1
    # barrier: 50 service years, and 5 preventive years
    barrier_initial_state = np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    barrier_intial_time = np.zeros((5 * 50,))
    barrier_intial_time[4 * 50] = 1
    # drainage: 50 service years
    drainage_initial_state = np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # lighting: 8 years
    lighting_initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    # arch, 3 compoents, 12 preventive years
    arch_initial_state = np.zeros((15))
    for i in range(3):
        arch_initial_state[i * 5] = 1
    arch_initial_time = np.array([1, 1, 1])
    # transverse, 20 components, 12 preventive years
    transverse_initial_state = np.zeros((20*5))
    for i in range(20):
        transverse_initial_state[i * 5] = 1
    transverse_initial_time = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # primary girder, 4 components, 15 preventive years
    pri_girder_initial_state = np.zeros((4 * 5))
    for i in range(4):
        pri_girder_initial_state[i * 5] = 1
    pri_girder_initial_time = np.array([1, 1, 1, 1])
    # secondary girder, 42 components, 15 preventive years
    sec_girder_initial_state = np.zeros((42 * 5))
    for i in range(42):
        sec_girder_initial_state[i * 5] = 1
    sec_girder_initial_time = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1])
    # slab, 44 components, 15 preventive years
    slab_initial_state = np.zeros((44 * 5))
    for i in range(44):
        slab_initial_state[i * 5] = 1
    slab_initial_time = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1])
    # hanger, 63 components, 5 preventive years
    hanger_initial_state = np.zeros((63 * 5))
    for i in range(63):
        hanger_initial_state[i * 5] = 1
    hanger_initial_time = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # support, 6 component, 5 preventive years
    support_initial_state = np.zeros((8 * 5))
    for i in range(8):
        support_initial_state[i * 5] = 1
    support_initial_time = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    # column, 6 component, 15 preventive years
    column_initial_state = np.zeros((6 * 5))
    for i in range(6):
        column_initial_state[i * 5] = 1
    column_initial_time = np.array([1, 1, 1, 1, 1, 1])
    # foundation, 6 component, 15 preventive years
    foundation_initial_state = np.zeros((6 * 5))
    for i in range(6):
        foundation_initial_state[i * 5] = 1
    foundation_initial_time = np.array([1, 1, 1, 1, 1, 1])

    initial_time = tf.one_hot(0, 100).numpy()

    initial_states = np.concatenate(
        (pavement_initial_state, pavement_intial_time, expansion_initial_state, expansion_intial_time,
         sidewalk_initial_state, sidewalk_intial_time, barrier_initial_state, barrier_intial_time,
         drainage_initial_state, lighting_initial_state,
         arch_initial_state, arch_initial_time, transverse_initial_state, transverse_initial_time,
         pri_girder_initial_state, pri_girder_initial_time, sec_girder_initial_state, sec_girder_initial_time,
         slab_initial_state, slab_initial_time, hanger_initial_state, hanger_initial_time,
         support_initial_state, support_initial_time, column_initial_state, column_initial_time,
         foundation_initial_state, foundation_initial_time, initial_time))
    initial_states = initial_states.reshape(1, len(initial_states))
    initial_hidden_state = np.zeros((3 + 20 + 4 + 42 + 44 + 63 + 8 + 6 + 6), dtype = int)
    # initial netrual network
    Actor_network = Actor_build(1941, 2432)
    Critic_network = Critic_build(1941)
    Environment = environment()

    num_episode = 120001
    max_over_step = 100
    gamma = 0.99
    agent = Agent(Actor_network, Critic_network, Environment)
    t_plot = []
    reward_plot = []
    action_episode = np.zeros((100, 7))
    hidden_state_episode = np.zeros((100, 196))

    exploration_step = 20

    continue_training = True
    if continue_training:
        agent.load('ActorCritic')
        estimate_value = np.loadtxt('Performance')
        print(estimate_value)
    else:
        estimate_value = -1000

    for i in range(0, num_episode):
        t = 0
        states = initial_states.copy()
        hidden_state = initial_hidden_state.copy()
        reward_sum = 0
        # cost_M = np.zeros((100,))
        # cost_P = np.zeros((100,))
        # cost_I = np.zeros((100,))

        pavement_state = np.zeros((100,))
        expansion_state = np.zeros((100,))
        sidewalk_state = np.zeros((100,))
        barrier_state = np.zeros((100,))
        drainage_state = np.zeros((100,))
        lighting_state = np.zeros((100,))
        arch_state = np.zeros((100,))
        transverse_state = np.zeros((100,))
        pri_girder_state = np.zeros((100,))
        sec_girder_state = np.zeros((100,))
        slab_state = np.zeros((100,))
        hanger_state = np.zeros((100,))
        support_state = np.zeros((100,))
        column_state = np.zeros((100,))
        foundation_state = np.zeros((100,))

        # 1 means necessary repair; 2 means optinal repair
        maintenance_action = np.zeros((100, 15))

        if i % 10000 == 0 and i > 10:
            New_reward_sum = agent.estimation()

            if New_reward_sum - estimate_value > 0:
                agent.save('ActorCritic')
                li = np.array(New_reward_sum, dtype=np.float64).reshape(1, 1)
                np.savetxt('Performance', li)
                print('save successful')
                estimate_value = New_reward_sum
            elif i == 140000 and New_reward_sum > 0:
                agent.save('ActorCritic')
                estimate_value = np.loadtxt('Performance')

        while t < max_over_step:

            memory_states, memory_actions, memory_reward, memory_new_states, New_belief_state, reward, hidden_state, memory_hidden_state, maintenance_list = agent.get_trajectory(
                states, hidden_state, exploration_step)

            TD_error = agent.Critic_learn(memory_states, memory_reward[:, 0], memory_new_states, gamma)
            agent.Actor_learn(memory_states, memory_actions, TD_error)

            action_episode[t:t + exploration_step] = np.transpose(action_ecoding(memory_actions))
            hidden_state_episode[t:t + exploration_step, :] = memory_hidden_state
            for j in range(exploration_step):
                for lai in maintenance_list[j, 0]:
                    maintenance_action[t + j, lai] = 1
                for li in maintenance_list[j, 1]:
                    if li < 15:
                        maintenance_action[t + j, li] = 2
                    else:
                        maintenance_action[t + j, li - 9] = 3

            pavement_state[t:t + exploration_step] = 1 - np.sum(memory_states[:, 0:15], axis = 1)
            expansion_state[t:t + exploration_step] = 1 - np.sum(memory_states[:, 75:85], axis = 1)
            sidewalk_state[t:t + exploration_step] = 1 - np.sum(memory_states[:, 115:147], axis = 1)
            barrier_state[t:t + exploration_step] = 1 - np.sum(memory_states[:, 307:357], axis = 1)
            drainage_state[t:t + exploration_step] = 1 - np.sum(memory_states[:, 607:657], axis = 1)
            lighting_state[t:t + exploration_step] = 1 - np.sum(memory_states[:, 657:665], axis = 1)

            arch_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 665:680], (-1, 3, 5)) @ np.array([1, 2, 3, 4, 5]), axis = 1)
            transverse_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 683:783], (-1, 20, 5)) @ np.array([1, 2, 3, 4, 5]), axis = 1)
            pri_girder_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 803:823], (-1, 4, 5)) @ np.array([1, 2, 3, 4, 5]), axis = 1)
            sec_girder_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 827:1037], (-1, 42, 5)) @ np.array([1, 2, 3, 4, 5]), axis=1)
            slab_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 1079:1299], (-1, 44, 5)) @ np.array([1, 2, 3, 4, 5]), axis = 1)
            hanger_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 1343:1658], (-1, 63, 5)) @ np.array([1, 2, 3, 4, 5]), axis = 1)
            support_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 1721:1761], (-1, 8, 5)) @ np.array([1, 2, 3, 4, 5]), axis = 1)
            column_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 1769:1799], (-1, 6, 5)) @ np.array([1, 2, 3, 4, 5]), axis = 1)
            foundation_state[t:t + exploration_step] = np.mean(np.reshape(memory_states[:, 1805:1835], (-1, 6, 5)) @ np.array([1, 2, 3, 4, 5]), axis = 1)

            states = New_belief_state.copy()
            reward_sum = reward_sum + reward

            t += exploration_step
            if t == max_over_step:
                print("epoch num:", i, " time stemp: ", t,  "   Reward: ", reward_sum)
                print("-----------------------------------------------------------------------------------------")
                plt.ion()
                fig1 = plt.figure(1, figsize=(3.5, 2))
                fig1.canvas.manager.window.move(50, 50)
                plt.clf()
                t_plot.append(i)
                reward_plot.append(reward_sum)
                plt.plot(t_plot, reward_plot, label='Sum reward', color='blueviolet', alpha=1, linewidth=0.4)
                plt.legend(loc='best', fontsize=8)
                plt.xlabel("life-cycle(year)", fontsize=8)
                plt.ylabel("CNY", fontsize=8)
                plt.ylim((-35, 15))
                plt.draw()
                # plt.pause(0.001)
                #
                # fig2 = plt.figure(2, figsize=(3.5, 2))
                # fig2.canvas.manager.window.move(400, 50)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 0]/5, c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), pavement_state, label='pavement state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("damage percentage", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig3 = plt.figure(3, figsize=(3.5, 2))
                # fig3.canvas.manager.window.move(750, 50)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 1]/5, c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), expansion_state, label='expansion state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig4 = plt.figure(4, figsize=(3.5, 2))
                # fig4.canvas.manager.window.move(1100, 50)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 2]/5, c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), sidewalk_state, label='sidewalk state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig5 = plt.figure(5, figsize=(3.5, 2))
                # fig5.canvas.manager.window.move(1450, 50)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 3]/5, c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), barrier_state, label='barrier state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig6 = plt.figure(6, figsize=(3.5, 2))
                # fig6.canvas.manager.window.move(50, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 4]/5, c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), drainage_state, label='drainage state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig7 = plt.figure(7, figsize=(3.5, 2))
                # fig7.canvas.manager.window.move(400, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 5]/5, c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), lighting_state, label='lighting state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig8 = plt.figure(8, figsize=(3.5, 2))
                # fig8.canvas.manager.window.move(750, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 6], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), arch_state, label='arch state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig9 = plt.figure(9, figsize=(3.5, 2))
                # fig9.canvas.manager.window.move(1100, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 7], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), transverse_state, label='transverse state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig10 = plt.figure(10, figsize=(3.5, 2))
                # fig10.canvas.manager.window.move(1450, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 8], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), pri_girder_state, label='girder state',
                #          color='deepskyblue')
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 9], c="red", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), sec_girder_state, label='girder state',
                #          color='pink')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig11 = plt.figure(11, figsize=(3.5, 2))
                # fig11.canvas.manager.window.move(50, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 10], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), slab_state, label='slab state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig12 = plt.figure(12, figsize=(3.5, 2))
                # fig12.canvas.manager.window.move(400, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 11], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), hanger_state, label='hanger state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig13 = plt.figure(13, figsize=(3.5, 2))
                # fig13.canvas.manager.window.move(750, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 12], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), support_state, label='support state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig14 = plt.figure(14, figsize=(3.5, 2))
                # fig14.canvas.manager.window.move(1100, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 13], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), column_state, label='column state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig15 = plt.figure(15, figsize=(3.5, 2))
                # fig15.canvas.manager.window.move(1450, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), maintenance_action[:, 14], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), foundation_state, label='foundation state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                plt.pause(0.01)


if __name__ == '__main__':
    main()
