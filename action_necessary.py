import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import math
import sympy as sympy
from sympy.solvers import solve

def repair_matrix(Utility, state_number):
    """
    :param Utility:
    :param state_number:
    :return:
    """
    # action utility is a value between 0-1, 0 means the no effects = Do nothing, 1 means replacement
    # [1  0  0  0  0]           [1  0  0  0  0]                   [1     0     0     0    0]
    # [0  1  0  0  0]           [1  0  0  0  0]                   [b-a   b     0     0    0]
    # [0  0  1  0  0]  ------>  [1  0  0  0  0]  ----utility----> [b-2a  b-a   b     0    0]
    # [0  0  0  1  0]           [1  0  0  0  0]                   [b-3a  b-2a  b-a   b    0]
    # [0  0  0  0  1]           [1  0  0  0  0]                   [b-4a  b-3a  b-2a  b-a  b]
    x, y = sympy.symbols('x y')
    Repair_matrix = np.zeros((state_number, state_number))
    Repair_matrix[0, 0] = 1
    for i in range(state_number - 1):
        coefficient = np.zeros((i + 2))
        for j in range(i + 1):
            Factor_1 = np.ones(i + 2 - j)
            if coefficient[0] < 0:
                Factor_2 = np.arange(1 + j, i + 3)
                Factor_3 = np.flip(np.arange(0, i + 2 - j))

            elif coefficient[-1] < 0:
                Factor_2 = np.arange(1, i + 3 - j)
                Factor_3 = np.flip(np.arange(0 + j, i + 2))
            else:
                Factor_2 = np.arange(1, i + 3)
                Factor_3 = np.flip(np.arange(0, i + 2))

            b_1 = np.sum(Factor_1 * Factor_2)
            b_2 = np.sum(Factor_1)

            a_1 = np.sum(Factor_2 * Factor_3)
            a_2 = np.sum(Factor_1 * Factor_3)

            solved_value = solve([b_1 * x - a_1 * y - (i + 2) + Utility * (i + 1), b_2 * x - a_2 * y - 1], [x, y])
            B = float(solved_value[x])
            A = float(solved_value[y])

            coefficient = Factor_1 * B - Factor_3 * A

            if coefficient[0] >= 0 and coefficient[-1] >= 0:
                Repair_matrix[i + 1, Factor_2[0] - 1: Factor_2[-1]] = coefficient
                break
    return Repair_matrix

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

def action_necessary_select(states, time, booling):
    """
    :param state: state is from the tensorflow.numpy()
    pavement: 20 * 132; expansion joint: 20 * 2; sidewalk: 132 * 2; barrier: 132 * 4; drainage: 130/5*2 = 52;
    lighting: 120
    arch: 3 concrete-steel tube (Markov chain);  Transverse: 20 (Markov chain); hanger: 21 * 3 (Markov chain);
    support: 6 (Markov chain)
    Column: 6 (Markov chain); foundation: 6 (Markov chain)
    :return: the standard-based state
    """
    bridge = np.array(
            [132 * 8 * 2, 10 * 4, 132 * 2 * 3, 132 * 4, (132 / 12 + 1) * 2, 8 * 4, 145.16 * 7, 10.7 * 3.17, 132, 10, 6 * 10,
             4.79, 9.04, 12.79, 16.05, 18.84, 21.18, 23.07, 24.53, 25.57, 26.19, 26.4,
             26.19, 25.57, 24.53, 23.07, 21.18, 18.84, 16.05, 12.79, 9.04, 4.79,
             6, 20 * 3 * 4, 8.6 * 3 * 2 + 13.6 * 3 * 2 + 8.6 * 13.6])
    state = states.flatten()
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

    pavement_damage_percentage = 1 - np.sum(pavement_state)
    if pavement_damage_percentage == 0:
        DMCI_pavement = 100
    elif pavement_damage_percentage > 0 and pavement_damage_percentage <= 0.1:
        DMCI_pavement = 75
    elif pavement_damage_percentage > 0.1 and pavement_damage_percentage <= 0.2:
        DMCI_pavement = 60
    else:
        DMCI_pavement = 50

    expansion_damage_percentage = 1 - np.sum(expansion_state)
    if expansion_damage_percentage == 0:
        DMCI_expansion = 100
    elif expansion_damage_percentage > 0 and expansion_damage_percentage <= 0.1:
        DMCI_expansion = 75
    elif expansion_damage_percentage > 0.1 and expansion_damage_percentage <= 0.2:
        DMCI_expansion = 60
    else:
        DMCI_expansion = 50

    sidewalk_damage_percentage = 1 - np.sum(sidewalk_state)
    if sidewalk_damage_percentage == 0:
        DMCI_sidewalk = 100
    elif sidewalk_damage_percentage > 0 and sidewalk_damage_percentage <= 0.1:
        DMCI_sidewalk = 75
    elif sidewalk_damage_percentage > 0.1 and sidewalk_damage_percentage <= 0.2:
        DMCI_sidewalk = 60
    else:
        DMCI_sidewalk = 50

    barrier_damage_percentage = 1 - np.sum(barrier_state)
    if barrier_damage_percentage == 0:
        DMCI_barrier = 100
    elif barrier_damage_percentage > 0 and barrier_damage_percentage <= 0.1:
        DMCI_barrier = 75
    elif barrier_damage_percentage > 0.1 and barrier_damage_percentage <= 0.2:
        DMCI_barrier = 60
    else:
        DMCI_barrier = 50

    drainage_damage_percentage = 1 - np.sum(drainage_state)
    if drainage_damage_percentage == 0:
        DMCI_drainage = 100
    elif drainage_damage_percentage > 0 and drainage_damage_percentage <= 0.1:
        DMCI_drainage = 80
    else:
        DMCI_drainage = 65

    lighting_damage_percentage = 1 - np.sum(lighting_state)
    if lighting_damage_percentage == 0:
        DMCI_lighting = 100
    elif lighting_damage_percentage > 0 and lighting_damage_percentage <= 0.1:
        DMCI_lighting = 75
    elif lighting_damage_percentage > 0.1 and lighting_damage_percentage <= 0.2:
        DMCI_lighting = 60
    else:
        DMCI_lighting = 50

    arch_expected_score = np.zeros((round(len(arch_state)/5)))
    for i in range(round(len(arch_state)/5)):
        arch_expected_score[i] = arch_state[5*i:5*(i+1)] @ np.array([100, 65, 55, 40, 0])

    transverse_expected_score = np.zeros((round(len(transverse_state)/5)))
    for i in range(round(len(transverse_state) / 5)):
        transverse_expected_score[i] = transverse_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    pri_girder_expected_score = np.zeros((round(len(pri_girder_state) / 5)))
    for i in range(round(len(pri_girder_state) / 5)):
        pri_girder_expected_score[i] = pri_girder_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    sec_girder_expected_score = np.zeros((round(len(sec_girder_state) / 5)))
    for i in range(round(len(sec_girder_state) / 5)):
        sec_girder_expected_score[i] = sec_girder_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    slab_expected_score = np.zeros((round(len(slab_state) / 5)))
    for i in range(round(len(slab_state) / 5)):
        slab_expected_score[i] = slab_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    hanger_expected_score = np.zeros((round(len(hanger_state)/5)))
    for i in range(round(len(hanger_state) / 5)):
        hanger_expected_score[i] = hanger_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    support_expected_score = np.zeros((round(len(support_state)/5)))
    for i in range(round(len(support_state) / 5)):
        support_expected_score[i] = support_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

    column_expected_score = np.zeros((round(len(column_state)/5)))
    for i in range(round(len(column_state) / 5)):
        column_expected_score[i] = column_state[5 * i:5 * (i + 1)] @ np.array([100, 65, 55, 40, 0])

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

    DMCI = np.array([DMCI_pavement, DMCI_expansion, DMCI_sidewalk, DMCI_barrier, DMCI_drainage, DMCI_lighting])
    PCCI = np.array(
        [PCCI_arch, PCCI_transverse, PCCI_Pri_girder, PCCI_Sec_girder, PCCI_slab, PCCI_hanger, PCCI_support])
    BCCI = np.array([BCCI_column, BCCI_foundation])

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

    # optional maintenance is worst than necessary maintenance
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

    deck_M_utility = np.concatenate(([pavement_necessary_utility], [pavement_optional_utility],
                                     [expansion_necessary_utility], [expansion_optional_utility],
                                     [sidewalk_necessary_utility], [sidewalk_optional_utility],
                                     [barrier_necessary_utility], [barrier_optional_utility],
                                     [drainage_maintenance_utility], [lighting_maintenance_utility]))
    deck_M_utility = np.round(deck_M_utility, 7)

    superstructure_M_utility = np.concatenate(
        ([arch_utility_necessary], [arch_utility_optional_1], [arch_utility_optional_2],
         [transverse_utility_necessary], [transverse_utility_optional_1], [transverse_utility_optional_2],
         [pri_girder_utility_necessary], [pri_girder_utility_optional_1], [pri_girder_utility_optional_2],
         [sec_girder_utility_necessary], [sec_girder_utility_optional_1], [sec_girder_utility_optional_2],
         [slab_utility_necessary], [slab_utility_optional_1], [slab_utility_optional_2],
         [hanger_utility_necessary], [hanger_utility_optional_1[hanger_utility_less]], [hanger_utility_optional_2],
         [support_utility_necessary], [support_utility_optional_1[support_utility_less]], [support_utility_optional_2]))
    superstructure_M_utility = np.round(superstructure_M_utility, 7)

    substructure_M_utility = np.concatenate(
        ([column_utility_necessary], [column_utility_optional_1], [column_utility_optional_2],
         [foundation_utility_necessary], [foundation_utility_optional_1], [foundation_utility_optional_2]))
    substructure_M_utility = np.round(substructure_M_utility, 7)

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

    necessary_utility = np.array(
        [deck_M_utility[0], deck_M_utility[2], deck_M_utility[4], deck_M_utility[6], deck_M_utility[8],
         deck_M_utility[9], superstructure_M_utility[0], superstructure_M_utility[3],
         superstructure_M_utility[6], superstructure_M_utility[9],
         superstructure_M_utility[12], superstructure_M_utility[15], superstructure_M_utility[18],
         substructure_M_utility[0], substructure_M_utility[3]])

    necessary_utility_list = necessary_utility * necessary_booling
    necessary_utility_index = np.nonzero(necessary_utility_list)[0]

    action_necessary_number = len(necessary_utility_index)

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
    if superstructure_M_utility[2] == 0 or superstructure_M_utility[0] == superstructure_M_utility[2] or \
            superstructure_M_utility[1] == superstructure_M_utility[2]:
        arch_optional_booling_2 = 1
    else:
        arch_optional_booling_2 = 0

    if superstructure_M_utility[4] == 0 or superstructure_M_utility[3] == superstructure_M_utility[4]:
        transverse_optional_booling_1 = 1
    else:
        transverse_optional_booling_1 = 0
    if superstructure_M_utility[5] == 0 or superstructure_M_utility[3] == superstructure_M_utility[5] or \
            superstructure_M_utility[4] == superstructure_M_utility[5]:
        transverse_optional_booling_2 = 1
    else:
        transverse_optional_booling_2 = 0

    if superstructure_M_utility[7] == 0 or superstructure_M_utility[6] == superstructure_M_utility[7]:
        pri_girder_optional_booling_1 = 1
    else:
        pri_girder_optional_booling_1 = 0
    if superstructure_M_utility[8] == 0 or superstructure_M_utility[6] == superstructure_M_utility[8] or \
            superstructure_M_utility[7] == superstructure_M_utility[8]:
        pri_girder_optional_booling_2 = 1
    else:
        pri_girder_optional_booling_2 = 0

    if superstructure_M_utility[10] == 0 or superstructure_M_utility[9] == superstructure_M_utility[10]:
        sec_girder_optional_booling_1 = 1
    else:
        sec_girder_optional_booling_1 = 0
    if superstructure_M_utility[11] == 0 or superstructure_M_utility[9] == superstructure_M_utility[11] or \
            superstructure_M_utility[10] == superstructure_M_utility[11]:
        sec_girder_optional_booling_2 = 1
    else:
        sec_girder_optional_booling_2 = 0

    if superstructure_M_utility[13] == 0 or superstructure_M_utility[12] == superstructure_M_utility[13]:
        slab_optional_booling_1 = 1
    else:
        slab_optional_booling_1 = 0
    if superstructure_M_utility[14] == 0 or superstructure_M_utility[12] == superstructure_M_utility[14] or \
            superstructure_M_utility[13] == superstructure_M_utility[14]:
        slab_optional_booling_2 = 1
    else:
        slab_optional_booling_2 = 0

    if superstructure_M_utility[16] > 1e10 or superstructure_M_utility[15] == superstructure_M_utility[16]:
        hanger_optional_booling_1 = 1
    else:
        hanger_optional_booling_1 = 0
    if superstructure_M_utility[17] == 0 or superstructure_M_utility[15] == superstructure_M_utility[17] or \
            superstructure_M_utility[16] == superstructure_M_utility[17]:
        hanger_optional_booling_2 = 1
    else:
        hanger_optional_booling_2 = 0

    if superstructure_M_utility[19] > 1e10 or superstructure_M_utility[18] == superstructure_M_utility[19]:
        support_optional_booling_1 = 1
    else:
        support_optional_booling_1 = 0
    if superstructure_M_utility[20] == 0 or superstructure_M_utility[18] == superstructure_M_utility[20] or \
            superstructure_M_utility[19] == superstructure_M_utility[20]:
        support_optional_booling_2 = 1
    else:
        support_optional_booling_2 = 0

    if substructure_M_utility[1] == 0 or substructure_M_utility[0] == substructure_M_utility[1]:
        column_optional_booling_1 = 1
    else:
        column_optional_booling_1 = 0
    if substructure_M_utility[2] == 0 or substructure_M_utility[0] == substructure_M_utility[2] or \
            substructure_M_utility[1] == substructure_M_utility[2]:
        column_optional_booling_2 = 1
    else:
        column_optional_booling_2 = 0

    if substructure_M_utility[4] == 0 or substructure_M_utility[3] == substructure_M_utility[4]:
        foundation_optional_booling_1 = 1
    else:
        foundation_optional_booling_1 = 0
    if substructure_M_utility[5] == 0 or substructure_M_utility[3] == substructure_M_utility[5] or \
            substructure_M_utility[4] == substructure_M_utility[5]:
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
         pri_girder_optional_booling_2, sec_girder_optional_booling_2, slab_optional_booling_2,
         hanger_optional_booling_2,
         support_optional_booling_2,
         column_optional_booling_2, foundation_optional_booling_2])

    optional_utility_index = np.where(optional_booling == 0)[0]
    action_optional_number = len(optional_utility_index)

    if booling:
        action_maintenance_number = action_necessary_number + int(action_optional_number / 4)
        if time % 30 == 0 and time > 0:
            action = np.array([1, 1, 1, 1, 1, 1, action_maintenance_number])
        elif time % 15 == 0 and time > 0:
            action = np.array([1, 1, 0, 1, 1, 1, action_maintenance_number])
        elif time % 10 == 0 and time > 0:
            action = np.array([1, 1, 1, 0, 1, 0, action_maintenance_number])
        elif time % 5 == 0 and time > 0:
            action = np.array([1, 1, 0, 0, 1, 0, action_maintenance_number])
        else:
            action = np.array([1, 0, 0, 0, 0, 0, action_maintenance_number])
        action_select = action_decoding(action)
    else:
        action_maintenance_number = action_necessary_number + int(action_optional_number / 2.86)
        if time % 30 == 0 and time > 0:
            action = np.array([0, 1, 1, 1, 1, 1, action_maintenance_number])
        elif time % 15 == 0 and time > 0:
            action = np.array([0, 1, 0, 1, 1, 1, action_maintenance_number])
        elif time % 10 == 0 and time > 0:
            action = np.array([0, 1, 1, 0, 1, 0, action_maintenance_number])
        elif time % 5 == 0 and time > 0:
            action = np.array([0, 1, 0, 0, 1, 0, action_maintenance_number])
        else:
            action = np.array([0, 0, 0, 0, 0, 0, action_maintenance_number])
        action_select = action_decoding(action)

    return action_select

