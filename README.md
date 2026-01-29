# Intelligent Bridge Management System

This repository presents an **AI-assisted bridge inspection & maintenance planning framework** based on **Deep Reinforcement Learning (DRL)** with **expert demonstrations**. The goal is to keep structural risk within an acceptable level while minimizing life-cycle maintenance cost, addressing limitations of traditional bridge management that often relies on conservative expert judgment and fixed inspection intervals.

Four scripts (`X_bridge_management.py`, where `X = A2C, PPO, SAC, ImitationL`) are provided to train decision-making agents. The trained agents/models are saved and uploaded in the repository. Using `Inspection&maintenance.py` and modifying the agent loading path (e.g., `load.(path)`), you can test and compare different methods' performance.

## Motivation

Bridge deterioration reduces structural capacity and can affect both serviceability and ultimate limit states, potentially leading to failure. Meanwhile, maintenance budgets are constrained and bridges are aging worldwide. A key challenge is to **optimize inspection and treatment decisions over the service life** under uncertainty, cost, and safety/risk constraints.

## Key Ideas & Contributions

This project (and the associated paper) develops core techniques for component-level bridge life-cycle management:

- **Actor–Critic DRL for maintenance decision-making**: learns a policy that maps structural states to inspection & maintenance actions.
- **Hybrid Markov decision processes (MDPs)**: accommodates **heterogeneous deterioration patterns** (different component types/models) within a unified decision process.
- **Action-space simplification via ranking**: reduces decision complexity from **exponential** to **linear growth** with the number of components.
- **Imitation learning + DRL training mechanism**: integrates expert demonstrations to guide early exploration, improve convergence stability in complex environments, and accelerate training.

## Training Performance (Example)

![Reward cumulative curve](Rewardcum.jpg)

## Requirements

- **Python**: 3.9
- **TensorFlow**: >= 2.12.0

## Repository Structure / Main Scripts

- `A2C_bridge_management.py`
- `PPO_bridge_management.py`
- `SAC_bridge_management.py`
- `ImitationL_bridge_management.py`

These scripts are used to train decision-making agents with different algorithms.

- `Inspection&maintenance.py`  
Run evaluation/testing. You can test different trained methods by changing the model loading path (e.g., `load.(path)` in the script).

## How to Use (Quick Start)

1. Train an agent with one of the following scripts:
   - `A2C_bridge_management.py`, `PPO_bridge_management.py`, `SAC_bridge_management.py`, or `ImitationL_bridge_management.py`

2. Evaluate the trained agent:
   - Open `Inspection&maintenance.py`
   - Modify the agent/model loading path (e.g., `load.(path)`)
   - Run the script to compare performance across methods

## Related Paper

**AI-assisted Bridge Management System based on Deep Reinforcement Learning with Expert Demonstrations**  
Li LAI, You DONG*, Aijun Wang, Dan M. Frangopol

Affiliations:  
State Key Laboratory of Climate Resilience for Coastal Cities, Department of Civil and Environmental Engineering, The Hong Kong Polytechnic University, Hong Kong, China  
Alibaba Group, Hangzhou, China  
Department of Civil and Environmental Engineering, ATLSS Engineering Research Center, Lehigh University, Bethlehem, Pennsylvania, USA

## Keywords

Deep reinforcement learning; Imitation learning; Bridge management; Weibull model; Markov model; Maintenance optimization
