# Optimal Graph-State Generation for Measurement-Based Quantum Computation

A reinforcement learning project for finding **optimal graph-state transformation sequences** for **Measurement-Based Quantum Computation (MBQC)** using **Proximal Policy Optimization (PPO)** and **Generative Flow Networks (GFlowNets)**.

## Overview

The goal is to learn sequences of valid graph transformations that transform an initial graph state into a target graph state while minimizing the transformation cost.

The problem is formulated as a sequential decision process:

```text
Initial Graph State
        │
        ▼
   Graph Operation
        │
        ▼
Intermediate State
        │
        ▼
   Graph Operation
        │
        ▼
       ...
        │
        ▼
 Target Graph State
````

Each **state** represents a graph state, while each **action** corresponds to a valid graph transformation.

## Methods

### PPO

**Proximal Policy Optimization (PPO)** learns a policy

$$
\pi_\theta(a \mid s)
$$

that selects graph transformations based on the current graph state.

The clipped PPO objective is

$$
L^{CLIP}(\theta)=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta) A_t,
\mathrm{clip}
\left(
r_t(\theta),1-\epsilon,1+\epsilon
\right) A_t
\right)
\right]
$$

where $A_t$ is the estimated advantage and

$$
r_t(\theta)=
\frac{
\pi_\theta(a_t \mid s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)
}.
$$

PPO is used to learn a policy that efficiently discovers high-reward transformation sequences.

### GFlowNet

A **Generative Flow Network (GFlowNet)** learns a distribution over graph-state transformation trajectories rather than optimizing for a single solution.

A trajectory is represented as

$$
s_0 \rightarrow s_1 \rightarrow \cdots \rightarrow s_T
$$

where each transition is produced by applying a valid graph transformation:

$$
s_{t+1}=T(s_t,a_t).
$$

The GFlowNet learns a forward policy

$$
P_F(a_t \mid s_t)
$$

that determines which transformation to apply at each state.

The objective is to generate trajectories with probability proportional to their terminal reward:

$$
P(\tau) \propto R(s_T),
$$

where $\tau$ is a complete transformation trajectory and $R(s_T)$ is the reward of the resulting graph state.

For a state $s$, the flow is distributed among its possible actions:

$$
F(s)=\sum_a F(s,a)
$$

with the forward policy given by

$$
P_F(a\mid s)=
\frac{F(s,a)}
{\sum_{a'}F(s,a')}.
$$

For terminal states, the flow is matched to the reward:

$$
F(x)=R(x).
$$

This makes GFlowNets useful for MBQC graph-state generation because **multiple high-quality transformation sequences may exist**. Instead of converging to a single solution, the model can explore a diverse set of high-reward sequences.

### PPO vs GFlowNet

|             | PPO                         | GFlowNet                                            |
| ----------- | --------------------------- | --------------------------------------------------- |
| Objective   | Maximize expected reward    | Learn a reward-proportional trajectory distribution |
| Output      | High-reward sequence/policy | Diverse high-reward sequences                       |
| Exploration | Policy-based                | Generative                                          |
| Main use    | Find an efficient solution  | Discover multiple good solutions                    |

## Objective

The learned transformation sequence should:

* Reach the target graph state
* Use only valid graph transformations
* Minimize transformation cost
* Efficiently generate graph states for MBQC
* Explore alternative high-quality transformation sequences

## Workflow

```text
                 Initial Graph State
                         │
                         ▼
                 State Representation
                         │
                         ▼
                  ┌────────────────┐ 
                  │ PPO / GFlowNet |
                  └───────┬────────┘
                          │
                          ▼
                  Select Transformation
                          │
                          ▼
                   New Graph State
                          │
                          └───────────┐
                                      │
                                      ▼
                              Repeat Until
                              Target Reached
                                      │
                                      ▼
                               Evaluate Cost
```

