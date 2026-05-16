# RL-Agents 🤖
**Implementing and comparing classical reinforcement learning algorithms from scratch in Python.**

A collection of reinforcement learning algorithms implemented from scratch in Python, covering bandit methods, dynamic programming, and temporal difference learning. Each module includes statistical analysis and Matplotlib visualizations comparing algorithm performance across stochastic environments.

---

## Algorithms Implemented

### Bandit Algorithms (10-arm, Bernoulli rewards)
| Algorithm | File | Description |
|---|---|---|
| Upper Confidence Bound (UCB) | `q1.py`, `q2.py` | Exploration via confidence bonuses, tuned across 8 c-values |
| Linear Reward-Inaction (LR-I) | `q2.py` | Probability-based selection, updates only on reward |
| Linear Reward-Penalty (LR-P) | `q2.py` | Probability-based selection, updates on reward and penalty |

### Dynamic Programming (4×4 Stochastic Gridworld)
| Algorithm | File | Description |
|---|---|---|
| Policy Iteration | `pia.py` | Iterative policy evaluation + greedy improvement, with per-iteration timing |
| Value Iteration | `via.py` | Bellman optimality sweeps until convergence (δ < θ) |

### Temporal Difference Learning (10×10 Gridworld with walls)
| Algorithm | File | Description |
|---|---|---|
| Monte Carlo | `mc.py` | On-policy first-visit MC with episode-end Q-value updates |
| SARSA | `sarsa.py` | On-policy TD learning, updates using Q(s', a') |
| Q-Learning | `q.py` | Off-policy TD learning, updates using max Q(s', *) |
| Comparison | `comparison.py` | All three across varying α ∈ {0.05, 0.1, 0.2} and ε ∈ {0.05, 0.1, 0.2} |

Environment: 10×10 grid with interior walls and 4 door passages, goal at state 9 (top-right), reward −1 per step / +500 at goal.

---

## Key Results

### Bandit Algorithms (5000 steps, 100 environments, 10 arms)
| Algorithm | Avg Reward | Optimal Action % |
|---|---|---|
| UCB (c=0.5) | **0.9174** | **84.61%** |
| LR-I (α=0.05) | 0.9004 | 66.79% |
| LR-P (α=0.5, β=0.1) | 0.8703 | 62.18% |

UCB outperformed both Learning Automata. LR-P converged faster early but plateaued sooner. Lower c-values in UCB led to faster convergence; c=0.5–0.75 was optimal.

### Dynamic Programming (4×4 Gridworld, γ=0.95, θ=0.001)
- Both algorithms converged to the **same optimal policy** across all 4 experimental setups
- **Policy Iteration**: 4–6 improvement iterations (first evaluation sweep dominates runtime)
- **Value Iteration**: 7–20 Bellman sweeps depending on stochasticity
- Deterministic environment (p1=1.0) converged ~3× faster than highly stochastic (p1=p2=0.25)

### TD Learning (5000 episodes, 10×10 gridworld, p1=1.0, p2=0.0, α=0.1, ε=0.1)
| Algorithm | Total Steps | Steps/Manhattan Dist | Computation Time |
|---|---|---|---|
| Monte Carlo | 794,439 | 17.52 | 11.3s |
| SARSA | 64,723 | 1.45 | 0.97s |
| Q-Learning | **63,190** | **1.41** | 1.08s |

- TD methods were **91.7% more efficient** than Monte Carlo
- Best hyperparameters: **α=0.2, ε=0.05** for both SARSA and Q-Learning
- Monte Carlo and TD methods respond **inversely** to ε: MC benefits from more exploration; SARSA/Q-Learning benefit from less

---

## Setup

```bash
git clone https://github.com/shaghayegh0/RL-Agents.git
cd RL-Agents
pip install numpy matplotlib
```

---

## Usage

```bash
# UCB (outputs reward and optimal action % every 100 steps)
python3 q1.py > output.txt

# LR-I, LR-P vs UCB comparison
python3 q2.py > output2.txt

# Policy Iteration and Value Iteration (prompts for p1, p2, rewards)
python3 pia.py
python3 via.py

# Example input for Dynamic Programming:
# Enter p1: 1.0
# Enter p2: 0.0
# Enter r_up: -1
# Enter r_down: -1
# Enter r_right: -1
# Enter r_left: -1

# Individual algorithms (prompts for p1, p2)
python3 mc.py
python3 sarsa.py
python3 q.py

# Full hyperparameter comparison
python3 comparison.py
```

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

No external ML frameworks — all algorithms implemented from scratch.

---

## Tech Stack

`Python` `NumPy` `Matplotlib` `Reinforcement Learning` `MDP` `Bandit Algorithms` `Dynamic Programming` `TD Learning` `Q-Learning` `SARSA` `Monte Carlo`
