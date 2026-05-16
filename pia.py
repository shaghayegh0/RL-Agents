# CPS824 W2026
# Assignment 2 q3 - Policy Iteration Algorithm
# Maarya Siddiqui (501159595)
# Shaghayegh Dehghanisanij (501080180)

import numpy as np
import time

GAMMA = 0.95    # Discount factor
THETA = 0.001   # Convergence factor
GRID_SIZE = 4
NUM_STATES = GRID_SIZE * GRID_SIZE

v = np.zeros(NUM_STATES)    # value function
TERMINAL_STATE = 0          # terminal state position @ state 0

# ACTIONS
actions = {
    "u": (-1, 0),   # move up
    "d": (1, 0),    # move down
    "r": (0, 1),    # move right
    "l": (0, -1)    # move left
}

# HELPER FUNCTION TO CONVERT (row, col) --> s
def get_state(row, col):
    '''Given row and col, return corresponding state'''
    return row * GRID_SIZE + col

# HELPER FUNCTION TO CONVERT s --> (row, col)
def get_coords(s):
    '''Given state, return coordinates of row and col'''
    row = s // GRID_SIZE
    col = s % GRID_SIZE
    return row, col

# HELPER FUNCTION TO DERIVE NEXT STATE GIVEN CURRENT STATE AND DESIRED ACTION
def next_state(row, col, a, GRID_SIZE):
    '''Param:
            - row: current row in state s
            - col: current column in state s
            - a: desired action to go from s to s'
            - GRID_SIZE: the dimension of our grid (4x4)
        Return:
            - state index for s'
    '''
    row_dir, col_dir = actions[a]
    new_r = row + row_dir
    new_col = col + col_dir

    if (0 <= new_r < GRID_SIZE) and (0 <= new_col < GRID_SIZE):
        return get_state(new_r, new_col)
    else:
        return get_state(row, col)

# HELPER FUNCTION TO CALCULATE PROBABILITIES OF TRANSITION P(s' | s, a)
def get_transition_probs(s, a, adjacent, p1, p2, GRID_SIZE):
    '''Param:
            - s: current state (value between 0-15)
            - a: desired action to go from s to s'
            - adjacent: dictionary of adjacent actions per intended action
            - p1 & p2: probabilities of transition as defined by user
            - GRID_SIZE: the dimension of our grid (4x4)
        Return:
            - transitions: dictionary with keys being next states s'
                            and values being corresponding probability of transition
    '''
    row, col = get_coords(s)
    transitions = {}

    # 1. TRANSITION VIA INTENDED ACTION
    new_s = next_state(row, col, a, GRID_SIZE)
    new_row, new_col = get_coords(new_s)
    transitions[new_s] = p1 + transitions.get(new_s, 0)

    # 2. TRANSITION TO SAME STATE (stay in place)
    transitions[s] = p2 + transitions.get(s, 0)

    # 3. TRANSITION TO TARGET STATES' ADJACENT STATES
    for adj_action in adjacent[a]:
        new_adj_state = next_state(new_row, new_col, adj_action, GRID_SIZE)
        transitions[new_adj_state] = ((1 - p1 - p2) / 2) + transitions.get(new_adj_state, 0)

    return transitions

# POLICY EVALUATION STEP
def policy_evaluation(policy, v, actions, adjacent, r, p1, p2, GRID_SIZE, NUM_STATES, GAMMA, THETA):
    '''Iteratively evaluate the given policy until convergence (delta < theta).
        Params:
            - policy: current policy array (one action per state)
            - v: current value function array
            - (others): shared parameters
        Returns:
            - v: updated value function
            - eval_iters: number of sweeps performed
    '''
    eval_iters = 0
    while True:
        eval_iters += 1
        delta = 0
        updated_V = v.copy()

        for s in range(NUM_STATES):
            if s == TERMINAL_STATE:
                continue

            a = policy[s]   # action prescribed by current policy
            transitions = get_transition_probs(s, a, adjacent, p1, p2, GRID_SIZE)

            new_val = 0
            for s_prime, prob in transitions.items():
                new_val += prob * (r[a] + GAMMA * v[s_prime])

            delta = max(delta, abs(new_val - v[s]))
            updated_V[s] = new_val

        v = updated_V

        if delta < THETA:
            break

    return v, eval_iters

# POLICY IMPROVEMENT STEP
def policy_improvement(policy, v, actions, adjacent, r, p1, p2, GRID_SIZE, NUM_STATES, GAMMA):
    '''Greedily improve the policy based on the current value function.
        Returns:
            - new_policy: updated policy after one greedy improvement pass
            - policy_stable: True if no policy changes were made
    '''
    policy_stable = True
    new_policy = policy.copy()

    for s in range(NUM_STATES):
        if s == TERMINAL_STATE:
            continue

        old_action = policy[s]
        q_a = {}

        for a in actions.keys():
            transitions = get_transition_probs(s, a, adjacent, p1, p2, GRID_SIZE)
            expected_val = 0
            for s_prime, prob in transitions.items():
                expected_val += prob * (r[a] + GAMMA * v[s_prime])
            q_a[a] = expected_val

        best_action = max(q_a, key=q_a.get)
        new_policy[s] = best_action

        if best_action != old_action:
            policy_stable = False

    return new_policy, policy_stable

# FULL POLICY ITERATION ALGORITHM
def policy_iteration(v, actions, adjacent, r, p1, p2, GRID_SIZE, NUM_STATES, GAMMA, THETA):
    '''Implementation of the policy iteration algorithm.
        Returns a 4-element tuple:
            - v: array of state values
            - policy: the optimal policy
            - k: number of policy improvement iterations until stability
            - iter_times: list of elapsed time per outer iteration (in seconds)
    '''
    # Initialize with an equiprobable random policy (random action per state)
    policy = [np.random.choice(list(actions.keys())) for _ in range(NUM_STATES)]
    policy[TERMINAL_STATE] = ""   # terminal state has no action

    k = 0
    iter_times = []

    while True:
        k += 1
        iter_start = time.perf_counter()

        # Step 1: Policy Evaluation -- evaluate current policy to convergence
        v, _ = policy_evaluation(policy, v, actions, adjacent, r, p1, p2,
                                  GRID_SIZE, NUM_STATES, GAMMA, THETA)

        # Step 2: Policy Improvement -- greedily update policy
        policy, policy_stable = policy_improvement(policy, v, actions, adjacent, r,
                                                    p1, p2, GRID_SIZE, NUM_STATES, GAMMA)

        iter_time = time.perf_counter() - iter_start
        iter_times.append(iter_time)

        # Convergence: policy did not change in this iteration
        if policy_stable:
            break

    return v, policy, k, iter_times

def display_values(v):
    '''Displays state values as a grid'''
    value_grid = v.reshape(GRID_SIZE, GRID_SIZE)
    line = "+" + ("-" * 6 + "+") * GRID_SIZE
    print(line)
    for row in value_grid:
        print("|", end="")
        for val in row:
            print(f"{val:^{6}.2f}|", end="")
        print()
        print(line)

def display_policy(p):
    '''Displays policy as a grid'''
    policy_grid = np.array(p).reshape(GRID_SIZE, GRID_SIZE)
    line = "+" + ("-" * 6 + "+") * GRID_SIZE
    print(line)
    for row in policy_grid:
        print("|", end="")
        for val in row:
            print(f"{val:^{6}}|", end="")
        print()
        print(line)

def main():
    '''Mainline for the program'''
    # Transition semantics:
    # 1. Move to desired state          --> p1
    # 2. Remain in same state           --> p2
    # 3. Move to adjacent of target     --> (1-p1-p2)/2 each
    # 4. Go off grid (intended)         --> stay: p1+p2, adjacent target: (1-p1-p2)/2

    # Get user input for probabilities and rewards
    p1 = float(input("Enter p1: "))
    p2 = float(input("Enter p2: "))

    r_up    = float(input("Enter r_up: "))
    r_down  = float(input("Enter r_down: "))
    r_r     = float(input("Enter r_right: "))
    r_l     = float(input("Enter r_left: "))

    # REWARDS
    r = {
        "u": r_up,
        "d": r_down,
        "r": r_r,
        "l": r_l
    }

    # ADJACENT CELL MAPPING
    adjacent = {
        "u": ["l", "r"],    # adjacent states of target when moving up
        "d": ["l", "r"],    # adjacent states of target when moving down
        "r": ["u", "d"],    # adjacent states of target when moving right
        "l": ["u", "d"]     # adjacent states of target when moving left
    }

    # Perform policy iteration algorithm
    value_function, optimal_policy, k, iter_times = policy_iteration(
        v, actions, adjacent, r, p1, p2, GRID_SIZE, NUM_STATES, GAMMA, THETA
    )

    print("\n===== POLICY ITERATION RESULTS =====\n")

    print(f"POLICY IMPROVEMENT ITERATIONS UNTIL CONVERGENCE: {k}")
    print("\nPER-ITERATION COMPUTATION TIMES:")
    for i, t in enumerate(iter_times, 1):
        print(f"  Iteration {i}: {t*1000:.4f} ms")
    print(f"  Total time: {sum(iter_times)*1000:.4f} ms")

    print("\nSTATE VALUES")
    display_values(value_function)

    print("\nOPTIMAL POLICY")
    display_policy(optimal_policy)

main()
