# CPS824 W2026
# Assignment 3 q3 - COMPARISON PROGRAM
# Maarya Siddiqui (501159595)
# Shaghayegh Dehghanisanij (501080180)

import time
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

P1 = 1.0
P2 = 0.0
GRID_SIZE = 10
NUM_STATES = 100
GOAL_STATE = 9
EPISODES = 5000
N_RUNS = 3

# Door locations for agent to walk through to different rooms
doors = {(2,4), (7,4), (4,2), (4,7)}

v = np.zeros(NUM_STATES)    # value function                          

# ACTIONS
actions = {
        "u":(-1,0),             # move up
        "d":(1,0),              # move down
        "r":(0,1),              # move right
        "l":(0,-1)              # move left
    }

# ADJACENT CELL MAPPING
adjacent = {
        "u":["l", "r"],     # adjacent states are left & right
        "d":["l", "r"],     # adjacent states are left & right
        "r":["u", "d"],     # adjacent states are up & down
        "l":["u", "d"]      # adjacent states are up & down
    }

# HELPER FUNCTION TO DETERMINE IF AGENT IS WALKING INTO A WALL
def is_wall(row, col):
    '''Given row and col, determine if we're walking into a wall'''
    # Vertical Wall
    if col == 4 and row not in [2,7]:
        return True
    
    # Horizontal Wall
    if row == 4 and col not in [2,7]:
        return True
    
    return False

# HELPER FUNCTION TO CONVERT (row, col) --> s
def get_state(row, col):
    '''Given row and col, return corresponding state'''
    return row*GRID_SIZE + col

# HELPER FUNCTION TO CONVERT s --> (row, col)
def get_coords(s):
    '''Given state, return coordinates of row and col'''
    row = s // GRID_SIZE 
    col = s%GRID_SIZE 
    return row, col

# HELPER FUNCTION TO DERIVE NEXT STATE GIVEN CURRENT STATE AND DESIRED ACTION
def next_state(row, col, a, GRID_SIZE):
    '''Param:
            - row: current row in state s
            - col: current column in state s
            - a: desired action to go from s to s'
            - GRID_SIZE: the dimension of our grid (4x4)
        Return:
            - tuple (row, col): row and col for state s'
    '''
    row_dir, col_dir = actions[a]   # obtain movement directions based on desired action, a {"u", "d", "l", "r"}
    new_r = row + row_dir           # calculate row for new state s'
    new_col = col + col_dir         # calculate col for new state s'

    # If the move is valid (does not go off the board), return coordinates of new state s'
    # Otherwise return the current state, s
    if not ((0 <= new_r < GRID_SIZE) and (0 <= new_col < GRID_SIZE)):
        return get_state(row, col)
    
    # Ensure we aren't running into a wall
    if is_wall(new_r, new_col):
        return get_state(row, col)
    
    return get_state(new_r, new_col)

# HELPER FUNCTION TO CALCULATE PROBABILITIES OF TRANSITION P(s' | s, a)
def get_transition_probs(s, a, adjacent, p1, p2, GRID_SIZE):
    '''Param:
            - s: current state (value between 0-15)
            - a: desired action to go from s to s'
            - p1 & p2: probabilities of transition as defined by user
            - GRID_SIZE: the dimension of our grid (4x4)
        Return:
            - transitions: dictionary with keys being next states s' 
                            and the values being corresponding probability of transition from s to s'
    '''

    # Derive current row and col from current state
    row, col = get_coords(s)      

    # Initialize dictionary to store transition probabilities
    transitions = {}

    # 1. TRANSITION VIA INTENDED ACTION
    # Determine coordinates of new state s' given the desired action using next_state helper function
    new_s = next_state(row, col, a, GRID_SIZE)
    new_row, new_col = get_coords(new_s)

    # Probability of s -> s' = p1 given we follow intended action
    # However, transitions.get(new_s, 0) returns 0 by default if key doesn't exist but if it does it returns value already stored
    # Covers edge case where we intend to move off grid and end up at same state (p1+p2 situation)
    transitions[new_s] = p1 + transitions.get(new_s, 0)

    # 2. TRANSITION TO SAME STATE
    transitions[s] = p2 + transitions.get(s, 0)

    # 3. TRANSITION TO TARGET STATES' ADJACENT STATE
    for adj_action in adjacent[a]:
        # For each adj_action you can take, find s' (s is the target state if we had actually performed the intended action)
        new_adj_state = next_state(new_row, new_col, adj_action, GRID_SIZE)

        # For each s' compute the probability of transitioning to it
        transitions[new_adj_state] = ((1-p1-p2)/2) + transitions.get(new_adj_state, 0)

    return transitions


def monte_carlo(p1,p2,episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    '''Implementation of the on-policy monte-carlo algorithm'''

    # a NUM_STATES by NUM_ACTIONS array
    # Q[s,a] represents the estimated expected return when the agent starts
    # in state, s, takes action a and continues following epsilon-soft policy
    # Initialized to 0
    Q = np.zeros((NUM_STATES,4))

    action_list = list(actions.keys())
    action_index = {a:i for i,a in enumerate(action_list)}

    total_steps = 0             # track overall steps
    total_distance = 0          # track total manhattan distance
    start_time = time.time()    # start timer

    # Loop through all episodes
    for ep in range(episodes):
        # Choose a random state to begin the episode
        while True:
            s = np.random.randint(NUM_STATES)
            r,c = get_coords(s)
            if not is_wall(r,c):
                break
        
        # Manhattan distance from start to goal
        goal_r, goal_c = get_coords(GOAL_STATE)
        total_distance += max(abs(r - goal_r) + abs(c - goal_c), 1)

        # Array to store information about this current episode
        # Format is [(state, action, reward)...]
        episode = []

        # Run through the episode. Episode terminates when agent reaches the goal state
        while s != GOAL_STATE:
            if np.random.rand() < epsilon:
                a = np.random.choice(action_list)   # explore
            else:
                a = action_list[np.argmax(Q[s])]    # exploit

            transitions = get_transition_probs(s,a,adjacent, p1,p2,GRID_SIZE)

            states = list(transitions.keys())
            probs = list(transitions.values())

            next_s = np.random.choice(states, p=probs)

            # Reward is -1 unless we are at the goal state
            reward = -1
            if next_s == GOAL_STATE:
                reward = 500

            # Append a tuple of this rounds' stats to the episode array
            episode.append((s,a,reward))

            s = next_s
        
        total_steps += len(episode) # accumulate steps after each episode

        # G is the return
        G = 0
        for s,a,r in reversed(episode):
            G = gamma*G + r
            idx = action_index[a]
            Q[s][idx] += alpha*(G - Q[s][idx])

    computation_time = time.time() - start_time     # stop timer

    return Q, episodes, total_steps, computation_time, total_distance

def sarsa(p1,p2,episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    '''Implementation of the SARSA algorithm'''

    # a NUM_STATES by NUM_ACTIONS array
    # Q[s,a] represents the estimated expected return when the agent starts
    # in state, s, takes action a and continues following epsilon-soft policy
    # Initialized to 0
    Q = np.zeros((NUM_STATES,4))

    action_list = list(actions.keys())
    action_index = {a:i for i,a in enumerate(action_list)}

    total_steps = 0
    total_distance = 0
    start_time = time.time()

    # Loop through all episodes
    for ep in range(episodes):

        # Choose a random state to begin the episode
        while True:
            s = np.random.randint(NUM_STATES)
            r,c = get_coords(s)
            if not is_wall(r,c):
                break

        goal_r, goal_c = get_coords(GOAL_STATE)
        total_distance += max(abs(r - goal_r) + abs(c - goal_c), 1)

        # Select first action with epsilon-greedy
        if np.random.rand() < epsilon:
            a = np.random.choice(action_list)   # explore
        else:
            a = action_list[np.argmax(Q[s])]    # exploit

        steps = 0   # tracking steps per episode

        # Run through the episode. 
        # Episode terminates when agent reaches the goal state
        while s != GOAL_STATE:
            transitions = get_transition_probs(s,a,adjacent, p1,p2,GRID_SIZE)
            states = list(transitions.keys())
            probs = list(transitions.values())

            next_s = np.random.choice(states, p=probs)

            # Reward is -1 unless we are at the goal state
            reward = -1
            if next_s == GOAL_STATE:
                reward = 500

            # Choose NEXT action with epsilon greedy again
            if np.random.rand() < epsilon:
                next_a = np.random.choice(action_list)   # explore
            else:
                next_a = action_list[np.argmax(Q[next_s])]    # exploit

            # SARSA update: uses Q(s',a') not max Q(s', *)
            idx = action_index[a]
            next_idx = action_index[next_a]
            # gamma * Q[next_s][next_idx] is SARSA's way of calculating
            # future return without waiting (bootstrapping)
            Q[s][idx] += alpha * (reward + gamma * Q[next_s][next_idx] - Q[s][idx])

            # Advance in episode, both the state and action carry forward
            s = next_s
            a = next_a
            steps += 1

        total_steps += steps
    
    computation_time = time.time() - start_time

    return Q, episodes, total_steps, computation_time, total_distance

def q_learning(p1,p2,episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    '''Implementation of the q learning algorithm'''

    # a NUM_STATES by NUM_ACTIONS array
    # Q[s,a] represents the estimated expected return when the agent starts
    # in state, s, takes action a and continues following epsilon-soft policy
    # Initialized to 0
    Q = np.zeros((NUM_STATES,4))

    action_list = list(actions.keys())
    action_index = {a:i for i,a in enumerate(action_list)}

    total_steps = 0
    total_distance = 0
    start_time = time.time()

    # Loop through all episodes
    for ep in range(episodes):
        # Choose a random state to begin the episode
        while True:
            s = np.random.randint(NUM_STATES)
            r,c = get_coords(s)
            if not is_wall(r,c):
                break

        goal_r, goal_c = get_coords(GOAL_STATE)
        total_distance += max(abs(r - goal_r) + abs(c - goal_c), 1)

        steps = 0

        # Run through the episode. Episode terminates when agent reaches the goal state
        while s != GOAL_STATE:
            if np.random.rand() < epsilon:
                a = np.random.choice(action_list)   # explore
            else:
                a = action_list[np.argmax(Q[s])]    # exploit

            # sample next state using transition probability function
            transitions = get_transition_probs(s,a,adjacent, p1,p2,GRID_SIZE)

            states = list(transitions.keys())
            probs = list(transitions.values())

            next_s = np.random.choice(states, p=probs)

            # Reward is -1 unless we are at the goal state
            reward = -1
            if next_s == GOAL_STATE:
                reward = 500

            # Perform the Q-Learning update
            # uses max over all actions at the next state - the value of the next possible
            idx = action_index[a]
            Q[s][idx] += alpha * (reward + gamma * np.max(Q[next_s]) - Q[s][idx])

            s = next_s
            steps += 1

        total_steps += steps

    computation_time = time.time() - start_time

    return Q, episodes, total_steps, computation_time, total_distance

# HELPER FUNCTION TO DETERMINE POLICY BASED ON MONTE-CARLO ALGORITHM
def extract_policy(Q):
    '''Returns 1D array of optimal policy'''
    policy = []
    action_list = list(actions.keys())
    for s in range(NUM_STATES):

        # find index of best action
        best_action_idx = np.argmax(Q[s])

        # find corresponding action string
        best_action = action_list[best_action_idx]

        policy.append(best_action)

    return policy

def display_policy(p):
    policy_grid = np.array(p).reshape(GRID_SIZE, GRID_SIZE)
    line = "+" + ("-"*6 + "+") * GRID_SIZE
    print(line)
    for row in policy_grid:
        print("|", end="")
        for v in row:
            print(f"{v:^{6}}|", end="")
        print()
        print(line)

alpha_values = [0.05, 0.1, 0.2]
epsilon_values = [0.05, 0.1, 0.2]
algo_names = ["Monte Carlo", "SARSA", "Q-Learning"]
algo_fns = [monte_carlo, sarsa, q_learning]

def main():

    # ── varying alpha (fixed epsilon=0.1) ─────────────────────────────────
    print("\n===== VARYING ALPHA (epsilon=0.1 fixed) =====\n")
    print(f"{'Algorithm':<15} {'Alpha':<10} {'Total Steps':<15} {'Steps/Manhattan':<15} {'Time (s)':<15} {'Time/Manhattan (s)':<15}")
    print("-" * 85)

    for alpha in alpha_values:
        for name, fn in zip(algo_names, algo_fns):
            all_steps = []
            all_steps_dist = []
            all_times = []
            all_times_dist = []
            for _ in range(N_RUNS):
                _, _, total_steps, comp_time, total_dist = fn(P1, P2,
                                                   episodes=EPISODES,
                                                   alpha=alpha,
                                                   epsilon=0.1)
                all_steps.append(total_steps)
                all_steps_dist.append(total_steps / total_dist)
                all_times.append(comp_time)
                all_times_dist.append(comp_time / total_dist)
            print(f"{name:<15} {alpha:<10} {np.mean(all_steps):<15.0f} {np.mean(all_steps_dist):<15.4f} {np.mean(all_times):<15.4f} {np.mean(all_times_dist):<15.6f}")
        print()

    # ── varying epsilon (fixed alpha=0.1) ─────────────────────────────────
    print("\n===== VARYING EPSILON (alpha=0.05 fixed) =====\n")
    print(f"{'Algorithm':<15} {'Epsilon':<10} {'Total Steps':<15} {'Steps/Manhattan':<15} {'Time (s)':<15} {'Time/Manthattan (s)':<15}")
    print("-" * 85)

    for epsilon in epsilon_values:
        for name, fn in zip(algo_names, algo_fns):
            all_steps = []
            all_steps_dist = []
            all_times = []
            all_times_dist = []
            for _ in range(N_RUNS):
                _, _, total_steps, comp_time, total_dist = fn(P1, P2,
                                                   episodes=EPISODES,
                                                   alpha=0.05,
                                                   epsilon=epsilon)
                all_steps.append(total_steps)
                all_steps_dist.append(total_steps / total_dist)
                all_times.append(comp_time)
                all_times_dist.append(comp_time / total_dist)
            print(f"{name:<15} {epsilon:<10} {np.mean(all_steps):<15.0f} {np.mean(all_steps_dist):<15.4f} {np.mean(all_times):<15.4f} {np.mean(all_times_dist):<15.6f}")
        print()


    # ── varying epsilon (fixed alpha=0.1) ─────────────────────────────────
    print("\n===== VARYING EPSILON (alpha=0.1 fixed) =====\n")
    print(f"{'Algorithm':<15} {'Epsilon':<10} {'Total Steps':<15} {'Steps/Manhattan':<15} {'Time (s)':<15} {'Time/Manthattan (s)':<15}")
    print("-" * 85)

    for epsilon in epsilon_values:
        for name, fn in zip(algo_names, algo_fns):
            all_steps = []
            all_steps_dist = []
            all_times = []
            all_times_dist = []
            for _ in range(N_RUNS):
                _, _, total_steps, comp_time, total_dist = fn(P1, P2,
                                                   episodes=EPISODES,
                                                   alpha=0.1,
                                                   epsilon=epsilon)
                all_steps.append(total_steps)
                all_steps_dist.append(total_steps / total_dist)
                all_times.append(comp_time)
                all_times_dist.append(comp_time / total_dist)
            print(f"{name:<15} {epsilon:<10} {np.mean(all_steps):<15.0f} {np.mean(all_steps_dist):<15.4f} {np.mean(all_times):<15.4f} {np.mean(all_times_dist):<15.6f}")
        print()


    # ── varying epsilon (fixed alpha=0.1) ─────────────────────────────────
    print("\n===== VARYING EPSILON (alpha=0.2 fixed) =====\n")
    print(f"{'Algorithm':<15} {'Epsilon':<10} {'Total Steps':<15} {'Steps/Manhattan':<15} {'Time (s)':<15} {'Time/Manthattan (s)':<15}")
    print("-" * 85)

    for epsilon in epsilon_values:
        for name, fn in zip(algo_names, algo_fns):
            all_steps = []
            all_steps_dist = []
            all_times = []
            all_times_dist = []
            for _ in range(N_RUNS):
                _, _, total_steps, comp_time, total_dist = fn(P1, P2,
                                                   episodes=EPISODES,
                                                   alpha=0.2,
                                                   epsilon=epsilon)
                all_steps.append(total_steps)
                all_steps_dist.append(total_steps / total_dist)
                all_times.append(comp_time)
                all_times_dist.append(comp_time / total_dist)
            print(f"{name:<15} {epsilon:<10} {np.mean(all_steps):<15.0f} {np.mean(all_steps_dist):<15.4f} {np.mean(all_times):<15.4f} {np.mean(all_times_dist):<15.6f}")
        print()

main()

