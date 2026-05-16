# CPS824 W2026
# Assignment 2 q3 - Value Iteration Algorithm
# Maarya Siddiqui (501159595)
# Shaghayegh Dehghanisanij (501080180)

import numpy as np

GAMMA = 0.95    # Discount factor
THETA = 0.001   # Convergence factor
GRID_SIZE = 4
NUM_STATES = GRID_SIZE*GRID_SIZE

v = np.zeros(NUM_STATES)    # value function                          
TERMINAL_STATE = 0      # terminal state position @ state 0 

# ACTIONS
actions = {
        "u":(-1,0),             # move up
        "d":(1,0),              # move down
        "r":(0,1),              # move right
        "l":(0,-1)              # move left
    }

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
    # Otherwise return 
    if (0 <= new_r < GRID_SIZE) and (0 <= new_col < GRID_SIZE):
        return get_state(new_r, new_col)
    else:
        return get_state(row, col)

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


def value_iteration(v, actions, adjacent, r, p1, p2, GRID_SIZE, NUM_STATES, GAMMA, THETA):
    '''Implementation of the value iteration algorithm
    Returns 3 element tuple:
        - v: array of state values
        - policy: the optimal policy
        - k: number of times until convergence'''
    
    k = 0                       # counter to keep track of number of iterations of while loop until convergence
    policy = [""]*NUM_STATES    # initialize policy to display at the end

    while True:
        k+=1
        delta = 0
        updated_V = v.copy()

        # SWEEP THROUGH ALL STATES, S
        for s in range(NUM_STATES):
            if s == TERMINAL_STATE:
                continue

            # Define action value function for state s
            q_a = {}

            # SWEEP THROUGH ALL ACTIONS
            for a in actions.keys():

                # Calculate all possible transitions from s -> s' for all actions a
                transitions = get_transition_probs(s, a, adjacent, p1, p2, GRID_SIZE)

                expectated_val = 0

                # SWEEP THROUGH ALL NEXT STATES, S' and accumulate expected reward
                # Given the transition probabilities for P(s' | s, a) calculate the expected value using Bellman Equation
                # r[a] is the reward of performing a given action
                for next_state, prob_of_transition in transitions.items():
                    expectated_val += prob_of_transition * (r[a] + GAMMA*v[next_state])

                # Store all actions and corresponding expected_values
                q_a[a] = expectated_val

            # After sweeping through all possible actions to perform in state s, find the action that yields the max return
            best_action, best_value = max(q_a.items(), key=lambda item: item[1])
            updated_V[s] = best_value       # best return
            policy[s] = best_action         # store best_action

            # Determine delta (max to then compare with theta for convergence)
            delta = max(delta, abs(updated_V[s] - v[s]))

        # Update the value function after sweeping through all states
        v = updated_V

        # Check for convergence
        if delta < THETA:
            break

    return v, policy, k

def display_values(v):
    '''Displays state values as a grid'''
    value_grid = v.reshape(GRID_SIZE, GRID_SIZE)
    line = "+" + ("-"*6 + "+") * GRID_SIZE
    print(line)
    for row in value_grid:
        print("|", end="")
        for v in row:
            print(f"{v:^{6}.2f}|", end="")
        print()
        print(line)

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

def main():
    '''Mainline for the program'''
    # 1. Move to desired state --> p1
    # 2. Remain in same state --> p2
    # 3. Go to desired states' adjacent cell --> 1-p1-p2/2
    # 4. Go off grid & end up in same state --> p1+p2 by (1) and (2)
    # 5. Go off grid & move to current states' adjacent state --> 1-p1-p2/2 by (3)

    # Get user input for probabilities and rewards
    p1 = float(input("Enter p1: "))
    p2 = float(input("Enter p2: "))

    r_up = float(input("Enter r_up: "))
    r_down = float(input("Enter r_down: "))
    r_r = float(input("Enter r_right: "))
    r_l = float(input("Enter r_left: "))

    # REWARDS
    r = {
        "u":r_up,
        "d":r_down,
        "r":r_r,
        "l":r_l
    }

    # ADJACENT CELL MAPPING
    adjacent = {
        "u":["l", "r"],     # adjacent states are left & right
        "d":["l", "r"],     # adjacent states are left & right
        "r":["u", "d"],     # adjacent states are up & down
        "l":["u", "d"]      # adjacent states are up & down
    }

    # Perform value iteration algorithm
    value_function, optimal_policy, k = value_iteration(v, actions, adjacent, r, p1, p2, GRID_SIZE, NUM_STATES, GAMMA, THETA)

    print("\n===== VALUE ITERATION RESULTS =====\n")

    print("ITERATIONS UNTIL CONVERGENCE: "+str(k))

    print("\nSTATE VALUES")
    display_values(value_function)

    print("\nOPTIMAL POLICY")
    display_policy(optimal_policy)

main()
