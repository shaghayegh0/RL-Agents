# CPS824 W2026
# Assignment 1 - Question 1
# Maarya Siddiqui (501159595)
# Shaghayegh Dehghanisanij (501080180)

import numpy as np
import matplotlib.pyplot as plt

NUM_STEPS = 5000    # agent interacts with environment at least 5000 times
NUM_ROUNDS = 100    # every 100 rounds, print the output
NUM_ENV = 100       # environment changes 100 times, then interactions are averaged for UCB algorithm
K = 10              # number of arms
SEED = 42           # Typical seed value

def ucb(c):
    """Implementation of the UCB algorithm
        Param: c-value
        Returns: nparray of average reward and nparray average optimum over 100 environments, 5000 steps each"""

    avg_reward = np.zeros(NUM_STEPS)    # array to track average reward over time
    avg_optimum = np.zeros(NUM_STEPS)   # array to track average number of times optimum was selected

    for env in range(NUM_ENV):
        # Begin by seeding for each new environment
        random_rng = np.random.default_rng(seed=SEED+env)

        # Initialize true values q*(a) for a=0-9, where each is a random float between (0 and 1]
        # np.random.rand by default chooses random floats between (0,1]
        arms = random_rng.random(K)
        optimal_arm = np.argmax(arms)

        # Initialize Nt(a) = 0 for all arms (a=0-9)
        # We will play each arm first, and then begin UCB
        n_t = np.zeros(K,dtype=int)

        # Initialize expected values for each arm Qt(a) for a=0-9
        expected_vals_Q = np.zeros(K, dtype=float)

        total_reward = 0
        optimal_count = 0

        # Based on textbook implementation, the UCB algorithm begings by playing each arm once and observing the expected reward
        for a in range(len(arms)):
            reward = random_rng.binomial(1, arms[a]) # play arm a
            n_t[a] += 1                              # increment number of times are is played
            expected_vals_Q[a] = reward              # initial observed reward is current expected reward
            total_reward += reward                   # keep track of total reward
            if a == optimal_arm:
                optimal_count += 1                   # keep track of whether optimal arm was selected

            avg_reward[a] += total_reward / (a+1)
            avg_optimum[a] += optimal_count / (a+1)
            

        # Agent interacts with environment 5000 times
        # Begin K+1 since we have already performed 10 initial interactions above
        for t in range(K+1, NUM_STEPS+1):
            # 1. ESTIMATE UCB OF EACH ARM --> expected_value + expected_bonus
            bonus = c * np.sqrt(np.log(t)/n_t)      # compute bonus for each arm
            apply_ucb = expected_vals_Q + bonus     # add bonus to expected value for each arm --> by default numpy keeps track of indexing

            # 2. SELECT ARM WITH HIGHEST UCB
            chosen_arm = np.argmax(apply_ucb)       # select arm with highest expected upper confidence bound

            # 3. UPDATE
            # Play arm a (with highest UCB)
            # and observe Bernoulli reward: 1 with prob q*(a) and 0 otherwise
            reward = random_rng.binomial(1, arms[chosen_arm])

            # Increment the number of times we've interacted with this arm
            n_t[chosen_arm]+=1

            # Update expected reward for selected arm using formula Qt+1 = Qt + (Rt - Qt)/Nt(a)
            expected_vals_Q[chosen_arm] += (reward - expected_vals_Q[chosen_arm])/(n_t[chosen_arm])

            # Update stats
            total_reward += reward
            if chosen_arm == optimal_arm:
                optimal_count += 1

            avg_reward[t-1] += total_reward/t   # for time=t in env=e, update the average reward, we will avg each of these np array elements once all envs are complete
            avg_optimum[t-1] += optimal_count/t # for time=t in env=e, update the average times the optimal arm was selected, we will avg each of these np array elements once all envs are complete

    avg_reward /= NUM_ENV
    avg_optimum /= NUM_ENV

    # PRINT STATISTICS
    print("AVERAGE REWARD EVERY 100 STEPS")
    for t in range(NUM_STEPS):
        if (t+1)%NUM_ROUNDS == 0:
            print("Step "+str(t+1)+" = "+str(avg_reward[t]))

    print("\nAVERAGE OPTIMAL ACTION % EVERY 100 STEPS")
    for t in range(NUM_STEPS):
        if (t+1)%NUM_ROUNDS == 0:
            print(f"Step {t+1} = {avg_optimum[t]*100:.2f}%")

    return avg_reward, avg_optimum

def main():
    """Implementing the UCB algorithm against 5 unique C values, averaged across 100 environments with 5000 steps each"""
    c_vals = [0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0]
    avg_reward_results = []
    avg_opt_results = []
    for c in c_vals:
        print("\n---- BEGIN TEST WITH C = "+str(c)+" ----")
        avg_reward, avg_optimum = ucb(c)
        avg_reward_results.append(avg_reward)
        avg_opt_results.append(avg_optimum*100)

    print("\n--- GENERATING GRAPHS USING MATPLOTLIB ---")
    # GENREATE GRAPH TO DISPLAY AVERAGE REWARD OVER 5000 STEPS NORMALIZED OVER 100 ENVIRONMENTS
    x_axis = np.arange(len(avg_reward_results[0]))
    plt.figure(1)
    for i in range(len(c_vals)):
        plt.plot(x_axis, avg_reward_results[i], label=f"c = {c_vals[i]}")
    plt.grid(which="both", axis="both")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over 100 Environments")
    plt.legend()

    # GENERATE GRAPH TO DISPLAY AVERAGE OPTIMAL ACTION OVER 5000 STEPS NORMALIZED OVER 100 ENVIRONMENTS
    plt.figure(2)
    for i in range(len(c_vals)):
        plt.plot(x_axis, avg_opt_results[i], label=f"c = {c_vals[i]}")
    plt.grid(which="both", axis="both")
    plt.xlabel("Steps")
    plt.ylabel("Average Optimal Action (%)")
    plt.title("Average Optimal Action % Over 100 Environments")
    plt.legend()

    plt.show()
        
main()


