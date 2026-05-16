# CPS824 W2026
# Assignment 1 - Question 2
# Maarya Siddiqui (501159595)
# Shaghayegh Dehghanisanij (501080180)

import numpy as np
import matplotlib.pyplot as plt

k_rounds = 5000    # agent interacts with environment at least 5000 times
every_round = 100    # every 100 rounds, print the output
n_env = 100       # environment changes 100 times, then interactions are averaged
k_actions = 10              # number of arms
SEED = 42           # Typical seed value


def LRP(alpha, beta):
   
    # to track 
    avg_reward = np.zeros(k_rounds)    
    avg_optimum = np.zeros(k_rounds)   

    for env in range(n_env):
        # seeding
        random_rng = np.random.default_rng(seed=SEED+env)

        #ASSUMPTION1: random values for each arm between (0,1]
        arms = random_rng.random(k_actions)
        optimal_arm = np.argmax(arms)

        #ASSUMPTION2: all arms have same initial probability
        p_t = np.ones(k_actions) / k_actions

        # to track
        total_reward = 0
        optimal_count = 0

        for t in range(1, k_rounds+1):
            # ASSUMPTION3: weighted selection of arms based on their probability 
            chosen_arm = random_rng.choice(k_actions, p=p_t)

            reward = random_rng.binomial(1, arms[chosen_arm])

            # REWARD CASE:
            if reward == 1:  
                p_t[chosen_arm] = p_t[chosen_arm] + alpha * (1 - p_t[chosen_arm])
                for j in range(k_actions):
                    if j != chosen_arm:
                        p_t[j] = (1 - alpha) * p_t[j]
            # PENALTY CASE:
            else: 
                for j in range(k_actions):
                    if j != chosen_arm:
                        p_t[j] = beta / (k_actions - 1) + (1 - beta) * p_t[j]
                p_t[chosen_arm] = (1 - beta) * p_t[chosen_arm]

            # tracking
            total_reward += reward
            if chosen_arm == optimal_arm:
                optimal_count += 1

            avg_reward[t-1] += total_reward/t
            avg_optimum[t-1] += optimal_count/t

    # ASSUMPTION4: normalizing
    avg_reward /= n_env
    avg_optimum /= n_env

    # STATISTICS
    print("AVERAGE REWARD EVERY 100 STEPS")
    for t in range(k_rounds):
        if (t+1) % every_round == 0:
            print(f"Step {t+1} = {avg_reward[t]:.4f}")

    print("\nAVERAGE OPTIMAL ACTION % EVERY 100 STEPS")
    for t in range(k_rounds):
        if (t+1) % every_round == 0:
            print(f"Step {t+1} = {avg_optimum[t]*100:.2f}%")

    return avg_reward, avg_optimum


def LRI(alpha):

    # to track
    avg_reward = np.zeros(k_rounds)    
    avg_optimum = np.zeros(k_rounds)   

    for env in range(n_env):
        # seeding
        random_rng = np.random.default_rng(seed=SEED+env)

        # ASSUMPTION1: random values for each arm between (0,1]
        arms = random_rng.random(k_actions)
        optimal_arm = np.argmax(arms)

        # ASSUMPTION2: all arms have same initial probability
        p_t = np.ones(k_actions) / k_actions

        # to track
        total_reward = 0
        optimal_count = 0

        for t in range(1, k_rounds+1):
            # ASSUMPTION3: weighted selection of arms based on their probability
            chosen_arm = random_rng.choice(k_actions, p=p_t)

            reward = random_rng.binomial(1, arms[chosen_arm])

            # REWARD CASE:
            if reward == 1:  
                
                p_t[chosen_arm] = p_t[chosen_arm] + alpha * (1 - p_t[chosen_arm])
                for j in range(k_actions):
                    if j != chosen_arm:
                        p_t[j] = (1 - alpha) * p_t[j]

            # PENALTY CASE:
            # do nothing 

            total_reward += reward
            if chosen_arm == optimal_arm:
                optimal_count += 1

            avg_reward[t-1] += total_reward/t
            avg_optimum[t-1] += optimal_count/t

    # ASSUMPTION4: normalizing
    avg_reward /= n_env
    avg_optimum /= n_env

    # STATISTICS
    print("AVERAGE REWARD EVERY 100 STEPS")
    for t in range(k_rounds):
        if (t+1) % every_round == 0:
            print(f"Step {t+1} = {avg_reward[t]:.4f}")

    print("\nAVERAGE OPTIMAL ACTION % EVERY 100 STEPS")
    for t in range(k_rounds):
        if (t+1) % every_round == 0:
            print(f"Step {t+1} = {avg_optimum[t]*100:.2f}%")

    return avg_reward, avg_optimum

###################### copied from q1.py ###########################

def ucb(c):

    avg_reward = np.zeros(k_rounds)    # array to track average reward over time
    avg_optimum = np.zeros(k_rounds)   # array to track average number of times optimum was selected

    for env in range(n_env):
        # Begin by seeding for each new environment
        random_rng = np.random.default_rng(seed=SEED+env)

        # Initialize true values q*(a) for a=0-9, where each is a random float between (0 and 1]
        arms = random_rng.random(k_actions)
        optimal_arm = np.argmax(arms)

        # Initialize Nt(a) = 0 for all arms (a=0-9)
        n_t = np.zeros(k_actions, dtype=int)

        # Initialize expected values for each arm Qt(a) for a=0-9
        expected_vals_Q = np.zeros(k_actions, dtype=float)

        total_reward = 0
        optimal_count = 0

        # Play each arm once initially
        for a in range(k_actions):
            reward = random_rng.binomial(1, arms[a])
            n_t[a] += 1
            expected_vals_Q[a] = reward
            total_reward += reward
            if a == optimal_arm:
                optimal_count += 1

            avg_reward[a] += total_reward / (a+1)
            avg_optimum[a] += optimal_count / (a+1)

        # Agent interacts with environment 5000 times
        for t in range(k_actions+1, k_rounds+1):
            # 1. ESTIMATE UCB OF EACH ARM
            bonus = c * np.sqrt(np.log(t)/n_t)
            apply_ucb = expected_vals_Q + bonus

            # 2. SELECT ARM WITH HIGHEST UCB
            chosen_arm = np.argmax(apply_ucb)

            # 3. UPDATE
            reward = random_rng.binomial(1, arms[chosen_arm])
            n_t[chosen_arm] += 1
            expected_vals_Q[chosen_arm] += (reward - expected_vals_Q[chosen_arm])/(n_t[chosen_arm])

            # Update stats
            total_reward += reward
            if chosen_arm == optimal_arm:
                optimal_count += 1

            avg_reward[t-1] += total_reward/t
            avg_optimum[t-1] += optimal_count/t

    avg_reward /= n_env
    avg_optimum /= n_env

    # PRINT STATISTICS
    print("AVERAGE REWARD EVERY 100 STEPS")
    for t in range(k_rounds):
        if (t+1) % every_round == 0:
            print(f"Step {t+1} = {avg_reward[t]:.4f}")

    print("\nAVERAGE OPTIMAL ACTION % EVERY 100 STEPS")
    for t in range(k_rounds):
        if (t+1) % every_round == 0:
            print(f"Step {t+1} = {avg_optimum[t]*100:.2f}%")

    return avg_reward, avg_optimum

####################################################################


def main():
    """Implementing and comparing LR-I, LR-P, and UCB algorithms"""
    
    # # Test different parameter values for LR-I
    # print("\n" + "="*70)
    # print("TESTING LR-I WITH DIFFERENT ALPHA VALUES")
    # print("="*70)
    # alpha_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    # LRI_results = []
    # LRI_opt_results = []
    
    # for alpha in alpha_vals:
    #     print(f"\n---- LR-I with α = {alpha} ----")
    #     avg_reward, avg_optimum = LRI(alpha)
    #     LRI_results.append(avg_reward)
    #     LRI_opt_results.append(avg_optimum*100)

    # # Test different parameter values for LR-P
    # print("\n" + "="*70)
    # print("TESTING LR-P WITH DIFFERENT ALPHA AND BETA VALUES")
    # print("="*70)
    # param_pairs = [(0.1, 0.05), (0.1, 0.1), (0.1, 0.5), (0.5, 0.1), (0.5, 0.5), (0.5, 0.9), (0.9, 0.5), (0.9, 0.9)]
    # LRP_results = []
    # LRP_opt_results = []
    
    # for alpha, beta in param_pairs:
    #     print(f"\n---- LR-P with α = {alpha}, β = {beta} ----")
    #     avg_reward, avg_optimum = LRP(alpha, beta)
    #     LRP_results.append(avg_reward)
    #     LRP_opt_results.append(avg_optimum*100)

    #     # Test UCB with selected c values
    #     print("\n" + "="*70)
    #     print("TESTING UCB WITH DIFFERENT C VALUES")
    #     print("="*70)
    #     c_vals = [0.5, 1.0, 2.0]
    #     ucb_results = []
    #     ucb_opt_results = []
        
    #     for c in c_vals:
    #         print(f"\n---- UCB with c = {c} ----")
    #         avg_reward, avg_optimum = ucb(c)
    #         ucb_results.append(avg_reward)
    #         ucb_opt_results.append(avg_optimum*100)

    # print("\n" + "="*70)
    # print("GENERATING COMPARISON GRAPHS")
    # print("="*70)

    x_axis = np.arange(k_rounds)

    # GRAPH 1: LR-I with different alpha values
    # plt.figure(1, figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # for i, alpha in enumerate(alpha_vals):
    #     plt.plot(x_axis, LRI_results[i], label=f"α = {alpha}")
    # plt.grid(which="both", axis="both")
    # plt.xlabel("Steps")
    # plt.ylabel("Average Reward")
    # plt.title("LR-I: Average Reward Over 100 Environments")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # for i, alpha in enumerate(alpha_vals):
    #     plt.plot(x_axis, LRI_opt_results[i], label=f"α = {alpha}")
    # plt.grid(which="both", axis="both")
    # plt.xlabel("Steps")
    # plt.ylabel("Optimal Action (%)")
    # plt.title("LR-I: Optimal Action % Over 100 Environments")
    # plt.legend()
    # plt.tight_layout()

    # GRAPH 2: LR-P with different alpha/beta values
    # plt.figure(2, figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # for i, (alpha, beta) in enumerate(param_pairs):
    #     plt.plot(x_axis, LRP_results[i], label=f"α={alpha}, β={beta}")
    # plt.grid(which="both", axis="both")
    # plt.xlabel("Steps")
    # plt.ylabel("Average Reward")
    # plt.title("LR-P: Average Reward Over 100 Environments")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # for i, (alpha, beta) in enumerate(param_pairs):
    #     plt.plot(x_axis, LRP_opt_results[i], label=f"α={alpha}, β={beta}")
    # plt.grid(which="both", axis="both")
    # plt.xlabel("Steps")
    # plt.ylabel("Optimal Action (%)")
    # plt.title("LR-P: Optimal Action % Over 100 Environments")
    # plt.legend()
    # plt.tight_layout()

    # GRAPH 3: Comparison of best performing configurations
    plt.figure(3, figsize=(12, 5))
    
    # Select best performer from each algorithm based on final performance
    # best_LRI_idx = np.argmax([LRI_opt_results[i][-1] for i in range(len(alpha_vals))])
    # best_LRP_idx = np.argmax([LRP_opt_results[i][-1] for i in range(len(param_pairs))])

    # Based on previous runs, we know the best parameters:
    avg_reward_lrp, avg_optimum_lrp = LRP(0.5, 0.1)
    avg_reward_lri, avg_optimum_lri = LRI(0.05)

    avg_reward_ucb, avg_optimum_ucb = ucb(0.5)
    rounds = np.arange(1, k_rounds + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x_axis, avg_reward_lri, label=f"LR-I (α= 0.05)", linewidth=2)
    plt.plot(x_axis, avg_reward_lrp, label=f"LR-P (α= 0.5, β= 0.1)", linewidth=2)
    plt.plot(rounds, avg_reward_ucb, label=f"UCB (c=0.5)", linewidth=2)
    plt.grid(which="both", axis="both")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Algorithm Comparison: Average Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, avg_optimum_lri*100, label=f"LR-I (α= 0.05)", linewidth=2)
    plt.plot(x_axis, avg_optimum_lrp*100, label=f"LR-P (α= 0.5, β= 0.1)", linewidth=2)
    plt.plot(rounds, avg_optimum_ucb*100, label=f"UCB (c=0.5)", linewidth=2)
    plt.grid(which="both", axis="both")
    plt.xlabel("Steps")
    plt.ylabel("Optimal Action (%)")
    plt.title("Algorithm Comparison: Optimal Action %")
    plt.legend()
    plt.tight_layout()

    plt.show()

    # Print summary comparison
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY (at step 5000)")
    print("="*70)

    print(f"\nBest LR-I (α= 0.05):")
    print(f"  Average Reward: {avg_reward_lri[-1]:.4f}")
    print(f"  Optimal Action %: {avg_optimum_lri[-1]:.2f}%")

    print(f"\nBest LR-P (α= 0.5, β= 0.1):")
    print(f"  Average Reward: {avg_reward_lrp[-1]:.4f}")
    print(f"  Optimal Action %: {avg_optimum_lrp[-1]:.2f}%")

    print(f"\nUCB (c=0.5):")
    print(f"  Average Reward: {avg_reward_ucb[-1]:.4f}")
    print(f"  Optimal Action %: {avg_optimum_ucb[-1]*100:.2f}%")
    print("="*70)



if __name__ == "__main__":
    main()