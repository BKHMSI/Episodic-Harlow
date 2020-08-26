import os
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

def plot_test_trial_1():
    base_path = "ckpts"
    run_title = "H_LSTM_FINAL_EP8"
    n_seeds = 1
    n_episodes = 500
    rt_gates = np.array([1.00, 0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.50, 0.20, 0])

    all_rewards = []
    for seed in range(n_seeds):
        for rt_val in rt_gates:
            path = os.path.join(base_path, run_title+f"_{seed+1}", f"rewards_{rt_val:.2f}.npy")
            rewards = np.load(path)
            all_rewards += [rewards[:n_episodes]]
    all_rewards = np.stack(all_rewards)

    q = all_rewards[:, :n_episodes, 0]
    performance = q.mean(axis=1)

    plt.plot(rt_gates, performance*100, 'o-')

    plt.xlabel("R-Gate Threshold")
    plt.ylabel("Performance (Trial 1)%")
    plt.title("Episodic Harlow 1D (R-Gate Analysis)")
    plt.show()

def plot_test():
    base_path = "ckpts"
    run_title = "H_LSTM_FINAL_EP8"
    n_seeds = 1
    n_episodes = 500

    all_rewards = []
    for seed in range(n_seeds):
        path = os.path.join(base_path, run_title+f"_{seed+1}", f"rewards_1.npy")
        if os.path.exists(path):
            rewards = np.load(path)
            all_rewards += [rewards[:n_episodes]]

    all_rewards = np.stack(all_rewards)

    quantiles = [0, 500]
    n_quantiles = len(quantiles) - 1
    n_trials = all_rewards.shape[2]

    for i in range(n_quantiles):
        line = []
        stds = []
        for j in range(n_trials):
            q = all_rewards[:, quantiles[i]:quantiles[i+1],j]
            performance = q.mean(axis=1)
            line += [performance.mean()*100]
            stds += [(performance.std()*100) / np.sqrt(all_rewards.shape[0])]
        plt.errorbar(np.arange(1,7), line, fmt='o-', yerr=stds)

    plt.plot([1,6], [50,50], '--')
    plt.xlabel("Trial")
    plt.ylabel("Performance (%)")
    # labels = ["Random", "1st", "2nd", "3rd", "4th", "Ep 1st", "Ep 2nd"]
    # legend_title = "Training Phase"
    # plt.legend(labels, title=legend_title)
    plt.title("Episodic Harlow 1D (Test)")
    plt.show()

def plot_all():
    base_path = "ckpts"
    run_titles = ["H_LSTM_FINAL_EP", "H_LSTM_FINAL_EP3"]
    n_seeds = 4
    n_workers = 8
    n_episodes = 3000

    all_rewards = []
    for seed in range(n_seeds):
        run_rewards = []
        for worker in range(n_workers):
            path = os.path.join(base_path, run_titles[0]+f"_{seed+1}", f"rewards_{worker}.npy")
            if os.path.exists(path):
                rewards = np.load(path)
                run_rewards += [rewards[:n_episodes]]
        run_rewards = np.array(run_rewards).mean(axis=0)

        path = os.path.join(base_path, run_titles[1]+f"_{seed+1}", f"rewards_0.npy")
        rewards = np.load(path)
        all_rewards += [np.array(list(run_rewards)+list(rewards))]

    all_rewards = np.stack(all_rewards)

    quantiles = [0, 500, 750, 1000, 3500, 6000, 8000]
    n_quantiles = len(quantiles) - 1
    n_trials = all_rewards.shape[2]

    for i in range(n_quantiles):
        line = []
        stds = []
        for j in range(n_trials):
            q = all_rewards[:, quantiles[i]:quantiles[i+1],j]
            performance = q.mean(axis=1)
            line += [performance.mean()*100]
            stds += [(performance.std()*100) / np.sqrt(all_rewards.shape[0])]
        plt.errorbar(np.arange(1,7), line, fmt='o-', yerr=stds)

    plt.plot([1,6], [50,50], '--')
    plt.xlabel("Trial")
    plt.ylabel("Performance (%)")
    labels = ["Random", "1st", "2nd", "3rd", "4th", "Ep 1st", "Ep 2nd"]
    legend_title = "Training Phase"
    plt.legend(labels, title=legend_title)
    plt.title("Episodic Harlow 1D")
    plt.show()

def plot_trial_1():

    base_path = "ckpts"
    run_titles = ["H_LSTM_FINAL_EP0", "H_LSTM_FINAL_EP2", "H_LSTM_FINAL_EP5", "H_LSTM_FINAL_EP4", "H_LSTM_FINAL_EP3"]
    labels = ["N/A", ">0", ">2", ">4", ">6"]
    n_seeds = 4
    n_episodes = 5000
    n_quantiles = 10
    quantiles = list(map(int, np.linspace(0, n_episodes, n_quantiles+1)))

    all_rewards = []
    for run_title in run_titles:
        run_reward = []
        for seed in range(n_seeds):
            path = os.path.join(base_path, run_title+f"_{seed+1}", f"rewards_0.npy")
            rewards = np.load(path)
            run_reward += [rewards[:n_episodes]]
        all_rewards += [run_reward]

    all_rewards = np.stack(all_rewards)

    data = []
    for k in range(len(run_titles)):
        for i in range(n_quantiles):
            q = all_rewards[k, :, quantiles[i]:quantiles[i+1], 0]
            performance = q.mean(axis=1)
            data += [{'Training Quantile': i+1, 'Performance': p, 'Threshold': labels[k]} for p in performance]
    
    df = pd.DataFrame(data)
    sns.lineplot(x="Training Quantile", y="Performance", hue="Threshold", data=df)
    plt.show()

if __name__ == "__main__":

    # plot_all()

    # plot_trial_1()

    # plot_test()

    plot_test_trial_1()



