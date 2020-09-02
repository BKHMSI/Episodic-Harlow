import os
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from tqdm import tqdm 
from tensorboard.backend.event_processing import event_accumulator

def read_data(load_dir, tag="perf/avg_reward_100", length=2500):

    events = os.listdir(load_dir)
    for event in events:
        path = os.path.join(load_dir, event)
        ea = event_accumulator.EventAccumulator(path, size_guidance={ 
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.SCALARS: length,
                event_accumulator.HISTOGRAMS: 0,
        })
        
        ea.Reload()
        tags = ea.Tags()

        if tag not in tags["scalars"]: continue

        if len(ea.Scalars(tag)) <= length:
            return np.array([s.value for s in ea.Scalars(tag)[:length]])

    return None 

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
    r_probs = [1, 0]
    n_seeds = 1
    n_episodes = 500

    all_rewards = []
    for r_prob in r_probs:
        for seed in range(n_seeds):
            path = os.path.join(base_path, run_title+f"_{seed+1}", f"rewards_{r_prob:.2f}.npy")
            if os.path.exists(path):
                rewards = np.load(path)
                all_rewards += [rewards[:n_episodes]]

    all_rewards = np.stack(all_rewards)

    quantiles = [0, 500]
    n_quantiles = len(quantiles) - 1
    n_trials = all_rewards.shape[2]

    for i in range(n_quantiles):
        for k in range(len(r_probs)):
            line = []
            for j in range(n_trials):
                q = all_rewards[k, quantiles[i]:quantiles[i+1],j]
                line += [q.mean()*100]
                # line += [performance.mean()*100]
                # stds += [(performance.std()*100) / np.sqrt(all_rewards.shape[0])]
            # plt.errorbar(np.arange(1,7), line, fmt='o-', yerr=stds)
            plt.plot(np.arange(1,7), line, 'o-')

    plt.plot([1,6], [50,50], '--')
    plt.xlabel("Trial")
    plt.ylabel("Performance (%)")
    labels = ["w/o memory", "w/ memory"]
    # legend_title = "Training Phase"
    plt.legend(labels, title="Legend")
    plt.title("Episodic Harlow 1D (Weights Frozen)")
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
    # labels = ["w/o memory", "w/ memory"]
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
        # for i in range(n_quantiles):
        for ep in range(all_rewards.shape[2]):
            q = all_rewards[k, :, ep-100:ep, 0]
            performance = q.mean(axis=1)
            data += [{'Episode': ep, 'Performance': p, 'Threshold': labels[k]} for p in performance]
    
    df = pd.DataFrame(data)
    sns.lineplot(x="Episode", y="Performance", hue="Threshold", data=df)
    plt.show()

def plot_rewards_curve():

    load_path = "logs_final/H_LSTM_FINAL_EP/H_LSTM_FINAL_EP"
    load_path2 = "logs_final/H_LSTM_FINAL_EP2/H_LSTM_FINAL_EP2"
    n_seeds = 4
    n_workers = 8
    data = np.zeros((n_seeds, 8000))
    for seed_idx in tqdm(range(n_seeds)):
        workers = []
        for worker in range(n_workers):
            data_ev = read_data(load_dir=load_path+f"_{seed_idx+1}_{worker}", length=5000)
            if data_ev is None:
                raise ValueError
            workers += [data_ev]

        data[seed_idx][:3000] = np.array(workers).mean(axis=0)[:3000] 

    
        data_ev = read_data(load_dir=load_path2+f"_{seed_idx+1}_0", length=5000)
        data[seed_idx][3000:] = data_ev

    table = []
    for seed_idx in range(n_seeds):
        for i in range(8000):
            table += [{'Episode': i, 'Reward': data[seed_idx][i], "RNN Type": "LSTM"}]
    
    df = pd.DataFrame(table)
    sns.lineplot(x="Episode", y="Reward", data=df, ci="sd")
    plt.show()

if __name__ == "__main__":

    # plot_all()

    # plot_trial_1()

    # plot_test()

    # plot_test_trial_1()

    plot_rewards_curve()

