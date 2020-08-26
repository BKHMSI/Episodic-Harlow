import os
import yaml
import pickle
import argparse
import numpy as np

import torch as T
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm 
from datetime import datetime 
from collections import namedtuple

from task import HarlowEpisodic_1D
from models.a2c import A2C_DND


def test(config):

    T.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    T.random.manual_seed(config["seed"])
    device = config["device"]

    writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"]))
    save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")

    env = HarlowEpisodic_1D(config["task"], episodic=config["dnd"])
    agent = A2C_DND(
        config["agent"]["rnn-type"],
        config["task"]["input-dim"],
        config["agent"]["mem-units"], 
        config["task"]["num-actions"],
        config["agent"]["dict-key-dim"],
        config["agent"]["dict-len"],
        config["agent"]["dict-kernel"],
        device=config["device"]
    )

    agent.to(device)
    agent.eval()

    filepath = os.path.join(
        config["save-path"], 
        config["load-title"], 
        f"{config['load-title']}_{config['start-episode']:04d}.pt"
    )
    print(f"> Loading Checkpoint {filepath}")
    model_data = T.load(filepath)
    agent.load_state_dict(model_data["state_dict"])

    agent.dnd.load_memories(os.path.join(os.path.dirname(save_path), "mem"))
    env.load_memories(os.path.dirname(save_path))

    ### hyper-parameters ###
    rnn_type = config["agent"]["rnn-type"]

    done = True 
    state = env.reset()
    cue = T.tensor(env.generate_uncue())
    p_action, p_reward = [0]*config["task"]["num-actions"], 0

    print('='*50)
    print(f"Starting Worker {config['run-title']}")
    print('='*50)

    total_rewards = []
    episode_reward = 0

    if config["dnd"]:
        agent.turn_on_retrieval()
        agent.turn_on_encoding()
    else:
        agent.turn_off_encoding()
        agent.turn_off_retrieval()

    progress = tqdm(range(env.n_episodes))

    r_gates_closed = []

    while True:

        if done:
            rnn_state = agent.get_init_states()
        elif rnn_type == "lstm":
            rnn_state = rnn_state[0].detach(), rnn_state[1].detach()
        else:
            rnn_state = rnn_state.detach()

        if -1 in cue:
            agent.turn_off_retrieval()
        else:
            agent.turn_on_retrieval()

        logit, _, rnn_state, _ = agent(
            T.tensor([state]).float().to(device), (
            T.tensor([p_action]).float().to(device), 
            T.tensor([[p_reward]]).float().to(device)), 
            rnn_state, cue
        )

        if -1 not in cue:
            agent.update_r_gates_closed_at_all_time()
            r_gates_closed += [agent.count_r_gate_closed()]

        logit = logit.squeeze(0)

        prob = F.softmax(logit, dim=-1)
        action = prob.multinomial(num_samples=1).detach()

        state, reward, done, _ = env.step(int(action))

        episode_reward += reward

        cue = env.context if reward == env.fix_reward and env.stage > 0 else env.generate_uncue()
        cue = T.tensor(cue)

        p_action = np.eye(env.n_actions)[int(action)]
        p_reward = reward
        
        if done:
            state = env.reset()
            total_rewards += [episode_reward]
            avg_reward_100 = np.array(total_rewards[-100:]).mean()
            writer.add_scalar("perf/reward_t", episode_reward, env.episode_num)
            writer.add_scalar("perf/avg_reward_100", avg_reward_100, env.episode_num)
            episode_reward = 0

            progress.update()

        if env.episode_num > env.n_episodes:
            print(f"Average of R-Gates Closed: {100*np.array(r_gates_closed).mean()}%")
            print(f"Percentage of R-Gates Closed at all times: {(100*len(agent.r_gates_closed))/agent.hidden_dim}%")
            np.save(os.path.join(os.path.dirname(save_path), f"rewards_0.20.npy"), env.reward_counter)
            break  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    T.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    T.random.manual_seed(config["seed"])

    n_runs = 1
    rt_prob = 0
    base_run_title = config["run-title"]
    base_load_title = config["load-title"]
    for i in range(1, n_runs+1):
        config["run-title"]  = f"{base_run_title}_{i}"
        config["load-title"] = f"{base_load_title}_{i}"

        exp_path = os.path.join(config["save-path"], config["run-title"])
        if not os.path.isdir(exp_path): 
            os.mkdir(exp_path)
        
        out_path = os.path.join(exp_path, os.path.basename(args.config))
        with open(out_path, 'w') as fout:
            yaml.dump(config, fout)

        test(config)

    