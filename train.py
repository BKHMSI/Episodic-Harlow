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
from models.a2c_stacked import A3C_DND_StackedLSTM

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), 
                                shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train_episodic(config, 
    shared_model=None, 
    optimizer=None, 
    rank=0
):

    T.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    T.random.manual_seed(config["seed"] + rank)
    device = config["device"]

    writer = SummaryWriter(log_dir=os.path.join(config["log-path"], config["run-title"] + f"_{rank}"))
    save_path = os.path.join(config["save-path"], config["run-title"], config["run-title"]+"_{epi:04d}")
    save_interval = config["save-interval"]

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
    agent.train()

    update_counter = 0
    if shared_model is None:
        optimizer = T.optim.RMSprop(agent.parameters(), config["agent"]["lr"])

        if config["resume"]:
            filepath = os.path.join(
                config["save-path"], 
                config["load-title"], 
                f"{config['load-title']}_{config['start-episode']:04d}.pt"
            )
            print(f"> Loading Checkpoint {filepath}")
            model_data = T.load(filepath)
            update_counter = model_data["update_counter"]
            agent.load_state_dict(model_data["state_dict"])

    ### hyper-parameters ###
    gamma = config["agent"]["gamma"]
    gae_lambda = config["agent"]["gae-lambda"]
    val_coeff = config["agent"]["value-loss-weight"]
    entropy_coeff = config["agent"]["entropy-weight"]
    n_step_update = config["agent"]["n-step-update"]
    rnn_type = config["agent"]["rnn-type"]
    min_reward = config["task"]["min-reward"]

    done = True 
    state = env.reset()
    cue = T.tensor(env.generate_uncue())
    p_action, p_reward = [0]*config["task"]["num-actions"], 0

    print('='*50)
    print(f"Starting Worker {config['run-title']}_{rank}")
    print('='*50)

    total_rewards = []
    episode_reward = 0

    if config["dnd"]:
        agent.turn_off_retrieval()
        agent.turn_on_encoding()
    else:
        agent.turn_off_encoding()
        agent.turn_off_retrieval()

    while True: 

        if env.stage > 0 and config["dnd"]:
            agent.turn_on_retrieval()

        if shared_model is not None:
            agent.load_state_dict(shared_model.state_dict())

        if done:
            rnn_state = agent.get_init_states()
        elif rnn_type == "lstm":
            rnn_state = rnn_state[0].detach(), rnn_state[1].detach()
        else:
            rnn_state = rnn_state.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for _ in range(n_step_update):

            if -1 in cue:
                agent.turn_off_retrieval()
            else:
                agent.turn_on_retrieval()

            logit, value, rnn_state, _ = agent(
                T.tensor([state]).float().to(device), (
                T.tensor([p_action]).float().to(device), 
                T.tensor([[p_reward]]).float().to(device)), 
                rnn_state, cue
            )

            logit = logit.squeeze(0)

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies += [entropy]
            action = prob.multinomial(num_samples=1).detach()

            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(int(action))

            episode_reward += reward

            if done and env.stage == 0 and episode_reward > min_reward:
                mt = rnn_state[1] if rnn_type == "lstm" else rnn_state
                agent.save_memory(T.tensor(env.context), mt)
                env.save_memory()

            cue = env.context if reward == env.fix_reward and env.stage > 0 else env.generate_uncue()
            cue = T.tensor(cue)

            p_action = np.eye(env.n_actions)[int(action)]
            p_reward = reward
            
            log_probs += [log_prob]
            values += [value]
            rewards += [reward]

            if done:
                state = env.reset()
                total_rewards += [episode_reward]
                avg_reward_100 = np.array(total_rewards[-100:]).mean()
                writer.add_scalar("perf/reward_t", episode_reward, env.episode_num)
                writer.add_scalar("perf/avg_reward_100", avg_reward_100, env.episode_num)

                # r_mean = agent.get_r_mean()
                # for i, rm in enumerate(r_mean):
                #     writer.add_scalar(f"rs-gate/{i}", rm, env.episode_num)
                # agent.reset_r_mean()

                episode_reward = 0
                if env.episode_num % save_interval == 0:
                    save_model = shared_model or agent
                    T.save({
                        "state_dict": save_model.state_dict(),
                        "avg_reward_100": avg_reward_100,
                        "update_counter": update_counter,
                    }, save_path.format(epi=env.episode_num) + ".pt")

                break 

        R = T.zeros(1, 1).to(device)
        if not done:
            _, value, _, _ = agent(
                T.tensor([state]).float().to(device), (
                T.tensor([p_action]).float().to(device), 
                T.tensor([[p_reward]]).float().to(device)), 
                rnn_state, cue
            )
            R = value.detach()
        
        values += [R]

        policy_loss = 0
        value_loss = 0
        gae = T.zeros(1, 1).to(device)

        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            
            # Generalized Advantage Estimation
            delta_t = rewards[i] + gamma * values[i + 1] - values[i]
            gae = gae * gamma * gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - entropy_coeff * entropies[i]

        loss = policy_loss + val_coeff * value_loss
        optimizer.zero_grad()
        loss.backward()
        if shared_model is not None:
            ensure_shared_grads(agent, shared_model)
        optimizer.step()

        update_counter += 1

        writer.add_scalar("losses/total_loss", loss.item(), update_counter)

        if env.episode_num > env.n_episodes:
            env.export_memories(os.path.dirname(save_path))
            agent.dnd.export_memories(os.path.join(os.path.dirname(save_path), "mem"))
            np.save(os.path.join(os.path.dirname(save_path), f"rewards_{rank}.npy"), env.reward_counter)
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

    n_seeds = 1
    base_seed = config["seed"]
    base_run_title = config["run-title"]
    for seed_idx in range(1, n_seeds+1):  
        config["seed"] = base_seed * seed_idx
        config["run-title"] = base_run_title + f"_{seed_idx}"
        config['start-episode'] = 3000
        config['load-title'] = "H_LSTM_FINAL_EP" + f"_{seed_idx}"

        exp_path = os.path.join(config["save-path"], config["run-title"])
        if not os.path.isdir(exp_path): 
            os.mkdir(exp_path)
        
        out_path = os.path.join(exp_path, os.path.basename(args.config))
        with open(out_path, 'w') as fout:
            yaml.dump(config, fout)

        train_episodic(config)

    