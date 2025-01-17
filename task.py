import os
import sys 
import imageio
import numpy as np 
import matplotlib.pyplot as plt 

"""helpers"""

def _binary2int(binary):
    return (binary * 2**np.arange(binary.shape[0]-1, -1, -1)).sum()

def _int2binary(decimal, length=10):
    return np.array([int(x) for x in format(decimal, f'#0{length+2}b')[2:]])

class HarlowEpisodic_1D:
    """A 1D episodic variant of the Harlow Task.
    """  
    def __init__(self, 
        config,
        episodic = True,
        verbose = False, 
        start_episode = 0,
        visualize = False,
        save_path = None,
        save_interval = None 
    ):

        '''environment constants'''
        self.max_length = config["max-length"]
        self.n_trials   = config["num-trials"]
        self.n_actions  = config["num-actions"]
        self.n_objects  = config["num-objects"]
        self.n_episodes = config["num-episodes"]
        self.n_stages   = config["num-stages"]
        self.state_len  = config["state-len"] # size of state
        self.obs_length = config["obs-len"]  # size of receptive field
        self.obj_offset = config["obj-offset"]
        self.fix_reward = config["fix-reward"]
        self.obj_reward = config["obj-reward"]
        self.map_action = config["map-action"]

        self.mode = config["mode"]
        if self.mode not in ["easy", "hard"]:
            raise NotImplementedError(f"Mode: {self.mode}")

        self.time_step  = 0
        self.ctx_length = int(np.ceil(np.log2(self.n_objects)))
        self.max_reward = self.n_trials * (self.fix_reward+self.obj_reward)

        self._generate_contexts()

        self.memory = []
        self.episode_num = start_episode
        self.episode_counter = 0
        self.episodic = episodic
        self.verbose = verbose 
        self.visualize = visualize
        self.center = self.state_len // 2
        self.reward_counter = np.zeros((self.n_episodes, self.n_trials))

        if self.visualize:
            self.frames = []
            self._create_palette()
            self.save_path = save_path
            self.save_interval = save_interval

    @property
    def current(self):
        return self.state[self.center]

    @property
    def stage(self):
        return 1
        # return (self.n_stages*self.episode_counter) // self.n_episodes

    def _generate_contexts(self):
        self.context_pool = np.arange(self.n_objects)
        np.random.shuffle(self.context_pool)

    def generate_uncue(self):
        return np.ones(self.ctx_length) * -1

    def _place_objects(self):
        self.state[self.center-self.obj_offset] = self.obj_1
        self.state[self.center+self.obj_offset] = self.obj_2 
        self.state[self.center] = 0

        swap = np.random.rand() < 0.5
        if swap:
            self.state[self.center-self.obj_offset] = self.obj_2 
            self.state[self.center+self.obj_offset] = self.obj_1

    def _place_fixation(self):
        self.state = np.zeros(self.state_len)
        if self.pointer > self.center:
            self.state[self.center - self.obj_offset] = 1
        else:
            self.state[self.center + self.obj_offset] = 1

    def observation(self):
        offset = (self.state_len - self.obs_length) // 2
        return self.state[offset:-offset] 

    def step(self, action):
        
        self.time_step += 1

        reward = 0
        self.state = np.roll(self.state, self.map_action[action])
        self.pointer -= self.map_action[action]

        if self.pointer >= self.state_len:
            self.pointer = 0 
        elif self.pointer < 0:
            self.pointer = self.state_len - 1

        if self.current != 0 and self.visualize:
            self._add_frames(self.observation())

        if self.current == 1:
            reward = self.fix_reward
            self._place_objects()
        elif self.current == self.obj_1:
            reward = self.obj_reward if self.reward_obj else -self.obj_reward
            if self.reward_obj:
                self.reward_counter[self.episode_counter-1][self.trial_num] = 1
            self.trial_num += 1
            self._place_fixation()
        elif self.current == self.obj_2:
            reward = self.obj_reward if not self.reward_obj else -self.obj_reward
            if not self.reward_obj:
                self.reward_counter[self.episode_counter-1][self.trial_num] = 1
            self.trial_num += 1
            self._place_fixation()

        obs = self.observation()
        if self.visualize:
            self._add_frames(obs)

        if self.verbose:
            print(f"Observation: {obs}")
            print(f"Reward: {reward} | Pointer: {self.pointer}")

        done = self.trial_num >= self.n_trials or self.time_step >= self.max_length
        return obs, reward, done, self.time_step 

    def reset(self):
        self.trial_num = 0
        self.time_step = 0
        self.episode_num += 1
        self.episode_counter += 1
        self.pointer = self.center

        if self.visualize and len(self.frames) > 0 and self.episode_counter % self.save_interval == 0:
            self._save_frames()
        self.frames = []

        # initialize state
        self.state = np.zeros(self.state_len)
        self.state[self.center] = 1

        shift = np.random.randint(-self.obj_offset, self.obj_offset)
        if shift == 0: shift = 1
        
        self.state = np.roll(self.state, shift)
        self.pointer -= shift 

        obs = self.observation()

        if self.visualize:
            self._add_frames(obs)

        if self.verbose:
            print(f"Observation: {obs}")
            print(f"Pointer: {self.pointer}")

        # episode objects
        if self.stage == 0 or not self.episodic:
            self.obj_1, self.obj_2 = np.random.randint(low=2, high=self.n_objects+2, size=2)
            self.reward_obj  = np.random.rand() < 0.5
        elif self.mode == "easy":
            self.obj_1, self.obj_2, self.reward_obj = self.memory[np.random.randint(len(self.memory))]
        elif self.mode == "hard":
            self.obj_1 = self.memory[np.random.randint(len(self.memory))]
            self.reward_obj = True 
            self.obj_2 = np.random.choice(list(np.arange(2, self.obj_1)) 
                                    + list(np.arange(self.obj_1+1,self.n_objects+2)))
        
        ctx_idx = self.context_pool[self.obj_1-2] if self.reward_obj else self.context_pool[self.obj_2-2]
        self.context = _int2binary(ctx_idx, self.ctx_length)  

        self.obj_1 /= self.n_objects
        self.obj_2 /= self.n_objects

        return obs 

    def _visualize_obs(self, obs):
        size = 20
        background = np.ones((size*5, size*(obs.shape[0]), 3), dtype=np.uint8) * 255
        bar = np.zeros((size, size*(obs.shape[0]), 3), dtype=np.uint8)
        for i, cell in enumerate(obs):
            if cell == 1:
                # draw fixation cross
                bar[0:9,i*size:i*size+9] = [255, 0, 0]
                bar[0:9,i*size+11:i*size+20] = [255, 0, 0]
                bar[11:20,i*size+11:i*size+20] = [255, 0, 0]
                bar[11:20,i*size:i*size+9] = [255, 0, 0]
            elif cell > 0:
                idx = int(cell*self.n_objects)
                bar[:,i*size:i*size+size] = self.palette[idx]

        background[size*2:size*3] = bar
        return background 

    def _create_palette(self):
        self.palette = []
        for _ in range(self.n_objects):
            color = list(np.random.choice(range(256), size=3))
            if color not in self.palette:
                self.palette += [color]

    def _add_frames(self, obs):
        bar = self._visualize_obs(obs)
        for _ in range(10):
            self.frames += [bar]

    def _save_frames(self):
        filepath = self.save_path.format(epi=self.episode_num)
        imageio.mimsave(filepath, self.frames)

    def save_memory(self):
        if self.mode == "easy":
            self.memory += [(int(self.obj_1*self.n_objects), int(self.obj_2*self.n_objects), self.reward_obj)]
        elif self.mode == "hard":
            self.memory += [int(self.obj_1*self.n_objects) if self.reward_obj else int(self.obj_2*self.n_objects)]
    
    def export_memories(self, path):
        np.save(os.path.join(path, "memories.npy"), np.array(self.memory))
        np.save(os.path.join(path, "context_pool.npy"), np.array(self.context_pool))

    def load_memories(self, path):
        self.memory = np.load(os.path.join(path, "memories.npy"))
        self.context_pool = np.load(os.path.join(path, "context_pool.npy"))


if __name__ == "__main__":
    
    config = {}
    env = HarlowEpisodic_1D(config, verbose=True)

    while True:
        action = int(input("Left (1) or Right (2): "))
        
        if action <= 0:
            break 

        env.step(action)
