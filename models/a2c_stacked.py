import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.dnd import DND
from models.ep_lstm import EpLSTM

class A3C_DND_StackedLSTM(nn.Module):

    def __init__(self, 
            input_dim, 
            hidden_dim, 
            num_actions,
            dict_key_dim,
            dict_len,
            kernel='l2', 
            bias=True,
            device="cpu",
    ):
        super(A3C_DND_StackedLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.device = device 

        feat_dim = 128

        self.encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
            nn.ReLU(),
        )

        # long-term memory 
        self.dnd_1 = DND(dict_len, dict_key_dim, hidden_dim, kernel)
        self.dnd_2 = DND(dict_len, dict_key_dim, hidden_dim // 2, kernel)

        # short-term memory
        self.ep_lstm_1 = EpLSTM(
            input_size=feat_dim+1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False
        )

        self.ep_lstm_2 = EpLSTM(
            input_size=feat_dim+num_actions+hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=False 
        )
        
        self.actor  = nn.Linear(hidden_dim // 2, num_actions)
        self.critic = nn.Linear(hidden_dim // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # reset lstm parameters
        self.ep_lstm_1.reset_parameters()
        self.ep_lstm_2.reset_parameters()
        # reset dnd 
        self.reset_memory()
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, state_1, state_2, cue=None):

        p_action, p_reward = p_input
        
        feats = self.encoder(obs)
        x_t1 = T.cat((feats, p_reward), dim=-1).unsqueeze(1)

        m_t1 = self.dnd_1.get_memory(cue).unsqueeze(1).to(self.device)
        m_t2 = self.dnd_2.get_memory(cue).unsqueeze(1).to(self.device)
    
        _, (h_t1, c_t1) = self.ep_lstm_1((x_t1, m_t1), state_1)

        x_t2 = T.cat((h_t1.squeeze(0), feats, p_action), dim=-1).unsqueeze(1)

        _, (h_t2, c_t2) = self.ep_lstm_2((x_t2, m_t2), state_2) 

        action_logits  = self.actor(h_t2)
        value_estimate = self.critic(h_t2)

        return action_logits, value_estimate, (h_t1, c_t1), (h_t2, c_t2)

    def get_init_states(self, layer=1):
        hidden_size = self.ep_lstm_1.hidden_size if layer == 1 else self.ep_lstm_2.hidden_size
        h0 = T.zeros(1, 1, hidden_size).float().to(self.device)
        c0 = T.zeros(1, 1, hidden_size).float().to(self.device)
        return (h0, c0)

    def turn_off_encoding(self):
        self.dnd_1.encoding_off = True
        self.dnd_2.encoding_off = True

    def turn_on_encoding(self):
        self.dnd_1.encoding_off = False
        self.dnd_2.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd_1.retrieval_off = True
        self.dnd_2.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd_1.retrieval_off = False
        self.dnd_2.retrieval_off = False

    def reset_memory(self):
        self.dnd_1.reset_memory()
        self.dnd_2.reset_memory()

    def save_memory(self, mem_key, mem_val, layer=1):
        dnd = self.dnd_1 if layer == 1 else self.dnd_2
        dnd.save_memory(mem_key, mem_val, replace_similar=True, threshold=0.99)
     
    def retrieve_memory(self, query_key, layer=1):
        dnd = self.dnd_1 if layer == 1 else self.dnd_2
        return dnd.get_memory(query_key)

    def get_all_mems(self):
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V
