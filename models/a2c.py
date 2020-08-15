import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.dnd import DND
from models.ep_gru import EpGRU
from models.ep_lstm import EpLSTM

class A2C_DND(nn.Module):

    def __init__(self,
            rnn_type, 
            input_dim, 
            hidden_dim, 
            num_actions,
            dict_key_dim,
            dict_len,
            kernel='l2', 
            device="cpu",
    ):
        super(A2C_DND, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device 

        self.encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # long-term memory 
        self.dnd = DND(dict_len, dict_key_dim, hidden_dim, kernel)

        # short-term memory
        rnn = EpLSTM if self.rnn_type == "lstm" else EpGRU
        self.ep_rnn = rnn(
            input_size=128+num_actions+1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False
        )
        
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # reset lstm parameters
        self.ep_rnn.reset_parameters()
        # reset dnd 
        self.reset_memory()
        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, p_input, state, cue):

        feats = self.encoder(obs)
        x_t = T.cat((feats, *p_input), dim=-1)

        m_t = self.dnd.get_memory(cue).to(self.device)
    
        _, rnn_state = self.ep_rnn((x_t.unsqueeze(1), m_t.unsqueeze(1)), state)

        h_t = rnn_state[0] if self.rnn_type == "lstm" else rnn_state

        action_logits = self.actor(h_t)
        value_estimate = self.critic(h_t)

        return action_logits, value_estimate, rnn_state, feats

    def get_init_states(self):
        h0 = T.zeros(1, 1, self.ep_rnn.hidden_size).float().to(self.device)
        c0 = T.zeros(1, 1, self.ep_rnn.hidden_size).float().to(self.device)
        return (h0, c0) if self.rnn_type == "lstm" else h0

    def turn_off_encoding(self):
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()

    def save_memory(self, mem_key, mem_val):
        self.dnd.save_memory(mem_key, mem_val, replace_similar=True, threshold=0.99)

    def retrieve_memory(self, query_key):
        return self.dnd.get_memory(query_key)

    def get_all_mems(self):
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V
