import numpy as np

import torch as T
import torch.nn.functional as F

# constants
ALL_KERNELS = ['cosine', 'l1', 'l2']
ALL_POLICIES = ['1NN']

class DND:
    """The differentiable neural dictionary (DND) class. This enables episodic
    recall in a neural network.

    notes:
    - a memory is a row vector

    Parameters
    ----------
    dict_len : int
        the maximial len of the dictionary
    memory_dim : int
        the dim or len of memory i, we assume memory_i is a row vector
    kernel : str
        the metric for memory search

    Attributes
    ----------
    encoding_off : bool
        if True, stop forming memories
    retrieval_off : type
        if True, stop retrieving memories
    reset_memory : func;
        if called, clear the dictionary
    check_config : func
        check the class config

    """

    def __init__(self, dict_len, key_dim, memory_dim, kernel='l2'):
        # params
        self.dict_len = dict_len
        self.kernel = kernel
        self.key_dim = key_dim
        self.memory_dim = memory_dim
        # dynamic state
        self.encoding_off = False
        self.retrieval_off = False
        # allocate space for memories
        self.reset_memory()
        # check everything
        self.check_config()

    def reset_memory(self):
        self.pointer = 0
        self.overflow = False 
        self.keys = T.empty(self.dict_len, self.key_dim)
        self.vals = T.empty(self.dict_len, self.memory_dim)

    def check_config(self):
        assert self.dict_len > 0
        assert self.kernel in ALL_KERNELS

    def export_memories(self, path):
        limit = self.dict_len if self.overflow else self.pointer
        np.save(path+"_keys.npy", self.keys[:limit].numpy())
        np.save(path+"_vals.npy", self.vals[:limit].numpy())

    def load_memories(self, path):
        self.keys = T.from_numpy(np.load(path+"_keys.npy"))
        self.vals = T.from_numpy(np.load(path+"_vals.npy"))
        self.pointer = len(self.keys)

    def inject_memories(self, input_keys, input_vals):
        """Inject pre-defined keys and values

        Parameters
        ----------
        input_keys : list
            a list of memory keys
        input_vals : list
            a list of memory content
        """
        assert len(input_keys) == len(input_vals)
        for k, v in zip(input_keys, input_vals):
            self.save_memory(k, v)

    def save_memory(self, 
        memory_key, 
        memory_val, 
        replace_similar=False,
        threshold=0
    ):
        """Save an episodic memory to the dictionary

        Parameters
        ----------
        memory_key : a row vector
            a DND key, used to for memory search
        memory_val : a row vector
            a DND value, representing the memory content
        """
        if self.encoding_off:
            return
        # add new memory to the the dictionary
        # get data is necessary for gradient reason

        replaced = False 
        if replace_similar and (self.pointer > 0 or self.overflow):
            similarities = compute_similarities(memory_key, self.keys[:self.pointer], self.kernel)
            closest_idx = T.argmax(similarities)
            if similarities[closest_idx] > threshold:
                self.keys[closest_idx] = T.squeeze(memory_key.data)
                self.vals[closest_idx] = T.squeeze(memory_val.data)
                replaced = True 

        if not replace_similar or not replaced:        
            self.keys[self.pointer] = T.squeeze(memory_key.data) 
            self.vals[self.pointer] = T.squeeze(memory_val.data) 
            self.pointer += 1
            if self.pointer >= self.dict_len:
                self.pointer = 0
                self.overflow = True 


    def get_memory(self, query_key, threshold=-1):
        """Perform a 1-NN search over dnd

        Parameters
        ----------
        query_key : a row vector
            a DND key, used to for memory search

        Returns
        -------
        a row vector
            a DND value, representing the memory content

        """
        # if no memory, return the zero vector
        if (self.pointer == 0 and not self.overflow) or self.retrieval_off:
            return _empty_memory(self.memory_dim)
        # compute similarity(query, memory_i ), for all i
        similarities = compute_similarities(query_key, self.keys[:self.pointer], self.kernel)
        # get the best-match memory
        best_memory_val = self._get_memory(similarities, threshold)
        return best_memory_val

    def _get_memory(self, similarities, threshold, policy='1NN'):
        """get the episodic memory according to some policy
        e.g. if the policy is 1nn, return the best matching memory
        e.g. the policy can be based on the rational model

        Parameters
        ----------
        similarities : a vector of len #memories
            the similarity between query vs. key_i, for all i
        policy : str
            the retrieval policy

        Returns
        -------
        a row vector
            a DND value, representing the memory content

        """
        best_memory_val = None
        if policy is '1NN':
            best_memory_id = T.argmax(similarities)
            if threshold <= 0 or similarities[best_memory_id] > threshold:
                best_memory_val = self.vals[best_memory_id].unsqueeze(0)
            else:
                best_memory_val = _empty_memory(self.memory_dim)
        else:
            raise ValueError(f'unrecog recall policy: {policy}')
        return best_memory_val


"""helpers"""

def compute_similarities(query_key, key_list, metric):
    """Compute the similarity between query vs. key_i for all i
        i.e. compute q M, w/ q: 1 x key_dim, M: key_dim x #keys

    Parameters
    ----------
    query_key : a vector
        Description of parameter `query_key`.
    key_list : list
        Description of parameter `key_list`.
    metric : str
        Description of parameter `metric`.

    Returns
    -------
    a row vector w/ len #memories
        the similarity between query vs. key_i, for all i

    """
    # reshape query to 1 x key_dim
    q = query_key.data.view(1, -1)
    # compute similarities
    if metric == 'cosine':
        similarities = F.cosine_similarity(q.float(), key_list.float())
    elif metric == 'l1':
        similarities = - F.pairwise_distance(q, key_list, p=1)
    elif metric == 'l2':
        similarities = - F.pairwise_distance(q, key_list, p=2)
    else:
        raise ValueError(f'unrecog metric: {metric}')
    return similarities


def _empty_memory(memory_dim):
    """Get a empty memory, assuming the memory is a row vector
    """
    return T.zeros(1, memory_dim)
