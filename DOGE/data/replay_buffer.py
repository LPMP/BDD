import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.keys = []
        self.values = []

    def reset(self):
        self.keys = []
        self.values = []
 
    def add(self, key, value):
        self.keys.append(key)
        self.values.append(value)
        
    def sample(self, index):
        if len(self.keys) == 0:
            return None
        index_distance = index - np.array(self.keys)
        index_distance = np.where(index_distance >= 0, index_distance, np.inf)
        best_index = np.where(index_distance == index_distance.min())[0][-1].item()
        if index_distance[best_index] == np.inf:
            return None
        best_key = self.keys[best_index]
        print(f'Sampled {best_key} instead of {index}.')
        return self.values[best_index], best_key