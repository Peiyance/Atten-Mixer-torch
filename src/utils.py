import numpy as np
import torch
from torch.utils.data import Dataset

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def data_masks(all_usr_pois, item_tail, maxl=200):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le + 1) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le + 1) for le in us_lens]
    
    return us_pois, us_msks, len_max

class Data(Dataset):
    def __init__(self, data, shuffle=False, graph=None, order=2):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        # print(len(inputs))
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph
        self.order = order

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices
    
    def __len__(self):
        
        return len(self.inputs)

    def __getitem__(self, i):

        return self.get_slice(i)
    
    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, masks, alias_inputs = [], [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        
        max_n_node = np.max(n_node) 
        l_seq = np.sum(mask, axis=1)
        max_l_seq = mask.shape[1]
        max_n_node_aug = max_n_node
        for k in range(self.order-1):
            max_n_node_aug += max_l_seq - 1 - k
        for idx, u_input in enumerate(inputs):
            node = np.array(np.unique(u_input)[1:].tolist() + [0])
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node_aug, max_n_node_aug))
            mask1 = np.zeros(max_n_node_aug)
            for i in np.arange(len(u_input)):
                if u_input[i + 1] == 0:
                    if i == 0:
                        mask1[0] = 1
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                mask1[u] = 1
                mask1[v] = 1
                u_A[u][v] += 1
                
                for t in range(self.order-1):
                    if i == 0:
                        k = max_n_node + t * max_l_seq - sum(list(range(t+1))) + i
                        mask1[k] = 1
                    if i < l_seq[idx] - t - 2:
                        k = max_n_node + t * max_l_seq - sum(list(range(t+1))) + i + 1
                        u_A[u][k] += 1
                        u_A[k-1][k] += 1
                        mask1[k] = 1
                    if i < l_seq[idx] - t - 2:
                        l = np.where(node == u_input[i + t + 2])[0][0]
                        if l is not None and l > 0:
                            u_A[k-1][l] += 1
                            mask1[l] = 1
                
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            masks.append(mask1)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        alias_inputs = torch.tensor(alias_inputs).long()
        A = torch.tensor(A).float()
        items = torch.tensor(items).long()
        mask = torch.tensor(mask).long()
        mask1 = torch.tensor(masks).long()
        targets = torch.tensor(targets).long()
        n_node = torch.tensor(n_node).long()
        return alias_inputs, A, items, mask, mask1, targets, n_node

if __name__ == "__main__":
    import pickle
    data = Data(pickle.load(open("../../data/gowalla/test.txt", 'rb')), shuffle=False, order=3)
    print(data.inputs[1365:1367])
    print(data.get_slice([1365, 1366, 1367]))
