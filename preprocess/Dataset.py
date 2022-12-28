import os
import torch
import numpy as np
import Constants as C
from coldstart.preprocess import select_cold_start_users
if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


class Dataset(object):
    def __init__(self):
        self.user_num = C.USER_NUMBER
        self.poi_num = C.POI_NUMBER
        self.directory_path = './data/{dataset}/'.format(dataset=C.DATASET)

        self.training_user, self.training_times = self.read_training_data()
        self.tuning_user, self.tuning_times = self.read_tuning_data()
        self.test_user, self.test_times = self.read_test_data()

        self.user_data, self.user_valid = self.read_data()

    def parse(self, data):
        user_traj, user_times = [[] for i in range(self.user_num)], [[] for i in range(self.user_num)]
        for eachline in data:
            uid, lid, times = eachline.strip().split()
            uid, lid, times = int(uid), int(lid), int(times)
            try:
                user_traj[uid].append(lid+1)
                user_times[uid].append(times+1)
            except Exception as e:
                print(uid, len(user_traj))
        return user_traj, user_times

    def read_data(self):
        user_data, user_valid = [], []
        if C.COLD_START:
            if not os.path.exists(self.directory_path + 'cold_start_useridx.npy'):
                print('Generating cold-start users ...')
                select_cold_start_users()
            cold_start_useridx_npy = np.load(self.directory_path + 'cold_start_useridx.npy')
            dict_ = {}  # to check if it is a cold start user with time complexity of O(n)
            for i in cold_start_useridx_npy:
                dict_[i] = 1

        for i in range(self.user_num):
            user_data.append((i, self.training_user[i], self.tuning_times[i], self.tuning_user[i], ), )
            if C.COLD_START:
                if i not in dict_:
                    continue
            valid_input = self.training_user[i].copy()
            valid_input.extend(self.tuning_user[i])
            valid_times = self.training_times[i].copy()
            valid_times.extend(self.tuning_times[i])
            user_valid.append((i, valid_input, valid_times, self.test_user[i], ), )

        if C.COLD_START:
            print('#User for Cold-Start', len(user_valid))
        return user_data, user_valid

    def read_training_data(self):
        train_file = '{dataset}_train.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + train_file, 'r').readlines())

    def read_tuning_data(self):
        tune_file = '{dataset}_tune.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + tune_file, 'r').readlines())

    def read_test_data(self):
        test_file = '{dataset}_test.txt'.format(dataset=C.DATASET)
        return self.parse(open(self.directory_path + test_file, 'r').readlines())

    def paddingLong2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        max_len = 700  # max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst[:max_len] + [C.PAD] * (max_len - len(inst))
            for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.long)

    def padding2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        max_len = 700  # max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst[:max_len] + [C.PAD] * (max_len - len(inst))
            for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.float32)

    def user_fn(self, insts):
        """ Collate function, as required by PyTorch. """
        (idx, event_type, event_time, test_label) = list(zip(*insts))
        idx = torch.tensor(idx, device='cuda:0')
        event_type = self.paddingLong2D(event_type)
        event_time = self.paddingLong2D(event_time)
        test_label = self.paddingLong2D(test_label)
        return idx, event_type, event_time, test_label

    def get_user_dl(self, batch_size):
        user_dl = torch.utils.data.DataLoader(
            self.user_data,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )
        return user_dl

    def get_user_valid_dl(self, batch_size):
        user_valid_dl = torch.utils.data.DataLoader(
            self.user_valid,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )
        return user_valid_dl

