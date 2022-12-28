# from sklearn.model_selection import train_test_split
import os.path

import scipy.sparse as sparse
import Constants as C
import numpy as np
# import networkx as nx
from preprocess.cal_poi_pairwise import read_interaction_by_trajectory, read_interaction

import torch
import scipy

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


class Dataset(object):
    def __init__(self):
        self.user_num = C.USER_NUMBER
        self.poi_num = C.POI_NUMBER

        self.training_user, self.training_times = self.read_training_data()
        self.tuning_user, self.tuning_times = self.read_tuning_data()
        self.test_user, self.test_times = self.read_test_data()

        self.user_data, self.user_valid = self.getDataByTxt()
        
        # self.training_user, self.training_times, self.tuning_user, self.tuning_times, self.test_user, self.test_times\
        #     = self.read_check_ins()
        #
        # self.user_data, self.user_valid = self.getDataByTxt()

    # def getDataByTxt(self):
    #     user_data = []  # [] for i in range(self.user_num)
    #     user_valid = []  # [] for i in range(self.user_num)
    # 
    #     with open('./data/{dataset}/train.txt'.format(dataset=C.DATASET), 'w') as f:
    #         with open('./data/{dataset}/test.txt'.format(dataset=C.DATASET), 'w') as f:
    #             for i in range(self.user_num):
    #                 user_data.append((i, self.training_user[i], self.training_times[i], self.tuning_user[i],),)
    # 
    #                 valid_input = self.training_user[i].copy()
    #                 valid_input.extend(self.tuning_user[i])
    #                 valid_times = self.training_times[i].copy()
    #                 valid_times.extend(self.tuning_times[i])
    # 
    #                 user_valid.append((i, valid_input, valid_times, self.test_user[i],),)
    # 
    #             return user_data, user_valid

    def getDataByTxt(self):

        user_data = []  # [] for i in range(self.user_num)
        user_valid = []  # [] for i in range(self.user_num)
        # with open('./data/{dataset}/train.txt'.format(dataset=C.DATASET), 'w') as f1:
        #     with open('./data/{dataset}/test.txt'.format(dataset=C.DATASET), 'w') as f2:
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
        if not os.path.exists(directory_path + 'poi_matrix.npy'):
            read_interaction(directory_path=directory_path)

        if C.COLD_START:
            cold_start_useridx_npy = np.load(directory_path + 'cold_start_useridx.npy')
            dict_ = {}
            for i in cold_start_useridx_npy:
                dict_[i] = 1
        for i in range(self.user_num):

            # index = np.array(self.training_user[i]) - 1
            # if len(self.training_user[i]) + len(self.training_user[i]) + len(self.training_user[i])>=5:
            user_data.append((i, self.training_user[i], self.tuning_times[i], self.tuning_user[i],
                                   # self.place_coords[index, :][:, index].toarray(),
                                   # [i + 1 for j in range(len(self.training_user[i]))],
                                   # [i + 1],
                                   ),
                                  )

            if C.COLD_START:
                if i not in dict_:
                    continue
            valid_input = self.training_user[i].copy()
            valid_input.extend(self.tuning_user[i])

            valid_times = self.training_times[i].copy()
            valid_times.extend(self.tuning_times[i])

            # for uid, (traj, time) in enumerate(zip(valid_input, valid_times)):
            #     print(traj, time)
            # for tr, ti in zip(valid_input, valid_times):
            #     f1.write(str(i)+"\t"+str(tr)+"\t"+str(ti)+"\n")
            # index2 = np.array(valid_input) - 1
            user_valid.append((i, valid_input, valid_times, self.test_user[i],
                                    # self.place_coords[index2, :][:, index2].toarray(),
                                    # # [i + 1 for j in range(len(valid_input))],
                                    # [i + 1],
                                    ), )

            # for uid, (traj, time) in enumerate(zip(self.test_user[i], self.test_times[i])):
            # for tr, ti in zip(self.test_user[i], self.test_times[i]):
            #     f2.write(str(i)+"\t"+str(tr)+"\t"+str(ti)+"\n")
    # f1.close()
    # f2.close()
        if C.COLD_START:
            print('#User for Cold-Start', len(user_valid))
        return user_data, user_valid

    def read_check_ins(self):
        directory_path = '../data/{dataset}/'.format(dataset=C.DATASET)
        train_file = 'train.txt'.format(dataset=C.DATASET)
        all_train_data = open(directory_path + train_file, 'r').readlines()

        test_file = 'test.txt'.format(dataset=C.DATASET)
        all_test_data = open(directory_path + test_file, 'r').readlines()

        user_num, poi_num = {}, {}
        for eachline in all_train_data:
            uid, lid, time = eachline.strip().split()
            if uid not in user_num:
                user_num[uid] = len(user_num)
            if lid not in poi_num:
                poi_num[lid] = len(poi_num)

        for eachline in all_test_data:
            uid, lid, time = eachline.strip().split()
            if uid not in user_num:
                user_num[uid] = len(user_num)
            if lid not in poi_num:
                poi_num[lid] = len(poi_num)
        # ml-1M #User: 6038 #item 3492
        print(C.DATASET, "#User:", len(user_num), '#item', len(poi_num))

        print(C.DATASET, "#User:", self.user_num, '#item', self.poi_num)
        training_traj = [[] for i in range(self.user_num)]
        training_time = [[] for i in range(self.user_num)]
        valid_traj = [[] for i in range(self.user_num)]
        valid_time = [[] for i in range(self.user_num)]
        test_traj = [[] for i in range(self.user_num)]
        test_time = [[] for i in range(self.user_num)]
        for eachline in all_train_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = user_num[uid], poi_num[lid], int(float(time))
            try:
                if lid not in training_traj[uid]:
                    training_traj[uid].append(lid)
                    training_time[uid].append(time)
            except Exception as e:
                print(e)

        if not os.path.exists(directory_path + 'poi_matrix.npy'):
            read_interaction_by_trajectory(training_traj)

        for uid, (traj, time) in enumerate(zip(training_traj, training_time)):
            len_ = int(len(traj) * 7/8)
            valid_traj[uid], valid_time[uid] = traj[len_:], time[len_:]
            training_traj[uid], training_time[uid] = traj[:len_], time[:len_]

        for eachline in all_test_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = user_num[uid], poi_num[lid], int(float(time))
            if lid not in training_traj[uid]:
                test_traj[uid].append(lid)
                test_time[uid].append(time)

        train_file = '{dataset}_train.txt'.format(dataset=C.DATASET)
        with open(directory_path + train_file, 'w') as f:
            for user_id, (traj, times) in enumerate(zip(training_traj, training_time)):
                for poi, time in zip(traj, times):
                    f.write(str(user_id)+"\t"+str(poi)+"\t"+str(time)+"\r")
        f.close()
        train_file = '{dataset}_tune.txt'.format(dataset=C.DATASET)
        with open(directory_path + train_file, 'w') as f:
            for user_id, (traj, times) in enumerate(zip(valid_traj, valid_time)):
                for poi, time in zip(traj, times):
                    f.write(str(user_id) + "\t" + str(poi) + "\t" + str(time)+"\r")
        f.close()
        train_file = '{dataset}_test.txt'.format(dataset=C.DATASET)
        with open(directory_path + train_file, 'w') as f:
            for user_id, (traj, times) in enumerate(zip(test_traj, test_time)):
                for poi, time in zip(traj, times):
                    f.write(str(user_id) + "\t" + str(poi) + "\t" + str(time)+"\r")
        f.close()

        return training_traj, training_time, valid_traj, valid_time, test_traj, test_time
    # #
    # 

    def read_training_data(self):
        user_traj = [[] for i in range(self.user_num)]
        user_times = [[] for i in range(self.user_num)]
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
        train_file = '{dataset}_train.txt'.format(dataset=C.DATASET)
        all_train_data = open(directory_path + train_file, 'r').readlines()
        for eachline in all_train_data:
            uid, lid, times = eachline.strip().split()
            uid, lid, times = int(uid), int(lid), int(times)
            try:
                user_traj[uid].append(lid+1)
                user_times[uid].append(times+1)
            except Exception as e:
                print(uid, len(user_traj))
        return user_traj, user_times

    def read_tuning_data(self):
        user_traj = [[] for i in range(self.user_num)]
        user_times = [[] for i in range(self.user_num)]
        # poi_traj = [[] for i in range(self.poi_num)]
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
        tune_file = '{dataset}_tune.txt'.format(dataset=C.DATASET)
        all_tune_data = open(directory_path + tune_file, 'r').readlines()
        for eachline in all_tune_data:
            uid, lid, times = eachline.strip().split()
            uid, lid, times = int(uid), int(lid), int(times)
            user_traj[uid].append(lid+1)
            user_times[uid].append(times+1)
        return user_traj, user_times

    def read_test_data(self):
        user_traj = [[] for i in range(self.user_num)]
        user_times = [[] for i in range(self.user_num)]
        # poi_traj = [[] for i in range(self.poi_num)]
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
        tune_file = '{dataset}_test.txt'.format(dataset=C.DATASET)
        all_test_data = open(directory_path + tune_file, 'r').readlines()
        for eachline in all_test_data:
            uid, lid, times = eachline.strip().split()
            uid, lid, times = int(uid), int(lid), int(times)
            user_traj[uid].append(lid+1)
            user_times[uid].append(times+1)
        return user_traj, user_times

    def read_poi_coos(self):
        sparse_mx = scipy.sparse.load_npz('./data/{dataset}/place_correlation_gamma60.npz'.format(dataset=C.DATASET))
        # sparse_mx = sparse_mx.tocoo().astype(np.float32)
        # indices = torch.from_numpy(
        #     np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        # values = torch.from_numpy(sparse_mx.data)
        # shape = torch.Size(sparse_mx.shape)
        # return torch.sparse.DoubleTensor(indices, values, shape)
        return sparse_mx  # .todense()

    def paddingLong2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        # max_len = max(len(inst) for inst in insts)
        max_len = 700
        batch_seq = np.array([
            inst[:max_len] + [C.PAD] * (max_len - len(inst))
            for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.long)

    def padding2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        # max_len = max(len(inst) for inst in insts)
        max_len = 700
        batch_seq = np.array([
            inst[:max_len] + [C.PAD] * (max_len - len(inst))
            for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.float32)

    # def padding3D(self, insts):  # (16, L)
    #     """ Pad the instance to the max seq length in batch. """
    #     # print(insts)
    #     max_len = max(len(inst) for inst in insts)
    #     inner_dis = []
    #     for i, io in enumerate(insts):
    #         len_ = max_len - len(io)
    #         pad_width1 = ((0, len_), (0, len_))
    #         inner_dis.append(
    #             np.pad(io, pad_width=pad_width1, mode='constant', constant_values=0))  # [:max_len,:max_len]
    #     return torch.tensor(np.array(inner_dis), dtype=torch.float32)

    def user_fn(self, insts):
        """ Collate function, as required by PyTorch. """
        ds = insts
        (idx, event_type, event_time, test_label) = list(zip(*ds))  # list(zip(*ds))
        idx = torch.tensor(idx, device='cuda:0')
        event_type = self.paddingLong2D(event_type)
        event_time = self.paddingLong2D(event_time)
        test_label = self.paddingLong2D(test_label)
        return idx, event_type, event_time, test_label

    def get_user_dl(self, batch_size):
        d = self.user_data
        user_dl = torch.utils.data.DataLoader(
            d,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )
        return user_dl

    def get_user_valid_dl(self, batch_size):
        f = self.user_valid
        user_valid_dl = torch.utils.data.DataLoader(
            f,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )

        return user_valid_dl

# Dataset().read_check_ins()

# if __name__ == '__main__':
#     # user_traj = [[] for i in range(C.USER_NUMBER)]
#     # user_time = [[] for i in range(C.USER_NUMBER)]
#     # training_traj = [[] for i in range(C.USER_NUMBER)]
#     # training_time = [[] for i in range(C.USER_NUMBER)]
#     # test_traj = [[] for i in range(C.USER_NUMBER)]
#     # test_time = [[] for i in range(C.USER_NUMBER)]
#     # directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
#     # train_file = '{dataset}_checkins.txt'.format(dataset=C.DATASET)
#     # all_train_data = open(directory_path + train_file, 'r').readlines()
#     #
#     # for eachline in all_train_data:
#     #     uid, lid, time = eachline.strip().split()
#     #     uid, lid, time = int(uid), int(lid), int(float(time))
#     #     if lid+1 not in user_traj[uid]:
#     #         user_traj[uid].append(lid+1)
#     #         user_time[uid].append(time)
#     # for uid, (traj, time) in enumerate(zip(user_traj, user_time)):
#     #     len_ = int(len(traj)*0.8)
#     #     training_traj[uid], training_time[uid] = traj[:len_], time[:len_]
#     #     test_traj[uid], test_time[uid] = traj[len_:], time[len_:]
#     #
#     # with open('./data/{dataset}/train.txt'.format(dataset=C.DATASET), 'w') as f:
#     #     for uid, (traj, time) in enumerate(zip(training_traj, training_time)):
#     #         for tr, ti in zip(traj, time):
#     #             f.write(str(uid)+"\t"+str(tr)+"\t"+str(ti)+"\n")
#     # with open('./data/{dataset}/test.txt'.format(dataset=C.DATASET), 'w') as f:
#     #     for uid, (traj, time) in enumerate(zip(test_traj, test_time)):
#     #         for tr, ti in zip(traj, time):
#     #             f.write(str(uid)+"\t"+str(tr)+"\t"+str(ti)+"\n")
#     Dataset().read_check_ins()