import numpy as np
import torch

import sys
sys.path.append("..")
import Constants as C
import os

def read_interaction_by_trajectory(user_trajectories):

    # for temporal feature
    start_time = time.time()
    print(start_time)

    directory_path = '../data/{dataset}/'.format(dataset=C.DATASET)
    # train_file = 'train.txt'.format(dataset=C.DATASET)
    # train_data = open(directory_path + train_file, 'r').readlines()
    count = 0

    interaction_matrix = torch.zeros((C.USER_NUMBER, C.POI_NUMBER), device='cuda:0')
    POI_matrix = torch.zeros((C.POI_NUMBER, C.POI_NUMBER), device='cuda:0')

    print(interaction_matrix.size())
    print(POI_matrix.size())
    for uid, user_traj in enumerate(user_trajectories):
        for lid in user_traj:
            interaction_matrix[uid][lid] = 1
        count += 1
        if count % 10000 == 0:
            print(count, time.time()-start_time)

    for i in range(C.USER_NUMBER):
        # poi_rev = interaction_matrix[:, i]
        nwhere = torch.where(interaction_matrix[i]==1)[0]
        for j in nwhere:
            POI_matrix[j][nwhere] = 1

    # print(nwhere)
    print(POI_matrix)
    np.save(directory_path + 'poi_matrix.npy', POI_matrix.cpu().numpy())


def create_douban_book():
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

    print(C.DATASET, "#User:", C.USER_NUMBER, '#item', C.POI_NUMBER)
    training_traj = [[] for i in range(C.USER_NUMBER)]
    training_time = [[] for i in range(C.USER_NUMBER)]
    valid_traj = [[] for i in range(C.USER_NUMBER)]
    valid_time = [[] for i in range(C.USER_NUMBER)]
    test_traj = [[] for i in range(C.USER_NUMBER)]
    test_time = [[] for i in range(C.USER_NUMBER)]
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


def read_interaction(train_data=None, directory_path=None):

    if C.DATASET == 'douban-book':
        create_douban_book()
        return

    # for temporal feature
    start_time = time.time()
    print(start_time)
    if directory_path is None:
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
    if train_data is None:
        train_file = 'train.txt'.format(dataset=C.DATASET)
        train_data = open(directory_path + train_file, 'r').readlines()
    count = 0

    interaction_matrix = torch.zeros((C.USER_NUMBER, C.POI_NUMBER), device='cuda:0')
    POI_matrix = torch.zeros((C.POI_NUMBER, C.POI_NUMBER), device='cuda:0')

    print(interaction_matrix.size())
    for eachline in train_data:
        uid, lid, timestamp = eachline.strip().split()
        uid, lid, timestamp = int(uid), int(lid)-1, int(timestamp)
        # print(uid, lid)
        interaction_matrix[uid][lid] = 1
        count += 1
        if count % 10000 == 0:
            print(count, time.time()-start_time)

    for i in range(C.USER_NUMBER):
        # poi_rev = interaction_matrix[:, i]
        nwhere = torch.where(interaction_matrix[i]==1)[0]
        for j in nwhere:
            POI_matrix[j][nwhere] = 1

    # print(nwhere)
    print(POI_matrix)
    np.save(directory_path + 'poi_matrix.npy', POI_matrix.cpu().numpy())


import time

def main():
    # try attention model
    # train_matrix, test_set, place_coords = Foursquare().generate_data()
    read_interaction()


if __name__ == '__main__':
    main()



