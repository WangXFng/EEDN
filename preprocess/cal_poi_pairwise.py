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


def read_interaction(train_data=None, directory_path=None):

    # for temporal feature
    start_time = time.time()
    # print(start_time)
    if directory_path is None:
        directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
    if train_data is None:
        train_data = open(directory_path + '{dataset}_train.txt'.format(dataset=C.DATASET), 'r').readlines()
        train_data.extend(open(directory_path + '{dataset}_tune.txt'.format(dataset=C.DATASET), 'r').readlines())
    count = 0

    interaction_matrix = torch.zeros((C.USER_NUMBER, C.POI_NUMBER), device='cuda:0')
    POI_matrix = torch.zeros((C.POI_NUMBER, C.POI_NUMBER), device='cuda:0')

    print(interaction_matrix.size())
    for eachline in train_data:
        uid, lid, timestamp = eachline.strip().split()
        uid, lid, timestamp = int(uid), int(lid), int(timestamp)
        # print(uid, lid)
        interaction_matrix[uid][lid] = 1
        count += 1
        if count % 500000 == 0:
            print(count, time.time()-start_time)

    for i in range(C.USER_NUMBER):
        # poi_rev = interaction_matrix[:, i]
        nwhere = torch.where(interaction_matrix[i]==1)[0]
        for j in nwhere:
            POI_matrix[j][nwhere] = 1

    # print(nwhere)
    # print(POI_matrix)
    np.save(directory_path + 'poi_matrix.npy', POI_matrix.cpu().numpy())


import time

def main():
    # try attention model
    # train_matrix, test_set, place_coords = Foursquare().generate_data()
    read_interaction()


if __name__ == '__main__':
    main()



