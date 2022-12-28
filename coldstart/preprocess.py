import sys
sys.path.append("..")
import Constants as C
import numpy as np

# Yelp2018  Gowalla  Foursquare  Yelp ml-1M douban-book

# user_traj = [[] for i in range(C.USER_NUMBER)]
user_traj = {}

directory_path = '../data/{dataset}/'.format(dataset=C.DATASET)

if C.DATASET not in {'douban-book'}:
    train_file = '{dataset}_train.txt'.format(dataset=C.DATASET)
    all_train_data = open(directory_path + train_file, 'r').readlines()
    for eachline in all_train_data:
        uid, lid, times = eachline.strip().split()
        uid, lid, times = int(uid), int(lid), int(times)
        if uid not in user_traj:
            user_traj[uid] = [lid + 1]
        else:
            user_traj[uid].append(lid + 1)

    tune_file = '{dataset}_tune.txt'.format(dataset=C.DATASET)
    all_tune_data = open(directory_path + tune_file, 'r').readlines()
    for eachline in all_tune_data:
        uid, lid, times = eachline.strip().split()
        uid, lid, times = int(uid), int(lid), int(times)
        if uid not in user_traj:
            user_traj[uid] = [lid + 1]
        else:
            user_traj[uid].append(lid + 1)

    test_file = '{dataset}_test.txt'.format(dataset=C.DATASET)
    all_test_data = open(directory_path + test_file, 'r').readlines()
    for eachline in all_test_data:
        uid, lid, times = eachline.strip().split()
        uid, lid, times = int(uid), int(lid), int(times)
        if uid not in user_traj:
            user_traj[uid] = [lid + 1]
        else:
            user_traj[uid].append(lid + 1)
else:
    train_file = 'train.txt'
    all_train_data = open(directory_path + train_file, 'r').readlines()
    for eachline in all_train_data:
        uid, lid, times = eachline.strip().split()
        uid, lid, times = int(uid), int(lid), int(times)
        if uid not in user_traj:
            user_traj[uid] = [lid + 1]
        else:
            user_traj[uid].append(lid + 1)

    test_file = 'test.txt'.format(dataset=C.DATASET)
    all_test_data = open(directory_path + test_file, 'r').readlines()
    for eachline in all_test_data:
        uid, lid, times = eachline.strip().split()
        uid, lid, times = int(uid), int(lid), int(times)
        if uid not in user_traj:
            user_traj[uid] = [lid + 1]
        else:
            user_traj[uid].append(lid + 1)

len_map, len_include_user, user_lem_dict = {}, {}, {}

for uid, user_j in enumerate(user_traj):
    l = len(user_traj[user_j])
    if l not in len_map:
        len_map[l] = 1
        len_include_user[l] = [uid]
    else:
        len_map[l] += 1
        len_include_user[l].append(uid)
    user_lem_dict[uid] = l

import scatter
max_ = 0
for key in len_map:
    if max_ < key:
        max_ = key

x = np.array(range(max_+1))
y = np.zeros(max_+1)
for key in len_map:
    y[key] = len_map[key]

# print(len_map)
print(y[:15])
print(y[80:100])
scatter.showScatter(x[:100], y[:100], C.DATASET)


split_line = int(C.USER_NUMBER * 0.15)

a = sorted(user_lem_dict.items(), key=lambda x: x[1])

cold_start_useridx, cold_start_useridx_dict = [], {}
for (userid, len) in a[:split_line]:
    cold_start_useridx.append(userid)
    cold_start_useridx_dict[userid] = 1

print('Cold-start users list:')
print(cold_start_useridx)


cold_start_useridx_npy = np.array(cold_start_useridx)
print(cold_start_useridx_npy)
np.save(directory_path + 'cold_start_useridx.npy', cold_start_useridx_npy)
#
#
cold_start_test_file = '{dataset}_cold_start_test.txt'.format(dataset=C.DATASET)
with open(directory_path + cold_start_test_file, 'w')as f:
    for eachline in all_test_data:
        uid, lid, times = eachline.strip().split()
        if C.DATASET in {'Gowalla', 'Foursquare', 'Yelp'} :
            uid, lid, times = int(uid), int(lid)+1, int(times)
        else:
            uid, lid, times = int(uid), int(lid), int(times)
        if uid in cold_start_useridx_dict:
            f.write(str(uid)+'\t'+ str(lid)+'\t'+ str(times)+'\r\n')
    f.close()