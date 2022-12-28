import sys
sys.path.append("..")
import Constants as C
import numpy as np

# Yelp2018  Gowalla  Foursquare  Yelp ml-1M douban-book


def parse(user_traj, data):
    for eachline in data:
        uid, lid, times = eachline.strip().split()
        uid, lid, times = int(uid), int(lid), int(times)
        if uid not in user_traj:
            user_traj[uid] = [lid + 1]
        else:
            user_traj[uid].append(lid + 1)


def select_cold_start_users():
    user_traj = {}
    directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
    parse(user_traj, open(directory_path + 'train.txt', 'r').readlines())
    parse(user_traj, open(directory_path + 'test.txt', 'r').readlines())

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

    # # ================ draw the figure of user distribution ==================
    # import scatter
    # max_ = 0
    # for key in len_map:
    #     if max_ < key: max_ = key
    # x = np.array(range(max_+1))
    # y = np.zeros(max_+1)
    # for key in len_map:
    #     y[key] = len_map[key]
    # scatter.showScatter(x[:100], y[:100], C.DATASET)
    # # ================ draw the figure of user distribution ==================

    # sort users by the lengths of their sequences
    sorted_users = sorted(user_lem_dict.items(), key=lambda x: x[1])

    cold_start_useridx, cold_start_useridx_dict = [], {}
    for (userid, len_) in sorted_users[:int(C.USER_NUMBER * 0.15)]:
        cold_start_useridx.append(userid)
        cold_start_useridx_dict[userid] = 1

    print('Cold-start users list:')
    print(cold_start_useridx)

    cold_start_useridx_npy = np.array(cold_start_useridx)
    np.save(directory_path + 'cold_start_useridx.npy', cold_start_useridx_npy)

    #
    cold_start_test_file = '{dataset}_cold_start_test.txt'.format(dataset=C.DATASET)
    with open(directory_path + cold_start_test_file, 'w')as f:
        for eachline in open(directory_path + 'test.txt', 'r').readlines():
            uid, lid, times = eachline.strip().split()
            uid, lid, times = int(uid), int(lid) + 1, int(times)
            if uid in cold_start_useridx_dict:
                f.write(str(uid)+'\t'+ str(lid)+'\t'+ str(times)+'\r\n')
        f.close()