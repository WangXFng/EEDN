import argparse
import numpy as np
import time
import metric

import torch
import torch.optim as optim

if torch.cuda.is_available():
    pass
else:
    pass

import optuna

import Constants as C
import Utils

from preprocess.Dataset import Dataset as dataset
from Models import Model
from tqdm import tqdm


def train_epoch(model, user_dl, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    pre, rec, map_, ndcg = [[] for i in range(4)], [[] for i in range(4)], [[] for i in range(4)], [[] for i in range(4)]
    for batch in tqdm(user_dl, mininterval=2, desc='  - (Training)   ', leave=False):
        optimizer.zero_grad()

        """ prepare data """
        user_idx, event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

        """ forward """
        prediction, users_embeddings = model(user_idx, event_type)

        """ compute metric """
        metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

        """ backward """
        loss = Utils.type_loss(prediction, event_type, event_time, test_label, opt.smooth)

        loss.backward(retain_graph=True)
        """ update parameters """
        optimizer.step()

    pre_np, rec_np, map_np, ndcg_np = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    for i in range(4):
        pre_np[i], rec_np[i], map_np[i], ndcg_np[i] = np.mean(pre[i]), np.mean(rec[i]), np.mean(map_[i]), np.mean(ndcg[i])

    return pre_np, rec_np, map_np, ndcg_np


def eval_epoch(model, user_valid_dl, opt, valid_user_embeddings):
    """ Epoch operation in evaluation phase. """

    model.eval()
    pre, rec, map_, ndcg = [[] for i in range(4)], [[] for i in range(4)], [[] for i in range(4)], [[] for i in range(4)]
    with torch.no_grad():
        for batch in tqdm(user_valid_dl, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare test data """
            user_idx, event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

            """ forward """
            prediction, users_embeddings = model(user_idx, event_type)  # X = (UY+Z) ^ T

            valid_user_embeddings[user_idx] = users_embeddings

            """ compute metric """
            metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

    pre_np, rec_np, map_np, ndcg_np = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    for i in range(4):
        pre_np[i], rec_np[i], map_np[i], ndcg_np[i] = np.mean(pre[i]), np.mean(rec[i]), np.mean(map_[i]), np.mean(ndcg[i])

    return pre_np, rec_np, map_np, ndcg_np


def train(model, data, optimizer, scheduler, opt):
    """ Start training. """

    valid_precision_max = 0.0

    (user_valid_dl, user_dl) = data
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        valid_user_embeddings = torch.zeros((C.USER_NUMBER, opt.d_model), device='cuda:0')

        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        start = time.time()  # loglikelihood: {ll: 8.5f},
        pre_np, rec_np, map_np, ndcg_np = train_epoch(model, user_dl, optimizer, opt)
        print('\r(Training)  P@k:{pre},    R@k:{rec}, \n'
              '(Training)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,
                      pre=pre_np, rec=rec_np, map_=map_np, ndcg=ndcg_np))

        start = time.time()
        pre_np, rec_np, map_np, ndcg_np = eval_epoch(model, user_valid_dl, opt, valid_user_embeddings)
        print('\r(Test)  P@k:{pre},    R@k:{rec}, \n'
              '(Test)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,
                      pre=pre_np, rec=rec_np, map_=map_np, ndcg=ndcg_np))

        scheduler.step()
        valid_precision_max = valid_precision_max if valid_precision_max > pre_np[1] else pre_np[1]

        # path = './result/{}/'.format(C.DATASET)
        # folder = "{accuracy:8.5f}/".format(accuracy=valid_precision_max)
        # if not os.path.exists(path) or len(os.listdir(path)) == 0:
        #     if not os.path.exists(path): os.mkdir(path)
        #     if not os.path.exists(path+folder): os.mkdir(path+folder)
        #     np.save(path+folder+"user.npy", valid_user_embeddings.cpu().numpy())
        #     np.save(path+folder+"poi.npy", model.event_emb.weight.detach().cpu().numpy())
        # for file_name in os.listdir(path):
        #     if float(file_name) < valid_precision_max:
        #         shutil.rmtree(path+file_name)
        #         os.mkdir(path + folder)
        #         np.save(path+folder+"user.npy", valid_user_embeddings.cpu().numpy())
        #         np.save(path+folder+"poi.npy", model.event_emb.weight.detach().cpu().numpy())

    return valid_precision_max


def get_user_embeddings(model, user_dl, opt):
    """ Epoch operation in training phase. """

    valid_user_embeddings = torch.zeros((C.USER_NUMBER, opt.d_model), device='cuda:0')

    for batch in tqdm(user_dl, mininterval=2, desc='  - (Computing user embeddings)   ', leave=False):
        """ prepare data """
        user_idx, event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

        """ forward """
        prediction, users_embeddings = model(event_type)  # X = (UY+Z) ^ Tc
        valid_user_embeddings[user_idx] = users_embeddings

    return valid_user_embeddings


def main(trial):
    """ Main function. """
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.device = torch.device('cuda')

    # # # optuna setting for tuning hyperparameters
    # opt.n_layers = trial.suggest_int('n_layers', 2, 2)
    # opt.d_inner_hid = trial.suggest_int('n_hidden', 512, 1024, 128)
    # opt.d_k = trial.suggest_int('d_k', 512, 1024, 128)
    # opt.d_v = trial.suggest_int('d_v', 512, 1024, 128)
    # opt.n_head = trial.suggest_int('n_head', 1, 5, 1)
    # # opt.d_rnn = trial.suggest_int('d_rnn', 128, 512, 128)
    # opt.d_model = trial.suggest_int('d_model', 128, 1024, 128)
    # opt.dropout = trial.suggest_uniform('dropout_rate', 0.5, 0.7)
    # opt.smooth = trial.suggest_uniform('smooth', 1e-2, 1e-1)
    # opt.lr = trial.suggest_uniform('learning_rate', 0.00008, 0.0002)

    opt.lr = 0.01
    opt.epoch = 30
    opt.n_layers = 1  # 2
    opt.batch_size = 32
    opt.dropout = 0.5
    opt.smooth = 0.03
    if C.DATASET == 'Foursquare': opt.d_model, opt.n_head = 768, 3
    elif C.DATASET == 'Gowalla': opt.d_model, opt.n_head = 1024, 1
    elif C.DATASET == 'Yelp2018': opt.d_model, opt.n_head = 1024, 2
    else: opt.d_model, opt.n_head = 1024, 1

    print('[Info] parameters: {}'.format(opt))
    num_types = C.POI_NUMBER
    num_user = C.USER_NUMBER

    """ prepare model """
    model = Model(
        num_types=num_types,
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        device=opt.device
    )
    model = model.cuda()

    """ loading data"""
    ds = dataset()
    print('[Info] Loading data...')
    user_dl = ds.get_user_dl(opt.batch_size)
    user_valid_dl = ds.get_user_valid_dl(opt.batch_size)
    data = (user_valid_dl, user_dl)

    """ optimizer and scheduler """
    parameters = [
                  {'params': model.parameters(), 'lr': opt.lr},
                  ]
    optimizer = torch.optim.Adam(parameters)  # , weight_decay=0.01
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    # user_embeddings_before_training = get_user_embeddings(model, user_valid_dl, opt)
    # path = './result/{}/'.format(C.DATASET)
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # np.save(path + "before_training.npy", user_embeddings_before_training.detach().cpu().numpy())
    # print(path + "before_training.npy")

    # user_embeddings_before_training = None

    """ train the model """
    return train(model, data, optimizer, scheduler, opt)


if __name__ == '__main__':
    # main()
    assert C.ENCODER in {'Transformer', 'gMLP', 'TransformerLS', 'hGCN', 'None'}
    assert C.ABLATION in {'Full', 'w/oImFe', 'w/oMMD', 'w/oMatcher', 'w/oFeTra', 'w/oGlobal', 'w/oAtt', 'w/oConv', 'w/oGraIm'}
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=100)

    # df = study.trials_dataframe()
    #
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

