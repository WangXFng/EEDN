import argparse
import numpy as np
import time
from EEDN import metric, Utils

import torch
import torch.optim as optim

if torch.cuda.is_available():
    pass
else:
    pass

import optuna

import Constants as C

from preprocess.Dataset import Dataset as dataset
from EEDN.Models import Model
from tqdm import tqdm


def train_epoch(model, user_dl, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    [pre, rec, map_, ndcg] = [[[] for i in range(4)] for j in range(4)]
    for batch in tqdm(user_dl, mininterval=2, desc='  - (Training)   ', leave=False):
        optimizer.zero_grad()

        """ prepare data """
        event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

        """ forward """
        prediction, users_embeddings = model(event_type)

        """ compute metric """
        metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

        """ backward """
        loss = Utils.type_loss(prediction, event_type, event_time, test_label, opt)

        loss.backward(retain_graph=True)
        """ update parameters """
        optimizer.step()

    results_np = map(lambda x: [np.around(np.mean(i), 5) for i in x], [pre, rec, map_, ndcg])
    return results_np


def eval_epoch(model, user_valid_dl, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    [pre, rec, map_, ndcg] = [[[] for i in range(4)] for j in range(4)]
    with torch.no_grad():
        for batch in tqdm(user_valid_dl, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare test data """
            event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

            """ forward """
            prediction, users_embeddings = model(event_type)  # X = (UY+Z) ^ T

            """ compute metric """
            metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

    results_np = map(lambda x: [np.around(np.mean(i), 5) for i in x], [pre, rec, map_, ndcg])
    return results_np


def train(model, data, optimizer, scheduler, opt):
    """ Start training. """
    (user_valid_dl, user_dl) = data

    best_ = [np.zeros(4) for i in range(4)]
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i + 1, ']')

        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        start = time.time()
        [pre, rec, map_, ndcg] = train_epoch(model, user_dl, optimizer, opt)
        print('\r(Training)  P@k:{pre},    R@k:{rec}, \n'
              '(Training)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

        start = time.time()
        [pre, rec, map_, ndcg] = eval_epoch(model, user_valid_dl, opt)
        print('\r(Test)  P@k:{pre},    R@k:{rec}, \n'
              '(Test)map@k:{map_}, ndcg@k:{ndcg}, '
              'elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

        scheduler.step()

        if best_[-1][1] < ndcg[1]: best_ = [pre, rec, map_, ndcg]

    with open('./result/beta_lambda_{}.log'.format(C.DATASET), 'a') as f:
        f.write("%s Beta:%.4f Lambda:%.4f " % (C.DATASET, opt.lambda_, opt.delta))
        f.write("P@k:{pre}, R@k:{rec}, {map_}, ndcg@k:{ndcg}\n"
                .format(pre=best_[0], rec=best_[1], map_=best_[2], ndcg=best_[3]))
        f.close()
    return best_[-1][1]


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

    # >> optuna setting for tuning hyperparameters
    # opt.n_layers = trial.suggest_int('n_layers', 2, 2)
    # opt.n_head = trial.suggest_int('n_head', 1, 5, 1)
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

    lambda_delta = {
        'Foursquare': [0.4, 0.7], # 0.4, 0.5  # 0.35, 0.5  # 0.5, 1
        'Gowalla': [1.5, 4],  # 0.38, 1  # 1.5, 4
        'Yelp2018': [1, 4],  # 0.35, 1  # 1, 4
        'douban-book': [0.5, 1],
        'ml-1M': [0.9, 1],
    }

    if C.DATASET in lambda_delta:
        [opt.lambda_, opt.delta] = lambda_delta[C.DATASET]
    else:
        opt.lambda_, opt.delta = trial.suggest_uniform('lambda', 0.1, 4), trial.suggest_uniform('delta', 0.1, 4)
        # opt.lambda_, opt.delta = 0.5, 1

    print('[Info] parameters: {}'.format(opt))
    """ prepare model """
    model = Model(
        num_types=C.POI_NUMBER,
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        device=opt.device
    )
    model = model.cuda()

    """ loading data"""
    print('[Info] Loading data...')
    ds = dataset()
    user_dl = ds.get_user_dl(opt.batch_size)
    user_valid_dl = ds.get_user_valid_dl(opt.batch_size)
    data = (user_valid_dl, user_dl)

    """ optimizer and scheduler """
    parameters = [{'params': model.parameters(), 'lr': opt.lr},]
    optimizer = torch.optim.Adam(parameters)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ train the model """
    return train(model, data, optimizer, scheduler, opt)


if __name__ == '__main__':
    assert C.ENCODER in {'Transformer', 'gMLP', 'TransformerLS', 'hGCN'}
    assert C.ABLATION in {'Full', 'w/oImFe', 'w/oFeTra', 'w/oGlobal', 'w/oAtt', 'w/oConv', 'w/oGraIm'}
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=100)


