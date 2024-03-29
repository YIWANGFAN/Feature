import argparse
import copy
import math
import sys
import os
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm
from dnfutils import AverageMeter

import dnftsne
from dnfscore import *
import model_wxf as fnn
from dnfdataload import dataset_prepare

pi = torch.from_numpy(np.array(np.pi))


def data_normlize(train_data, test_data):
    return (test_data - train_data.mean(axis=0)) / train_data.std(axis=0)


def supervise_mean_var(data, labels):
    assert (data.shape[0] == labels.shape[0]), 'data and label must have the same length'
    # init class mean vectors
    label_class = np.array(list(set(copy.deepcopy(labels).cpu().numpy())))
    mean_list = []
    for lb in label_class:
        data_j = data[labels == lb]
        mean_j = torch.mean(data_j, 0, True)

        mean_list.append(mean_j)
    class_mean = torch.cat(mean_list, 0)
    return class_mean


def initial(train_data, train_label, epoch):
    if epoch == 0:
        with torch.no_grad():
            print("SCH 2: init mean by z, init var by predifined value")
            train_data = train_data.to(device)
            out, _ = model(train_data)
            class_mean = supervise_mean_var(out, train_label)
            # print(class_mean)
        model.class_mean.data = class_mean.clone()
        print("SCH2: init the mean adaptable parameters, and var un-adaptable")
        model.class_mean.requires_grad = True


def train(epoch):
    # print('train:::',model)
    model.train()
    if epoch == 0:
        initial(train_data, train_label, epoch)
    class_mean = model.class_mean

    # statistics box
    losses_ag = AverageMeter()
    logjac_ag = AverageMeter()
    logp_ag = AverageMeter()

    # utilities
    pbar = tqdm(total=len(train_loader.dataset))

    # strat to train
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer_model.zero_grad()

        # flow loss with class specific means
        mean_j = torch.index_select(class_mean, 0, labels)

        u_out, dnf_loss, log_probs, logdet = model.dnf_Gaussian_log_likelihood(data, mean_j, args.vc)

        loss = dnf_loss.mean()

        loss.backward()
        optimizer_model.step()

        # update statistics
        losses_ag.update(loss.item(), labels.shape[0])
        logp_ag.update(log_probs.mean().item(), labels.shape[0])
        logjac_ag.update(logdet.mean().item(), labels.shape[0])

        pbar.update(data.size(0))
        pbar.set_description(
            'Total val/avg={:.3f}/{:.3f}  LogP val/avg={:.3f}/{:.3f} LogDet val/avg= {:.3f}/{:.3f}'.format(
                losses_ag.val, losses_ag.avg, logp_ag.val, logp_ag.avg, logjac_ag.val, logjac_ag.avg))

    # utility
    pbar.close()

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu) / std
if __name__ == "__main__":

    if sys.version_info < (3, 6):
        print('Sorry, this code might need Python 3.6 or higher')
    else:
        print("%s" % ' '.join(sys.argv))
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Flows')
    parser.add_argument('--batch-size', type=int, default=4, help='input batch size for training (default: 300)')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.003)')
    parser.add_argument('--num-blocks', type=int, default=5, help='number of invertible blocks (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU used (default: 0)')
    parser.add_argument('--n_filter', type=int, default=0, help='dryrun a small number of classes (default: 0)')
    parser.add_argument('--vc', type=float, default=1, help='variance of the class space (default: 1.0)')

    args = parser.parse_args()

    # working env
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # load data
    print("loading data ...")
    root = '/home/daip/share/old_share/wxf/feature/DNF/car/resnet'
    backboon_name = 'resnet'
    train_path = '/home/daip/share/old_share/wxf/feature/feature/car/resnet/car_train_res.npz'
    test_path = '/home/daip/share/old_share/wxf/feature/feature/car/resnet/car_test_res.npz'
    t0_dataset, t2_dataset = dataset_prepare(args.n_filter,train_path,test_path)
    train_loader = torch.utils.data.DataLoader(t0_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    print("data loaded")
    # train_data  = t0_dataset[:][0].to(device)
    # train_label = t0_dataset[:][1].to(device)
    train_data = t0_dataset[:][0]
    print(len(train_data[0]))
    # train_data = feature_normalize(train_data)
    # train_data = train_data.numpy()
    # train_data = feature_normalize(train_data)
    # train_data = train_data*100
    # train_data = torch.Tensor(train_data)
    # print('no',train_data.shape)



    # print(train_data)
    train_label = t0_dataset[:][1]


    # print(train_label)

    # define the network structure
    num_inputs = train_data.shape[1]
    num_hidden = train_data.shape[1]
    act = 'sigmoid'  # 'relu' 'PReLU' 'LeakyReLU' 'sigmoid' 'tanh'

    print("Flow structure: %d blocks with activation=%s" % (args.num_blocks, act))

    modules = []
    for i in range(args.num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, act=act),
        ]
    model = fnn.FlowSequential(*modules)
    # print(model)
    class_mean = model.set_class_mean()
    # print(class_mean)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    model.to(device)
    # print(module)

    # define the optimizer and its optimizer
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr)

    # if not os.path.exists('./z_trn6'):
    #     os.mkdir('./z_trn6');
    # if not os.path.exists('./z_Sitwn6'):
    #     os.mkdir('./z_Sitwn6');
    # if not os.path.exists('./chkptn6'):
    #     os.mkdir('./chkptn6');
    save_model_path = root + backboon_name + '/weights'
    save_train_data_path = root + backboon_name + '/train_data'
    save_test_data_path = root + backboon_name + '/test_data'
    save_tnes_path = root + backboon_name + '/tnes'
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.isdir(save_train_data_path):
        os.makedirs(save_train_data_path)
    if not os.path.isdir(save_test_data_path):
        os.makedirs(save_test_data_path)
    if not os.path.isdir(save_tnes_path):
        os.makedirs(save_tnes_path)
    # initial
    print('\nDrawing tsne of train_data ......')
    dnftsne.main(train_path, -1,root,backboon_name)
    # # scoing
    # print('do scoring ......')
    # trails_path = './data/xvector/Sitw/core-core.lst'
    # eer = cosine_scoring_by_trails('./data/xvector/vox_4k.npz', './data/xvector/Sitw/enroll.npz',
    #                                './data/xvector/Sitw/test.npz', trails_path)
    # print('>>>> epoch = {}  cosine eer% = {:.2f}%'.format(-1, eer * 100))
    # file = open('result_eer', 'a')
    # file.write('epoch = {} cosine-eer% = {:.2f} \n'.format(-1, eer * 100))
    # file.close()

    #save_path


    for epoch in range(args.epochs):
        print('\nEpoch: {}'.format(epoch))
        print(time.asctime(time.localtime(time.time())))

        train(epoch)
        # save model to checkpoint

        print("saving model to"+ save_model_path +"/model_epoch%d.pt" % (epoch))
        torch.save(model, save_model_path+'/model_epoch{}.pt'.format(epoch))

        # evaluation
        with torch.no_grad():
                print('do evaluation ......\n')
                path0 = root + backboon_name+'/train_data/'+backboon_name+'{}.npz'.format(epoch)
                u0, _ = model(t0_dataset[:][0].to(device))
                # u0, _ = model(t0_dataset[:][0])
                # print("epoch",model)
                label0 = np.load(train_path)['utt']
                data0 = u0.cpu().detach().numpy()
                data00 = data_normlize(data0, data0)
                print(len(data00[1]))
                np.savez(path0, vector=data00, utt=label0)

                path2 = root + backboon_name+'/test_data/'+backboon_name+'{}.npz'.format(epoch)
                u2, _ = model(t2_dataset[:][0].to(device))
                # u2, _ = model(t2_dataset[:][0])
                label2 = np.load(test_path)['utt']
                data2 = u2.cpu().detach().numpy()
                data2 = data_normlize(data0, data2)
                np.savez(path2, vector=data2, utt=label2)

                # scoing
                # print('do scoring ......')
                # eer = cosine_scoring_by_trails(path0, path1, path2, trails_path)
                # print('>>>> epoch = {}  cosine eer% = {:.2f}%'.format(epoch, eer * 100))

                # file = open('result_eer', 'a')
                # file.write('epoch = {} cosine-eer% = {:.2f} \n'.format(epoch, eer * 100))
                # file.close()

                # draw latent tsne

                print('\nDrawing tsne of latent space ......')
                dnftsne.main(path0, epoch,root,backboon_name)

    print("Training end..")
    print(time.asctime(time.localtime(time.time())))
