import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask



def AttentioNCE(out_1, out_2, out_3, out_4, out_5, out_6, batch_size, temperature, d_pos, d_neg):
    '''
    Parameters
    ----------
    # anchor points
    out_1 : anchor features [bs,dim]
    out_2 : anchor features [bs,dim]

    # 4 positive views for  each anchor point 
    out_3 : positive view1 features [bs,dim]
    out_4 : positive view2 features [bs,dim]
    out_5 : positive view3 features [bs,dim]
    out_6 : positive view4 features [bs,dim]

    batch_size : Under SimCLR framework, negative sample size N = 2 x (batch size - 1)
    temperature: temperature scaling
    d_pos : positive scaling factor
    d_neg : negative scaling factor
    Returns: AttentioNCE loss values [2bs,]
    -------
    '''
    
    anchor = torch.cat([out_1, out_2], dim=0)  # [2bs*dim]

    # pos score
    pos_views = torch.cat([out_3, out_4, out_5, out_6],dim=1).view(-1, 4, feature_dim) # [bs,4,dim]
    pos_sim = torch.sum(anchor.view(2, batch_size, 1, feature_dim) * pos_views, dim=-1).view(-1, 4)
    # [2,bs,1,dim] x [bs,4,dim]  -> [2,bs,4,dim] ->  [2,bs,4]-> [2bs,4]
    alpha = torch.nn.functional.softmax(pos_sim.detach()/ d_pos, dim=-1) # [2bs,4]
    pos_score = torch.exp(torch.sum(pos_sim * alpha, dim=-1) / temperature)  # [2bs,] This step is obtained based on the additivity property of inner product operations.

    # neg score
    sim = torch.mm(anchor, anchor.t().contiguous())  # [2bs,2bs]
    mask = get_negative_mask(batch_size).to(device)
    neg_sim = sim.masked_select(mask).view(2 * batch_size, -1)  # [2bs, 2N-2]
    beta = (torch.nn.functional.softmax(neg_sim.detach() / d_neg, dim=-1) + 1e-6) * (2 * batch_size - 2)
    neg_score = torch.exp(neg_sim * beta / temperature)  # [2bs, 2N-2] This step is obtained based on the additivity property of inner product operations.

    # contrastive loss
    loss = - torch.log(pos_score / (pos_score + neg_score.sum(dim=-1))).mean() # [2bs,]
    return loss


def train(net, data_loader, train_optimizer, temperature, d_pos, d_neg):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, pos_3,pos_4, pos_5, pos_6, target in train_bar:
        pos_1, pos_2, pos_3,pos_4, pos_5, pos_6 = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True),pos_3.to(device, non_blocking=True),pos_4.to(device, non_blocking=True), pos_5.to(device, non_blocking=True), pos_6.to(device, non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        feature_3, out_3 = net(pos_3)
        feature_4, out_4 = net(pos_4)
        feature_5, out_5 = net(pos_5)
        feature_6, out_6 = net(pos_6)

        loss = AttentioNCE(out_1, out_2, out_3, out_4, out_5, out_6, batch_size, temperature, d_pos, d_neg)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='SimCLR', type=str, help='SimCLR Framework')
    parser.add_argument('--dataset_name', default='cifar10', type=str, help='Choose dataset')
    parser.add_argument('--d_pos', default=1, type=float, help='Number of images in each mini-batch')
    parser.add_argument('--d_neg', default=1, type=float, help='Number of images in each mini-batch')


    # args parse
    args = parser.parse_args()
    print(args)

    feature_dim, temperature,  k = args.feature_dim, args.temperature, args.k
    batch_size, epochs, estimator = args.batch_size, args.epochs, args.estimator
    dataset_name = args.dataset_name
    d_pos,d_neg = args.d_pos, args.d_neg


    # data prepare
    train_data = utils.get_dataset(dataset_name, root=args.root)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,drop_last=True)
    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(train_data.classes)
    print('# Classes: {}'.format(c))

    # training loop
    if not os.path.exists('../results/{}'.format(dataset_name)):
        os.makedirs('../results/{}'.format(dataset_name))

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, temperature, d_pos, d_neg)

        if epoch % 200 == 0:
            torch.save(model.state_dict(), '../results/{}/{}_{}_4model_{}_{}_{}_{}.pth'.format(dataset_name, dataset_name, estimator, batch_size, epoch,d_pos,d_neg))
