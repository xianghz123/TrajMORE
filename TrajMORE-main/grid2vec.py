import json
import os
import random
import time
import utils
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.neighbors import KDTree
import torch.nn as nn


class GridEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, grid2idx: dict, window_size, neg_rate):
        self.window_size = window_size
        self.neg_rate = neg_rate
        self.grid2idx = grid2idx
        idx2grid = {grid2idx[c]: c for c in grid2idx}
        self.sorted_grid = []
        for i in range(len(grid2idx)):
            self.sorted_grid.append(idx2grid[i])
        self.tree = KDTree(self.sorted_grid)
        distance, index = self.tree.query(self.sorted_grid, k=window_size + 1)
        index = torch.tensor(index)
        distance = torch.tensor(distance)
        index = torch.unsqueeze(index[:, 1:], 2)
        distance = torch.unsqueeze(distance[:, 1:], 2)
        weight = F.softmax(-distance, dim=1) * window_size
        self.positive = torch.cat((index, weight), 2)

    def __len__(self):
        return len(self.grid2idx)

    def __getitem__(self, idx):
        ones = torch.ones(len(self.grid2idx))
        ones[idx] = 0
        p_i = self.positive[idx, :, 0].long()
        ones[p_i] = 0
        neg_index = torch.multinomial(ones, self.neg_rate * self.window_size, replacement=True)
        return torch.tensor([idx]), self.positive[idx], neg_index


class Grid2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Grid2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.in_embedding = nn.Embedding(vocab_size, embedding_size)
        self.out_embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, center: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        """
        :param center: [batch_size]
        :param positive: [batch_size, window_size, 2]
        :param negative: [batch_size, window_size * neg_rate, 2]
        :return: loss, [batch_size]
        """
        center.squeeze_()
        c_vec = self.in_embedding(center).unsqueeze(2)  # [batch, embedding_size, 1]
        p_vec = self.out_embedding(positive[:, :, 0].long()).squeeze()  # [batch, window_size, embedding_size]
        n_vec = self.out_embedding(negative).squeeze()  # [batch, window_size * neg_rate, embedding_size]

        p_dot = torch.bmm(p_vec, c_vec)  # [batch, w_s]
        n_dot = torch.bmm(n_vec, -c_vec)  # [batch, w_s * n_r]
        log_pos = torch.mul(F.logsigmoid(p_dot).squeeze(), positive[:, :, 1])  # [batch, w_s]
        log_neg = F.logsigmoid(n_dot)  # [batch, w_s * n_r]
        loss = -(log_pos.sum(1) + log_neg.sum(1))  # [batch]
        return loss

    def input_embedding(self):
        return self.in_embedding.weight.data.cpu().numpy()


def train_grid2vec(file, window_size, embedding_size, batch_size, epoch_num, learning_rate, checkpoint, visdom_port):
    # init
    timer = utils.Timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_rate = 100  # negative sampling rate

    # read dict
    timer.tik("read json")
    with open(file) as f:
        str_grid2idx = json.load(f)
        f.close()
    grid2idx = {eval(c): str_grid2idx[c] for c in list(str_grid2idx)}
    timer.tok()

    # build dataset
    timer.tik("build dataset")
    dataset = GridEmbeddingDataset(grid2idx, window_size, neg_rate)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    timer.tok()

    # training preparation
    model = Grid2Vec(len(grid2idx), embedding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    cp_save_rate = 0.8
    np_save_rate = 0.95
    iter_num = len(dataset) // batch_size + 1
    epoch_start = 0
    best_loss = float('inf')
    loss_list = []
    best_accuracy = 0
    last_save_epoch = 0

    # start visdom
    if visdom_port != 0:
        from visdom import Visdom
        env = Visdom(port=visdom_port)
        pane1 = env.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(title='loss'))
        pane2 = env.line(
            X=np.array([0]),
            Y=np.array([0]),
            opts=dict(title='accuracy'))

    # load checkpoint / pretrained_state_dict
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        if checkpoint.get('model'):
            model.load_state_dict(checkpoint['model'])
        if checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('epoch'):
            epoch_start = checkpoint['epoch'] + 1
            last_save_epoch = epoch_start

    # start training
    print(f'\n-------------training config-------------\n'
          f'start time : {timer.now()}\n'
          f'window_size : {dataset.window_size}\n'
          f'batch_size : {dataloader.batch_size}\n'
          f'embedding_size : {model.embedding_size}\n'
          f'epoch_num : {epoch_num}\n'
          f'learning_rate : {learning_rate}\n'
          f'device : {device}\n')
    timer.tik("training")
    for epoch in range(epoch_start, epoch_num):
        for i, (center, positive, negative) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(center.to(device), positive.to(device), negative.to(device)).mean()
            loss.backward()
            optimizer.step()
            loss_list.append(float(loss))
            if visdom_port != 0:
                env.line(
                    X=np.array([(epoch - epoch_start) * iter_num + i]),
                    Y=np.array([float(loss)]),
                    win=pane1,
                    update='append')
            if i % (iter_num // 4 + 1) == 0:
                acc = evaluate_grid2vec(model.input_embedding(), dataset, test_num=100)
                timer.tok(f"epoch:{epoch} iter:{i}/{iter_num} loss:{round(float(loss), 3)} acc:{round(acc, 3)}")
                if i == 0 and epoch == epoch_start:
                    best_loss = np.mean(loss_list)
                    best_accuracy = acc
                    continue
                if np.mean(loss_list) < cp_save_rate * best_loss:
                    best_loss = np.mean(loss_list)
                    loss_list.clear()
                    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(checkpoint, f'model/checkpoint_{embedding_size}_{epoch}_{i}_{round(float(loss), 3)}.pth')
                elif epoch - last_save_epoch > 5:
                    last_save_epoch = epoch
                    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(checkpoint, f'model/checkpoint_{embedding_size}_{epoch}_{i}_{round(float(loss), 3)}.pth')
                if visdom_port != 0:
                    env.line(
                        X=np.array([(epoch - epoch_start) * iter_num + i]),
                        Y=np.array([acc]),
                        win=pane2,
                        update='append')
                if acc * np_save_rate > best_accuracy:
                    best_accuracy = acc
                    np.save(f'model/grid_embedding_{embedding_size}_{round(acc, 2)}', model.input_embedding())


def evaluate_grid2vec(embedding_weights, dataset, test_num=10):
    random.seed(20000221)
    samples_index = random.sample(range(len(dataset)), test_num)
    samples_weights = embedding_weights[samples_index, :]

    from scipy.spatial.distance import cdist
    nearest_index = cdist(samples_weights, embedding_weights, metric='cosine').argsort(axis=1)

    predict = list(nearest_index[:, 1:dataset.window_size + 1])
    truth = [dataset[idx][1].numpy() for idx in samples_index]
    accuracy = np.mean([len(np.intersect1d(predict[i], truth[i])) for i in range(test_num)]) / dataset.window_size
    return accuracy