import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models import SessionGraphAttn
from dataset import SessionData

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica',
                    help='dataset name: diginetica/gowalla/lastfm')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]0.001

parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', type=bool, default=False, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--alpha', type=float, default=0.75, help='parameter for beta distribution')
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--use_lp_pool', type=str, default="True")
parser.add_argument('--train_flag', type=str, default="True")
parser.add_argument('--PATH', default='../checkpoint/Atten-Mixer_gowalla.pt', help='checkpoint path')
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--softmax', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dot', default=True, action='store_true')
parser.add_argument('--last_k', type=int, default=7)
parser.add_argument('--l_p', type=int, default=4)
parser.add_argument_group()

opt = parser.parse_args()
print(opt)

hyperparameter_defaults = vars(opt)
config = hyperparameter_defaults

class AreaAttnModel(pl.LightningModule):

    def __init__(self, opt, n_node):
        super().__init__()

        self.opt = opt
        self.cnt = 0
        self.best_res = [0, 0]
        self.model = SessionGraphAttn(opt, n_node)
        self.loss = nn.Parameter(torch.Tensor(1))

    def forward(self, *args):

        return self.model(*args)

    def training_step(self, batch, batch_idx):

        alias_inputs, A, items, mask, mask1, targets, n_node = batch

        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        mask1.squeeze_()
        targets.squeeze_()
        n_node.squeeze_()

        hidden = self(items)

        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        loss = self.model.loss_function(scores, targets - 1)

        return loss

    def validation_step(self, batch, batch_idx):

        alias_inputs, A, items, mask, mask1, targets, n_node = batch
        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        mask1.squeeze_()
        targets.squeeze_()
        n_node.squeeze_()
        hidden = self(items)
        assert not torch.isnan(hidden).any()
        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(20)[1]
        sub_scores = sub_scores.cpu().detach().numpy()
        res = []
        for score, target in zip(sub_scores, targets):
            hit = float(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr = 0
            else:
                mrr = 1 / (np.where(score == target - 1)[0][0] + 1)
            phi = 0
            res.append([hit, mrr, phi / 20])

        return torch.tensor(res)

    def validation_epoch_end(self, validation_step_outputs):

        output = torch.cat(validation_step_outputs, dim=0)
        hit = torch.mean(output[:, 0]) * 100
        mrr = torch.mean(output[:, 1]) * 100
        arp = torch.sum(output[:, 2]) / len(output)
        if hit > self.best_res[0]:
            self.best_res[0] = hit
        if mrr > self.best_res[1]:
            self.best_res[1] = mrr
        self.log('hit@20', self.best_res[0])
        self.log('mrr@20', self.best_res[1])
        self.log('arp@20', arp)
        print(mrr, hit, arp)

    def test_step(self, batch, idx):

        alias_inputs, A, items, mask, mask1, targets, n_node = batch
        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        mask1.squeeze_()
        targets.squeeze_()
        n_node.squeeze_()
        hidden = self(items)
        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(20)[1]
        sub_scores = sub_scores.cpu().detach().numpy()
        res = []
        for score, target in zip(sub_scores, targets):
            hit = float(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr = 0
            else:
                mrr = 1 / (np.where(score == target - 1)[0][0] + 1)
            res.append([hit, mrr])

        return torch.tensor(res)

    def test_epoch_end(self, test_step_outputs):

        output = torch.cat(test_step_outputs, dim=0)
        hit = torch.mean(output[:, 0]) * 100
        mrr = torch.mean(output[:, 1]) * 100
        if hit > self.best_res[0]:
            self.best_res[0] = hit
        if mrr > self.best_res[1]:
            self.best_res[1] = mrr
        self.log('hit@20', self.best_res[0])
        self.log('mrr@20', self.best_res[1])
        print(mrr, hit)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_dc_step, gamma=opt.lr_dc)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def main():

    seed = 123
    pl.seed_everything(seed)
    if opt.dataset == 'diginetica':
        n_node = 42597
    elif opt.dataset == 'gowalla':
        n_node = 29511
    elif opt.dataset == 'lastfm':
        n_node = 38616

    def get_freer_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        # memory_available = memory_available[1:6]
        return int(np.argmax(memory_available))

    session_data = SessionData(name=opt.dataset, batch_size=opt.batchSize)
    early_stop_callback = EarlyStopping(
        monitor='mrr@20',
        min_delta=0.00,
        patience=opt.patience,
        verbose=False,
        mode='max'
    )
    trainer = pl.Trainer(gpus=[get_freer_gpu()], deterministic=True, max_epochs=10, num_sanity_val_steps=2,
                         callbacks=[early_stop_callback])
    if opt.train_flag == "True":
        model = AreaAttnModel(opt=opt, n_node=n_node)
        trainer.fit(model, session_data)
    else:
        model = AreaAttnModel(opt=opt, n_node=n_node)
        model.load_state_dict(torch.load(opt.PATH))
        model.eval()
        trainer.test(model, session_data.test_dataloader())

if __name__ == "__main__":
    main()