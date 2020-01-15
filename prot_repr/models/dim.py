import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prot_repr.models.components import get_encoder, FcNet


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation='ReLU', b_norm=False, dropout=0.0):
        super().__init__()
        self.network = nn.Sequential(
                        FcNet(input_size, hidden_sizes=hidden_sizes,
                              activation=activation, b_norm=b_norm, dropout=dropout),
                        nn.Linear(in_features=hidden_sizes[-1], out_features=1))

    def forward(self, inputs):
        # expecting b * d
        return self.network(inputs).squeeze(dim=-1)


class DIM(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""

    def __init__(self, encoder_params, local_mine_params, global_mine_params, mode='concat', max_t=10):
        super(DIM, self).__init__()
        assert mode in ['concat', 'product'], "mode should be either concat or product"
        self.max_t = max_t
        self.encoder = get_encoder(**encoder_params)
        self.g_dim, self.l_dim = self.encoder.output_dim, self.encoder.local_output_dim
        self.global_discr = Discriminator(input_size=(max_t*self.l_dim + self.g_dim), **global_mine_params)
        self.local_discr = Discriminator(input_size=self.g_dim+self.l_dim, **local_mine_params)

    def forward(self, x):
        g_dim, l_dim = self.g_dim, self.l_dim
        globals_, locals_ = self.encoder(x, return_locals=True)
        b_dim, t_dim, _ = locals_.shape
        # locals_: b , t , f_dim
        # globals_: b , f_dim

        # global_MI  # sample and concat all local features together and then with the global repr
        idx_t = torch.randint(0, t_dim, size=(b_dim, self.max_t)).sort(dim=1)[0] + \
                torch.arange(0, b_dim*t_dim, step=t_dim)[:, None]
        locals_sampled_t = locals_.reshape(-1, l_dim)[idx_t.view(-1)].reshape(b_dim, self.max_t * l_dim)

        globals_inputs = torch.cat((locals_sampled_t.repeat_interleave(b_dim, dim=0),
                                   globals_.repeat(b_dim, 1)), dim=-1).reshape(b_dim, b_dim, -1) # b , b
        globals_mi = self.global_discr(globals_inputs)

        # local MI
        locals_l = locals_.repeat_interleave(b_dim, dim=0)
        globals_l = globals_.repeat(b_dim, 1).unsqueeze(1).expand(b_dim*b_dim, t_dim, g_dim)
        locals_mi = self.local_discr(torch.cat((locals_l, globals_l), dim=-1)).transpose(0, 1) # t , b*b
        locals_mi = locals_mi.reshape(t_dim, b_dim, b_dim)


        # local_left_term_inputs = self.local_discr(torch.cat((globals_l, locals_), dim=-1)) # b , t
        # samples_prime = torch.randperm(b_dim*t_dim) # b * n_samples
        # locals_prime = locals_.reshape(-1, f_dim)[samples_prime.view(-1)].reshape(*locals_.shape)
        # local_right_term_inputs = self.local_discr(torch.cat((globals_.unsqeeze(1).reshape(*locals_prime.shape),
        #                               locals_prime), dim=-1)) # b , t

        return globals_, locals_, globals_mi, locals_mi


class DIMModel(pl.LightningModule):
    def __init__(self, encoder_params, local_mine_params, global_mine_params, mode, max_t,
                 train_loader, valid_loader, alpha=0., beta=1.0, gamma=0.001, optimizer='Adam', lr=1e-3):
        super().__init__()
        # self.hparams =
        self.network = DIM(encoder_params, local_mine_params, global_mine_params, mode, max_t)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=self.lr/1000)
        return [opt], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.valid_loader

    def train_val_step(self, batch, is_train=True):
        # REQUIRED
        n, t = batch.shape
        globals_, _, globals_mi, locals_mi = self.network(batch)
        t_eff = locals_mi.shape[0]
        prior_loss = (globals_ ** 2).sum()
        locals_logsoftmax = nn.LogSoftmax(dim=-1)(locals_mi)
        globals_logsoftmax = nn.LogSoftmax(dim=-1)(globals_mi)
        locals_loss = - locals_logsoftmax.diagonal(dim1=1, dim2=2).mean()
        globals_loss = - globals_logsoftmax.diagonal().mean()
        loss = self.alpha * globals_loss + self.beta * locals_loss + self.gamma * prior_loss
        acc_globals = (torch.argmax(globals_logsoftmax, dim=-1) == torch.arange(n)).float().mean()
        acc_locals = (torch.argmax(locals_logsoftmax, dim=-1) == torch.arange(n).repeat(t_eff, 1)).float().mean()
        tensorboard_logs = dict(prior_loss=prior_loss,
                                local_acc=acc_locals, global_acc=acc_globals,
                                local_loss=locals_loss, global_loss=globals_loss)
        mode = 'train_' if is_train else 'val_'
        tensorboard_logs[mode+'loss'] = loss
        return {'loss': loss, 'log': tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self.train_val_step(batch, True)

    def validation_step(self, batch, batch_idx):
        return self.train_val_step(batch, False)