#!/usr/bin/env python
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from Bio import SeqIO
from prot_repr.utils import dataset_folder, result_folder
from prot_repr.models.dim import DIMModel

VOCAB_SIZE = 27

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return tensor.cuda().cuda()
    return tensor


class FastaDataset(Dataset):
    """Custom PyTorch Dataset that takes a file containing proteins.

        Args:
                fname : path to a fasta file
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the model.
    """
    padding_idx = 0
    def __init__(self, fname=os.path.join(dataset_folder, 'uniprotkb_human.fasta')):
        self.voc = {chr(ord('A')+el): el+1 for el in range(VOCAB_SIZE)}
        with open(fname, "r") as fd:
            self.seqs = [torch.tensor([self.voc[s] for s in str(record.seq)])
                         for i, record in enumerate(SeqIO.parse(fd, "fasta"))
                         if len(record.seq) < 200]
        print(self.seqs[0].dtype)
        print(f"Fasta dataset loaded: {len(self.seqs)} sequences recorded")

    def __getitem__(self, i):
        return Variable(self.seqs[i])

    def __len__(self):
        return len(self.seqs)

    def __str__(self):
        return "Dataset containing {} proteins.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([len(seq) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length)).long() + FastaDataset.padding_idx
        for i, seq in enumerate(arr):
            collated_arr[i, :len(seq)] = seq
        return collated_arr


def pretrain(output_path=result_folder, restore_existing=True,
             valid_size=0.2, n_epochs=5, batch_size=128, seed=42, **model_params):
    """Trains the model RNN"""
    #seed the program
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load vocabulary and data
    dataset = FastaDataset()
    valid_size = int(len(dataset) * valid_size) if valid_size < 1 else valid_size
    train_size = len(dataset) - valid_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                              collate_fn=FastaDataset.collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True,
                              collate_fn=FastaDataset.collate_fn)

    # create model and train
    model = DIMModel(train_loader=train_loader, valid_loader=valid_loader, **model_params)
    # model.to_cuda()
    os.makedirs(output_path, exist_ok=True)

    # Can restore from a saved RNN
    es_callback = None
    if restore_existing:
        logger = TestTubeLogger(
            save_dir=output_path,
            version=1  # An existing version with a saved checkpoint
        )
    else:
        logger = True
    trainer = Trainer(
        logger=logger,
        default_save_path=output_path,
        max_nb_epochs=n_epochs,
        early_stop_callback=es_callback)

    trainer.fit(model)


if __name__ == '__main__':
    config = dict(
        encoder_params=dict(
            arch='cnn',
            vocab_size=VOCAB_SIZE,
            padding_idx=FastaDataset.padding_idx,
            embedding_size=32,
            cnn_sizes=[32]*3,
            output_size=512,
            kernel_size=11,
            pooling_len=2,
            pooling='avg',
            dilatation_rate=1,
            activation='ReLU',
            b_norm=True,
            dropout=0.0),
        local_mine_params=dict(
            hidden_sizes=[256]*2,
            activation='ReLU',
            b_norm=False,
            dropout=0.0),
        global_mine_params=dict(
            hidden_sizes=[256]*2,
            activation='ReLU',
            b_norm=False,
            dropout=0.0),
        mode='concat',
        max_t=10,

        alpha=1.0,
        beta=1.0,
        gamma=0.1,

        valid_size=0.2,
        n_epochs=5,
        batch_size=64,
        optimizer='Adam',
        lr=1e-3
    )

    pretrain(**config)