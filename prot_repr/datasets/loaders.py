import os
from Bio import SeqIO
from prot_repr.utils import dataset_folder


def load_fasta_proteins(fname=os.path.join(dataset_folder, 'proteins_seqs.fasta')):
    with open(fname, "r") as fd:
        return [str(record.seq) for record in SeqIO.parse(fd, "fasta")]