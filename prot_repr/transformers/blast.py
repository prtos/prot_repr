import os
import numpy as np
import pandas as pd
import pickle as pkl
import tempfile
from prot_repr.utils import result_folder

def write_fasta(seqs, fname):
    with open(fname, "w") as f:
        for id_seq, seq in enumerate(seqs):
            f.write(">" + str(id_seq) + "\n" + seq + "\n")


def blast_transform(protein_sequences, ref_sequences=None, output_dir=None, n_jobs=-1):
    if output_dir is None:
        output_dir = tempfile.mkdtemp(dir=result_folder)

    if not os.path.exists(output_dir):
        print(f">>> Creating the output dir: {output_dir}")
        os.makedirs(output_dir)

    if ref_sequences is None:
        ref_sequences = protein_sequences
    n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs

    # some utilities files
    input_fasta_file = os.path.join(output_dir, 'inputs' + '.fasta')
    ref_fasta_file = os.path.join(output_dir, 'refs' + '.fasta')
    blast_results_file = os.path.join(output_dir, 'blast.out')
    ref_db_name = os.path.join(output_dir, 'refs_db')
    output_pkl = os.path.join(output_dir, 'out.pkl')

    # writing the fasta files containing both inputs and references
    write_fasta(protein_sequences, input_fasta_file)
    write_fasta(ref_sequences, ref_fasta_file)

    # create the reference database required by blast
    makedb_cmd = 'makeblastdb -in {} -dbtype prot -out {}'.format(ref_fasta_file, ref_db_name)
    if not os.path.exists(ref_db_name + '.phr'):
        os.system(makedb_cmd)

    # computing protein protein comparison using blast
    blast_cmd = "blastp -query {} -db {} -out {} -outfmt 10 -num_threads {}"
    blast_cmd = blast_cmd.format(input_fasta_file, ref_db_name, blast_results_file, n_jobs)
    os.system(blast_cmd)

    # get_results and output feature matrix
    data = pd.read_csv(blast_results_file, header=None, float_precision='high', engine='c')
    data.columns = ['i', 'j', 'identity', 'align_len', 'mismatches', 'gap_opens',
                    'i_start', 'i_end', 'j_start', 'j_end', 'evalue', 'bit_score']
    data = data[['i', 'j', 'evalue']]
    data = data.groupby(by=['i', 'j']).min().reset_index()
    n, m = len(protein_sequences), len(ref_sequences)
    arr = np.zeros((n, m), dtype=np.float)
    x = -np.log10(data.evalue)
    arr[data.i, data.j] = x
    for i, row in enumerate(arr):
        if not np.all(np.isfinite(row)):
            idx_j = np.where(~np.isfinite(row))[0]
            idx_i = np.array([i] * len(idx_j))
            arr[idx_i, idx_j] = np.max(row[np.isfinite(row)]) + 2
    arr = arr / arr.max(axis=1, keepdims=True)

    with open(output_pkl, 'wb') as fd:
        pkl.dump(arr, fd)
    return arr

if __name__ == '__main__':
    from prot_repr.datasets.loaders import load_fasta_proteins
    prots = load_fasta_proteins()# [:100]
    res = blast_transform(prots)
    print(res)