import os
import re
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import OPTICS
from sklearn.manifold.t_sne import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')


def preprocess(fname, run_dry=False):
    prot_fasta, bd_fname = os.path.join(DATA_FOLDER, 'proteins_seqs.fasta'), os.path.join(DATA_FOLDER, 'bindingdb.tsv')
    prot_infos = prot_fasta.replace('.fasta', '_infos.tsv')

    if run_dry:
        return bd_fname, prot_fasta, prot_infos

    columns_of_interest = ['Ligand SMILES', 'Target Name Assigned by Curator or DataSource',
                           'Target Source Organism According to Curator or DataSource',
                           'Ki (nM)', 'IC50 (nM)',
                           'Kd (nM)', 'EC50 (nM)', 'kon (M-1-s-1)',  'koff (s-1)',
                           'Number of Protein Chains in Target (>1 implies a multichain complex)',
                           'BindingDB Target Chain  Sequence',
                           'PDB ID(s) of Target Chain', 'UniProt (SwissProt) Primary ID of Target Chain',
                           ]
    table = pd.read_table(fname, sep='\t', usecols=columns_of_interest)
    readouts = ['ki', 'ic50', 'kd', 'ec50', 'kon', 'koff']
    table.columns = ['SMILES', 'target_name', 'target_organism'] + readouts +['n_chains_prot', 'target_seq', 'pdb_code', 'uniprot_id']
    table = table[table['n_chains_prot'] == 1]
    targets = table[['target_seq', 'target_name', 'target_organism', 'pdb_code', 'uniprot_id']].drop_duplicates()

    # fill in missing uniprot codes
    pdb_codes = [codes.split(',')[0] for codes in targets['pdb_code'].dropna()]
    uniprot_codes = ids_converter(pdb_codes, p_from="PDB_ID", p_to="ACC")
    targets.loc[~targets['pdb_code'].isna(), 'uniprot_id'] = uniprot_codes
    targets = targets.drop(columns=['pdb_code'])

    targets['ko_id'] = None
    targets['ontology'] = None
    uniprot_codes = targets['uniprot_id'].dropna().tolist()
    ko_codes = ids_converter(uniprot_codes, p_from="ACC", p_to="KO_ID")
    targets.loc[~targets['uniprot_id'].isna(), 'ko_id'] = ko_codes
    targets.loc[~targets['ko_id'].isna(), 'ontology'] = get_kegg_ontology(targets['ko_id'].dropna().tolist())
    print(targets.head())
    print(targets.count())

    list_temps = []
    for col in readouts:
        temp = table[['SMILES', 'target_seq', col]].dropna()
        temp.rename(columns={col:'y'}, inplace=True)
        temp['readout'] = col
        list_temps.append(temp)
    table = pd.concat(list_temps, axis=0)
    seqs = set(pd.unique(table['target_seq']).tolist())
    targets = targets[[(s in seqs) for s in targets['target_seq']]]
    ids = range(len(seqs))
    prot_mapping, prot_mapping_inv = dict(zip(ids, seqs)), dict(zip(seqs, ids))
    table['target_seq'] = [prot_mapping_inv[x] for x in table['target_seq']]
    targets['target_seq'] = [prot_mapping_inv[x] for x in targets['target_seq']]

    with open(prot_fasta, "w") as f:
        for id_seq, seq in prot_mapping.items():
            f.write(">" + str(id_seq) + "\n" + seq + "\n")
    table.to_csv(bd_fname, index=False, sep='\t')
    targets.to_csv(prot_infos, index=False, sep='\t')
    return bd_fname, prot_fasta, prot_infos


def gen_protein_similarity_graph(fasta_file, reuse_blast_results=False, run_dry=False):
    # expect fasta file to be labelled by numbers from 0 to n, where n is the number of proteins
    assert fasta_file.endswith('.fasta')
    outpkl = fasta_file[:-6] + '_graph.pkl'

    if run_dry:
        return outpkl
    nb_cpus = os.cpu_count()
    makedb_cmd = 'makeblastdb -in {} -dbtype prot -out {}'.format(fasta_file, fasta_file[:-6])
    outfile = os.path.join(os.path.dirname(fasta_file), 'blast_results.out')
    blast_cmd = "blastp -query {} -db {} -out {} -outfmt 10 -num_threads {}".format(fasta_file, fasta_file[:-6], outfile, nb_cpus)
    if not (reuse_blast_results and os.path.exists(fasta_file[:-6] + '.phr')):
        os.system(makedb_cmd)
        os.system(blast_cmd)

    data = pd.read_csv(outfile, header=None, float_precision='high', engine='c')
    data.columns = ['i', 'j', 'identity', 'align_len', 'mismatches', 'gap_opens',
                    'i_start', 'i_end', 'j_start', 'j_end', 'evalue', 'bit_score']
    data = data[['i', 'j', 'evalue']]
    data = data.groupby(by=['i', 'j']).min().reset_index()
    n = np.max(pd.unique(data.i)) + 1
    arr = np.zeros((n, n), dtype=np.float)
    x = -np.log10(data.evalue)
    arr[data.i, data.j] = x
    for i, row in enumerate(arr):
        if not np.all(np.isfinite(row)):
            idx_j = np.where(~np.isfinite(row))[0]
            idx_i = np.array([i]*len(idx_j))
            arr[idx_i, idx_j] = np.max(row[np.isfinite(row)]) + 2
    arr = arr / arr.max(axis=1, keepdims=True)

    with open(outpkl, 'wb') as fd:
        pkl.dump((list(range(n)), arr), fd)
    return outpkl


def cluster_proteins_by_family(prot_info_fname):
    data = pd.read_table(prot_info_fname)
    data = data.dropna()
    koids = data.ko_id.tolist()
    ontologies = data.ontology.tolist()

    data = ['\n'.join([line for line in ontology.split('\n') if koid not in line])
            for koid, ontology in zip(koids, ontologies)]
    # transformer = TfidfVectorizer()
    transformer = CountVectorizer(ngram_range=(1,3))
    x = transformer.fit_transform(data).todense()
    pca = PCA(n_components=100)
    x = pca.fit_transform(x)
    print('explained variance', pca.explained_variance_ratio_.sum())
    print(len(transformer.vocabulary_))


    clusters = OPTICS(cluster_method='dbscan').fit_predict(x)
    print(Counter(clusters))
    transformer = TSNE(n_components=2, n_iter_without_progress=10)
    x, y = transformer.fit_transform(x).T
    cmap = plt.get_cmap('jet', np.max(clusters)+2)
    cmap.set_under('gray')

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=clusters, s=10, cmap=cmap)
    outfile = os.path.join(os.path.dirname(prot_info_fname), 'protein_ontology_tsne_clusters.png')
    plt.savefig(outfile)
    plt.close()



    exit()



    families = []
    for koid, ontology in zip(koids, ontologies):
        kwords = ['Brite Hierarchies', 'Enzymes']
        for kword in kwords:
            start = ontology.find(kword)
            temp = ontology[start:]
            end = temp.find(koid)
            try:
                family = temp[:end].split('\n')[-2].strip()[5:].strip()
            except:
                print(ontology)
            families.append(family)
    # idxs = [i - 1 for i, line in enumerate(lines) if koid in line]
    # ontology = [lines[idx].strip() for idx in idxs]



def cluster_proteins_by_sim(prot_graph_fname):
    print('here')
    with open(prot_graph_fname, 'rb') as fd:
        nodes, adj_mat = pkl.load(fd)

    model = OPTICS(min_cluster_size=5, n_jobs=-1)
    clusters = model.fit_predict(adj_mat)
    print(Counter(clusters))

    transformer = eGTM()
    x, y = transformer.fit_transform(adj_mat).T
    cmap = plt.get_cmap('jet', np.max(clusters)+2)
    cmap.set_under('gray')

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=clusters, s=10, cmap=cmap)
    outfile = os.path.join(os.path.dirname(prot_graph_fname), 'protein_egtm_clusters.png')
    plt.savefig(outfile)
    plt.close()

    transformer = TSNE(n_components=2, n_iter_without_progress=10)
    x, y = transformer.fit_transform(adj_mat).T
    cmap = plt.get_cmap('jet', np.max(clusters)+2)
    cmap.set_under('gray')

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=clusters, s=10, cmap=cmap)
    outfile = os.path.join(os.path.dirname(prot_graph_fname), 'protein_tsne_clusters.png')
    plt.savefig(outfile)
    plt.close()



if __name__ == '__main__':
    fname = '/home/prtos/Downloads/BindingDB_All.tsv'
    bd_fname, prot_fasta, prot_infos = preprocess(fname, run_dry=True)
    cluster_proteins_by_family(prot_infos)
    exit()
    # bd_fname, prot_fasta, prot_infos = preprocess(fname, run_dry=True)
    # outpkl = gen_protein_similarity_graph(prot_fasta, reuse_blast_results=True, run_dry=True)
    # cluster_proteins(outpkl)
