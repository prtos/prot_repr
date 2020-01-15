import os
import random
import numpy as np
from prot_repr.transformers.utils import aa_vocab, aa_blosum
from numpy.lib.stride_tricks import as_strided
from joblib import Parallel, delayed
from prot_repr.datasets.loaders import load_fasta_proteins
from sklearn.decomposition import PCA


def get_default_sigma_c():
    mat = np.array(list(aa_blosum.values()))
    temp = np.sqrt(((mat[:, None, :] - mat[None, :, :])**2).sum(-1))
    return np.std(temp)/2


DEFAULT_SIGMA_C = get_default_sigma_c()
DEFAULT_SIGMA_P = 1


def rolling_window(a, window=1):
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return as_strided(a, shape=shape, strides=strides)


def gs_kernel(str1, str2, sigma_p=1., sigma_c=1., L=1):
    """
    Implementation of the GS (i.e. Generic String) kernel.

    The proposed kernel is designed for small bio-molecules (such as peptides) and pseudo-sequences
    of binding interfaces. The GS kernel is an elegant generalization of eight known kernels for local signals.
    Despite the richness of this new kernel, we provide a simple and efficient dynamic programming
    algorithm for its exact computation.

    str1, str2 -- Both sequence of amino acids to compare.
    sigma_p -- Control the position uncertainty of the sub-strings of str1 and str2.
    sigma_c -- Control the influence of amino acids properties (physico-chemical for example),
               can by any kind of properties.
    L -- Length of substrings
    """

    x1 = np.array([aa_blosum[i] for i in str1])
    x2 = np.array([aa_blosum[i] for i in str2])
    d = x1.shape[1]

    p1, p2 = np.arange(len(str1)), np.arange(len(str2))
    p_dist = np.exp(-(p1[:, None] - p2[None, :]) / 2*(sigma_p**2))

    res = 0
    for l in range(1, L):
        xx1 = rolling_window(x1, l).reshape((-1, L*d))
        xx2 = rolling_window(x2, l).reshape((-1, L*d))
        xx_dist = np.exp(-((xx1[:, None, :] - xx2[None, :, :]) ** 2).sum(-1) / 2 * (sigma_c ** 2))
        n, m = xx_dist.shape
        pp_dist = p_dist[:n, :m]
        res = res + (pp_dist * xx_dist).sum()
    return res


def gs_gram_matrix(X, Y, sigma_p=1.0, sigma_c=1.0, max_kmer=2, normalize_matrix=True, n_jobs=-1):
    """
    Return the gram matrix K using the GG kernel such that K_i,j = k(x_i, y_j).
    If X == Y, M is a squared positive semi-definite matrix.
    When training call this function with X == Y, during the testing phase
    call this function with X containing the training examples and Y containing
    the testing examples.

    We recommend to normalize both the training and testing gram matrix.
    If the training matrix is normalized so should be the testing matrix.
    If the training matrix is un-normalized so should be the testing matrix.

    X -- List of examples, can be any kind of amino acid sequence
        (peptides, small protein, binding interface pseudo-sequence, ...)

    Y -- Second list of examples, can be the same as X for training or
        the testing examples.

    sigma_p -- Float value for \sigma_p. Control the position uncertainty of
        sub-strings in the GS kernel.
        Values in [0.0, 16.0] seem to empirically work well.

    sigma_c -- Float value for \sigma_c. Control the trade of between the
        amino-acids properties and the dirac delta.
        Values in [0.0, 16.0] seem to empirically work well.

    max_kmer -- Length of the sub-strings. Should smaller or equal that the sequences in X or Y.
        Values in [1,6] seem to empirically work well.

    normalize_matrix -- Normalize the gram matrix. We recommend to normalize.
    """

    # X and Y should be np.array
    X = np.array(X)
    Y = np.array(Y)
    n, m = len(X), len(Y)

    if X.shape == Y.shape and np.all(X == Y):
        K = np.zeros((n, m))
        # Fill the symmetric matrix
        temp = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(gs_kernel)(X[i], X[j], sigma_p, sigma_c, max_kmer)
            for i in range(len(X)) for j in range(i+1))
        n = 0
        for i in range(n):
            K[i, :i] = temp[n:n+i+1]
            n += i+1
        K = K + K.T
        K[np.arange(n), np.arange(n)] /= 2

        if normalize_matrix:
            normX = np.sqrt(K.diagonal())
            K = ((K / normX).T / normX).T
    else:
        # Fill the non-symetric possibly rectangle matrix
        temp = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(gs_kernel)(X[i], Y[j], sigma_p, sigma_c, max_kmer)
            for i in range(len(X)) for j in range(len(Y)))
        K = np.array(temp).reshape((n, m))

        if normalize_matrix:
            normX = np.sqrt([gs_kernel(x, x, sigma_p, sigma_c, max_kmer) for x in X])
            normY = np.sqrt([gs_kernel(y, y, sigma_p, sigma_c, max_kmer) for y in Y])
            K = ((K / normY).T / normX).T

    return K


def get_kmers(seq, k):
    n = len(seq)
    if n < k: return []
    return [seq[i:i+k] for i in range(n-k)]


def gs_based_features(protein_sequences, k=31, sigma_c=DEFAULT_SIGMA_C, sigma_p=DEFAULT_SIGMA_P):
    prot_kmers = [get_kmers(prot, k) for prot in protein_sequences]
    n_prot, l_max = len(prot_kmers), max([len(x) for x in prot_kmers])
    x = list(set(sum(prot_kmers, [])))
    print('nb_kmers: ', len(x))
    random.shuffle(x)
    n_samples = 1000
    y = x[:n_samples]

    kmers_dict = {kmer: i for i, kmer in enumerate(x)}

    K = gs_gram_matrix(x, y, sigma_c, sigma_p, 3, n_jobs=-1)
    res = [[K[kmers_dict[seq_kmers[i]]] if i < len(seq_kmers) else [0]*n_samples for i in range(l_max)]
            for seq_kmers in prot_kmers]

    res = np.array(res).reshape(n_prot, -1)
    res = PCA().fit_transform(res)
    return res


if __name__ == '__main__':
    prots = load_fasta_proteins()
    print(len(prots))

    res = gs_based_features(prots)


