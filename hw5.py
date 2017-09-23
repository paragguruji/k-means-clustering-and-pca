# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:38:11 2017

@author: Parag
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from time import time
from collections import Counter
from random import choice, sample
from matplotlib import pyplot as plt, patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


MAX_ITERATIONS = 50
K = [2, 4, 8, 16, 32]
B4_CHOSEN_K = [8, 4, 2]
C5_CHOSEN_K = [26, 18, 18]
EXPERIMENT_K = range(0, 101, 2)
P = 10


def timing(f):
    """Timer Decorator. Unused in final submission to avoid unwanted printing.
        paste @timing on the line above the def of function to be timed.
    """
    def wrap(*args, **kwargs):
        time1 = time()
        ret = f(*args, **kwargs)
        time2 = time()
        print '%s function took %0.3f ms' % \
              (f.func_name, (time2 - time1) * 1000.0)
        sys.stdout.flush()
        return ret
    return wrap


def read_data(datafile):
    with open(datafile, 'r') as f:
        data = np.array([[float(n) for n in line.split(',')] for line in f])
    return data


def read_raw_digits_data(datafile):
    data = {}
    with open(datafile, 'r') as f:
        for line in f:
            row = [float(n) for n in line.split(',')]
            data[int(row[0])] = {'class': int(row[1]),
                                 'raster': np.reshape(row[2:], (28, 28))}
    return data


def exploration1(data):
    _dir = os.path.join('output', 'exploration1')
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    classes = {k: 0 for k in set([data[_id]['class'] for _id in data])}
    while not all(classes.values()):
        _id = choice(data.keys())
        if classes[data[_id]['class']] == 0:
            classes[data[_id]['class']] = _id
    for c in classes:
        plt.xticks([])
        plt.yticks([])
        plt.title('Example {_id}: Label = {label}'.format(_id=str(classes[c]),
                  label=str(c)))
        plt.imshow(data[classes[c]]['raster'], cmap='gray')
        plt.gcf().savefig(os.path.join(_dir, str(c) + '.png'),
                          dpi=1200, bbox_inches='tight')
        plt.gcf().clear()


def scatter(D, L, imagefilepath):
    colors = ['red', 'blue', 'green', 'yellow', 'gray',
              'brown', 'orange', 'cyan', 'purple', 'magenta']
    Ls = sorted(list(set(L)))
    colormap = {Ls[i]: colors[i] for i in range(len(Ls))}
    mapper = np.vectorize(lambda x: colormap[x])
    plt.figure(figsize=(10, 8))
    plt.xticks([])
    plt.yticks([])
    plt.title("Digit-clusters - scatterplot of embeddings")
    plt.legend(handles=[mpatches.Patch(color=colormap[i], label=str(i))
                        for i in Ls],
               loc='best', title='Legend',
               fancybox=True, framealpha=0.5)
    plt.scatter(D[:, 0], D[:, 1], s=10, lw=0, c=mapper(L))
    plt.gcf().savefig(imagefilepath, dpi=1200, bbox_inches='tight')


def kmeans(D, K):
    C = np.array(sample(D, K))
    M_old = np.array([-1]*len(D))
    for _ in range(MAX_ITERATIONS):
        M = np.argmin(np.sqrt(((D - C[:, np.newaxis])**2).sum(axis=2)), axis=0)
        if all(M_old == M):
            break
        C = np.array([D[M == k].mean(axis=0) for k in range(C.shape[0])])
        M_old = M.copy()
    return C, M


def wc(D, M, C):
    return sum([((D[M == i] - C[i])**2).sum(axis=1).sum(axis=0)
                for i in set(M)])


def sc(D, M):
    def p2c(p, c):
        return np.sqrt(((D[M == c] - p)**2).sum(axis=1)).sum(axis=0)

    counts = Counter(M)
    if len(counts) < 2:
        return 0.0
    sum_si = 0.0
    for i in range(len(D)):
        if counts[M[i]] > 1:
            ai = (p2c(D[i], M[i]) / (counts[M[i]] - 1))
            bi = min([p2c(D[i], j) / counts[j] for j in counts if j != M[i]])
            sum_si += (bi - ai) / max(ai, bi)
    return sum_si / len(D)


def nmi(M, L):
    Ms = sorted(list(set(M)))
    Ls = sorted(list(set(L)))

    def count(m, l):
        try:
            return np.bincount(((M[:, np.newaxis][:, 0] == m) &
                                (L[:, np.newaxis][:, 0] == l)))[int(True)]
        except IndexError:
            return 0

    Nml = np.array([[count(m, l) for l in Ls] for m in Ms])
    Nm = Nml.sum(axis=1).astype(float)
    Nl = Nml.sum(axis=0).astype(float)
    N = float(Nml.sum())

    def calc(m, l):
        if Nml[m][l] == 0:
            return 0.0
        return ((Nml[m][l] / N) * np.log2((N * Nml[m][l]) / (Nm[m] * Nl[l])))

    I = sum([sum([calc(Ms.index(m), Ls.index(l)) for l in Ls]) for m in Ms])
    Hm = ((Nm / N) * np.log2(Nm / N)).sum()
    Hl = ((Nl / N) * np.log2(Nl / N)).sum()
    return (2.0 * I) / (-(Hm + Hl))


def split_data(D):
    L = D[:, 1].astype(int)
    D1 = D.copy()
    D2 = D[reduce(lambda x, y: x | y, [(L == i) for i in {2, 4, 6, 7}])]
    D3 = D[reduce(lambda x, y: x | y, [(L == i) for i in {6, 7}])]
    return D1, D2, D3


def b1(data, _dir=None):
    if not _dir:
        _dir = os.path.join('output', 'analysis')
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    D = split_data(data)
    result = np.array([-1.0]*35).reshape(5, 7)
    for k in range(len(K)):
        result[k][0] = K[k]
        for d in range(len(D)):
            C, M = kmeans(D[d][:, 2:], K[k])
            result[k][d + 1] = wc(D[d][:, 2:], M, C)
            result[k][d + 1 + len(D)] = sc(D[d][:, 2:], M)
    df = pd.DataFrame(result, columns=['K', 'Data_1_WC-SSD',
                                       'Data_2_WC-SSD',
                                       'Data_3_WC-SSD',
                                       'Data_1_SC', 'Data_2_SC',
                                       'Data_3_SC'])
    df.to_csv(os.path.join(_dir, 'B1.csv'), index=False)


def plot_b1(_dir=None):
    if not _dir:
        _dir = os.path.join('output', 'analysis')
    df = pd.read_csv(os.path.join(_dir, 'B1.csv'))
    p1 = df.plot(x=df.columns.values[0],
                 y=df.columns.values[1:4],
                 title="Analysis B.1: Variation in WC-SSD with K")
    p1.set_xticks(df[df.columns.values[0]], minor=True)
    p1.grid(which='both', linestyle='dotted', alpha=0.5)
    p1.get_figure().savefig(os.path.join(_dir, 'B1WC.png'))
    p2 = df.plot(x=df.columns.values[0],
                 y=df.columns.values[4:],
                 title="Analysis B.1: Variation in SC with K")
    p2.set_xticks(df[df.columns.values[0]], minor=True)
    p2.grid(which='both', linestyle='dotted', alpha=0.5)
    p2.get_figure().savefig(os.path.join(_dir, 'B1SC.png'))


def b3(data, _dir=None):
    if not _dir:
        _dir = os.path.join('output', 'analysis')
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    D = split_data(data)
    result = np.array([-1.0]*350).reshape(50, 7)
    for i in range(len(K) * 10):
        k = i // 10
        result[i][0] = K[k]
        for d in range(len(D)):
            C, M = kmeans(D[d][:, 2:], K[k])
            result[i][d + 1] = wc(D[d][:, 2:], M, C)
            result[i][d + 1 + len(D)] = sc(D[d][:, 2:], M)
    df = pd.DataFrame(result, columns=['K', 'Data_1_WC-SSD',
                                       'Data_2_WC-SSD',
                                       'Data_3_WC-SSD',
                                       'Data_1_SC', 'Data_2_SC',
                                       'Data_3_SC'])
    df.to_csv(os.path.join(_dir, 'B3.csv'), index=False)


def plot_b3(_dir=None):
    if not _dir:
        _dir = os.path.join('output', 'analysis')
    df = pd.read_csv(os.path.join(_dir, 'B3.csv'))
    gb = df.groupby('K')
    mdf = gb.mean().reset_index()
    vdf = gb.std().reset_index()
    mdf.to_csv(os.path.join(_dir, 'B3_mean.csv'), index=False)
    vdf.to_csv(os.path.join(_dir, 'B3_var.csv'), index=False)
    colors = ['magenta', 'blue', 'green', 'red', 'black']
    c = list(mdf.columns)
    _title = "Analysis B.3: Mean WC-SSD by K (Errorbars: Std. Dev.)"
    for j in range(1, 4):
        plt.errorbar(x=mdf[c[0]],
                     y=mdf[c[j]],
                     yerr=vdf[c[j]],
                     color=colors[j],
                     label=c[j])
    plt.xlabel('K')
    plt.ylabel('Mean WC-SSD')
    plt.title(_title)
    plt.legend(loc='best', title='Legend',
               fancybox=True, framealpha=0.5)
    plt.xticks(mdf[c[0]])
    plt.grid(which='both', linestyle='dotted', alpha=0.5)
    plt.gcf().savefig(os.path.join(_dir, 'B3WC.png'),
                      dpi=1200, bbox_inches='tight')
    plt.gcf().clear()
    _title = "Analysis B.3: Mean SC by K (Errorbars: Std. Dev.)"
    for j in range(4, 7):
        plt.errorbar(x=mdf[c[0]],
                     y=mdf[c[j]],
                     yerr=vdf[c[j]],
                     color=colors[j - 3],
                     label=c[j])
    plt.xlabel('K')
    plt.ylabel('Mean SC')
    plt.title(_title)
    plt.legend(loc='best', title='Legend',
               fancybox=True, framealpha=0.5)
    plt.xticks(mdf[c[0]])
    plt.grid(which='both', linestyle='dotted', alpha=0.5)
    plt.gcf().savefig(os.path.join(_dir, 'B3SC.png'),
                      dpi=1200, bbox_inches='tight')
    plt.gcf().clear()


def b4(data, K, _dir=None):
    if not _dir:
        _dir = os.path.join('output', 'analysis')
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    dataset = split_data(data)
    NMI = np.array([-1.0]*len(dataset))
    for i in range(len(dataset)):
        D, L = dataset[i][:, 2:], dataset[i][:, 1].astype(int)
        C, M = kmeans(D, K[i])
        NMI[i] = nmi(M, L)
        idx = range(len(dataset[i]))
        chosen = sample(idx, 1000)
        filename = 'B4ClusterData' + str(i + 1) + 'K' + str(K[i]) + '.png'
        scatter(D[chosen, :], M[chosen], os.path.join(_dir, filename))
    df = pd.DataFrame(NMI, columns=['NMI'],
                      index=['Data' + str(i + 1) + '_K=' + str(K[i])
                             for i in range(len(K))])
    ax = df.plot.bar(alpha=0.5, table=True, xticks=[], ylim=(0.0, 1.0),
                     title='Analysis B.4: NMI for chosen K')
    ax.set_xticklabels([])
    ax.get_figure().savefig(os.path.join(_dir, 'B4NMI.png'),
                            dpi=1200, bbox_inches='tight')


def stratified_sample(arr, k):
    a = arr[arr[:, 1].argsort()]
    strata = np.split(a, np.where(np.diff(a[:, 1]))[0] + 1)
    samples = [sample(stratum, k) for stratum in strata]
    return np.vstack(samples)


def plot_dendogram(X, method, xidx, dendogram_imagefile):
    Z = linkage(X, method=method)
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram - ' + method + ' linkage')
    plt.xticks(xidx)
    plt.xlabel('sample image id')
    plt.ylabel('Euclidean Distance')
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.,)
    plt.gcf().savefig(dendogram_imagefile,
                      dpi=1200, bbox_inches='tight')
    plt.gcf().clear()
    return Z


def c123(embedding_data):
    _dir = os.path.join('output', 'comparison')
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    data = stratified_sample(embedding_data, 10)
    X, L = data[:, 2:], data[:, 1].astype(int)
    Z1 = plot_dendogram(X, 'single', data[:, 0].astype(int),
                        os.path.join(_dir, 'C1DendogramSingleLinkage.png'))
    Z2 = plot_dendogram(X, 'complete', data[:, 0].astype(int),
                        os.path.join(_dir, 'C2DendogramCompleteLinkage.png'))
    Z3 = plot_dendogram(X, 'average', data[:, 0].astype(int),
                        os.path.join(_dir, 'C2DendogramAverageLinkage.png'))
    Ks = EXPERIMENT_K
    Zs = [Z1, Z2, Z3]
    result = np.array([-1.0]*(len(Ks) * 7)).reshape(len(Ks), 7)
    for k in range(len(Ks)):
        result[k][0] = min(Ks[k], len(data))
        for z in range(len(Zs)):
            Mkz = fcluster(Zs[z], Ks[k], criterion='maxclust') - 1
            Ckz = np.array([X[Mkz == m].mean(axis=0) for m in set(Mkz)])
            result[k][z + 1] = wc(X, Mkz, Ckz)
            if result[k][0] == len(data):
                result[k][z + 1 + len(Zs)] = 0.0
            else:
                result[k][z + 1 + len(Zs)] = sc(X, Mkz)
    df = pd.DataFrame(result, columns=['K',
                                       'Single_Linkage_WC-SSD',
                                       'Complete_Linkage_WC-SSD',
                                       'Average_Linkage_WC-SSD',
                                       'Single_Linkage_SC',
                                       'Complete_Linkage_SC',
                                       'Average_Linkage_SC'])
    df.to_csv(os.path.join(_dir, 'C3.csv'), index=False)
    np.savetxt(os.path.join(_dir, 'C5ZSingle.txt'), Z1)
    np.savetxt(os.path.join(_dir, 'C5ZComplete.txt'), Z2)
    np.savetxt(os.path.join(_dir, 'C5ZAverage.txt'), Z3)
    np.savetxt(os.path.join(_dir, 'C5ImageLabels.txt'), L)


def plot_c3():
    _dir = os.path.join('output', 'comparison')
    df = pd.read_csv(os.path.join(_dir, 'C3.csv'))
    p1 = df.plot(x=df.columns.values[0],
                 y=df.columns.values[1:4],
                 title="Comparison C.3: Variation in WC-SSD with K")
    p1.set_xticks(df[df.columns.values[0]], minor=True)
    p1.grid(which='both', linestyle='dotted', alpha=0.5)
    p1.get_figure().savefig(os.path.join(_dir, 'C3WC.png'))
    p2 = df.plot(x=df.columns.values[0],
                 y=df.columns.values[4:],
                 title="Comparison C.3: Variation in SC with K")
    p2.set_xticks(df[df.columns.values[0]], minor=True)
    p2.grid(which='both', linestyle='dotted', alpha=0.5)
    p2.get_figure().savefig(os.path.join(_dir, 'C3SC.png'))


def c5(Ks):
    _dir = os.path.join('output', 'comparison')
    methods = ['Single', 'Complete', 'Average']
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    Zs = [np.loadtxt(os.path.join(_dir, 'C5Z' + method + '.txt'))
          for method in methods]
    L = np.loadtxt(os.path.join(_dir, 'C5ImageLabels.txt')).astype(int)
    NMI = [nmi(fcluster(Zs[i], Ks[i], criterion='maxclust') - 1, L)
           for i in range(len(Zs))]
    df = pd.DataFrame(NMI, columns=['NMI'],
                      index=[methods[i] + 'Linkage_K=' + str(Ks[i])
                             for i in range(len(Ks))])
    ax = df.plot.bar(alpha=0.5, table=True, xticks=[], ylim=(0.0, 1.0),
                     title='Comparison C.5: NMI for chosen K')
    ax.set_xticklabels([])
    ax.get_figure().savefig(os.path.join(_dir, 'C5NMI.png'),
                            dpi=1200, bbox_inches='tight')


def bonus(data):
    _dir = os.path.join('output', 'bonus')
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    X, L = data[:, 2:], data[:, 1]
    X_std = X - X.mean(axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    idx = eig_vals.argsort()[::-1]
    PCV = eig_vecs[:, idx][:, :P]
    for i in range(P):
        plt.xticks([])
        plt.yticks([])
        plt.title('Eigen Vector: ' + str(i + 1))
        plt.imshow(PCV[:, i].reshape(28, 28), cmap='gray')
        plt.gcf().savefig(os.path.join(_dir, 'eigen_' + str(i + 1) + '.png'),
                          dpi=1200, bbox_inches='tight')
        plt.gcf().clear()
    reduced_X = X_std.dot(PCV)
    idx = range(len(X))
    chosen = sample(idx, 1000)
    filename = 'Bonus3ClustersByPCA.png'
    scatter(reduced_X[chosen, :2], L[chosen], os.path.join(_dir, filename))
    target_dir = os.path.join('output', 'bonus', 'analysis')
    reduced_data = np.hstack((data[:, 0].reshape(data.shape[0], 1),
                              L.reshape(data.shape[0], 1), reduced_X))
    b1(reduced_data, _dir=target_dir)
    plot_b1(_dir=target_dir)
    df = pd.read_csv(os.path.join(target_dir, 'B1.csv'))
    colors = ['blue', 'green', 'red']
    for i in range(1, 4):
        p1 = df.plot(x=df.columns.values[0],
                     y=df.columns.values[i],
                     color=colors[i - 1],
                     title="Analysis (PCA) B.1: Variation in WC-SSD with K")
        p1.legend(borderaxespad=0.)
        p1.set_xticks(df[df.columns.values[0]], minor=True)
        p1.grid(which='both', linestyle='dotted', alpha=0.5)
        p1.get_figure().savefig(os.path.join(target_dir,
                                'B1Data' + str(i) + 'WC.png'))
        p1.get_figure().clear()
        p2 = df.plot(x=df.columns.values[0],
                     y=df.columns.values[(i + 3)],
                     color=colors[i - 1],
                     title="Analysis (PCA) B.1: Variation in SC with K")
        p2.legend(borderaxespad=0.)
        p2.set_xticks(df[df.columns.values[0]], minor=True)
        p2.grid(which='both', linestyle='dotted', alpha=0.5)
        p2.get_figure().savefig(os.path.join(target_dir,
                                'B1Data' + str(i) + 'SC.png'))
        p2.get_figure().clear()
    b4(reduced_data, K=B4_CHOSEN_K, _dir=target_dir)


if __name__ == '__main__':
    """Process commandline arguments and make calls to appropriate functions
    """
    parser = argparse.ArgumentParser(
                    description='CS 573 Data Mining HW5 Clustering',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataFilename',
                        help='file-path of embeddings input data')
    parser.add_argument('kValue',
                        metavar='kValue',
                        type=int,
                        help='Number of clusters to use')
    parser.add_argument('-r', '--rawDigitsFilename',
                        metavar='rawDigitsFilename',
                        help='file-path of raw digits input data')
    parser.add_argument('-x', '--exploration',
                        metavar='exploration',
                        type=int,
                        choices=[1, 2],
                        default=None,
                        help="Run exploration question 1 or 2 as chosen.")
    parser.add_argument('-a', '--analysis',
                        metavar='analysis',
                        type=int,
                        choices=[1, 2, 3],
                        default=None,
                        help="Solutions to Part B - Analysis of k-means: \
                        1. Question B.1 \
                        2. Question B.3 \
                        3. Question B.4")
    parser.add_argument('-c', '--comparison',
                        metavar='comparison',
                        type=int,
                        choices=[1, 2],
                        default=None,
                        help="Solutions to Part C - \
                        Comparison to hierarchical clustering: \
                        1. Question C.1, C.2 and C.3 \
                        2. Question C.5")
    parser.add_argument('-p', '--pca',
                        action='store_true',
                        help="Solutions to Bonus - PCA")
    args = parser.parse_args()

    embedding_data = read_data(args.dataFilename)
    if args.exploration:
        if args.exploration == 1:
            if not args.rawDigitsFilename:
                print "No raw digit file specified. Specify option -r"
                sys.exit(0)
            else:
                raw_data = read_raw_digits_data(args.rawDigitsFilename)
                exploration1(raw_data)
        elif args.exploration == 2:
            idx = range(len(embedding_data))
            chosen = sample(idx, 1000)
            scatter(embedding_data[:, [2, 3]][chosen, :],
                    embedding_data[:, 1][chosen],
                    os.path.join('output', 'exploration2/digit-clusters.png'))
    elif args.analysis:
        _dir = os.path.join('output', 'analysis')
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        if args.analysis == 1:
            b1(embedding_data)
            plot_b1()
        elif args.analysis == 2:
            b3(embedding_data)
            plot_b3()
        elif args.analysis == 3:
            b4(embedding_data, B4_CHOSEN_K)
    elif args.comparison:
        if args.comparison == 1:
            c123(embedding_data)
            plot_c3()
        elif args.comparison == 2:
            c5(C5_CHOSEN_K)
    elif args.pca:
        if not args.rawDigitsFilename:
                print "No raw digit file specified. Specify option -r"
                sys.exit(0)
        else:
            raw_data = read_data(args.rawDigitsFilename)
            bonus(raw_data)
    else:
        D, L = embedding_data[:, -2:], embedding_data[:, 1].astype(int)
        C, M = kmeans(D, args.kValue)
        print "WC-SSD", wc(D, M, C)
        print "SC", sc(D, M)
        print "NMI", nmi(M, L)


#    b3(reduced_data, _dir=target_dir)
#    df = pd.read_csv(os.path.join(target_dir, 'B3.csv'))
#    gb = df.groupby('K')
#    mdf = gb.mean().reset_index()
#    vdf = gb.std().reset_index()
#    mdf.to_csv(os.path.join(target_dir, 'B3_mean.csv'), index=False)
#    vdf.to_csv(os.path.join(target_dir, 'B3_var.csv'), index=False)
#    colors = ['magenta', 'red', 'blue', 'green', 'black']
#    c = list(mdf.columns)
#    for j in range(1, 4):
#        plt.errorbar(x=mdf[c[0]],
#                     y=mdf[c[j]],
#                     yerr=vdf[c[j]],
#                     color=colors[j],
#                     label=c[j])
#        _title = "Analysis B.3 Data " + str(j) + \
#            " : Mean WC-SSD by K (Errorbars: Variance)"
#        plt.xlabel('K')
#        plt.ylabel('Mean WC-SSD')
#        plt.title(_title)
#        plt.legend(loc='best', title='Legend',
#                   fancybox=True, framealpha=0.5)
#        plt.xticks(mdf[c[0]])
#        plt.grid(which='both', linestyle='dotted', alpha=0.5)
#        plt.gcf().savefig(os.path.join(target_dir,
#                                       'B3Data' + str(j) + 'WC.png'),
#                          dpi=1200, bbox_inches='tight')
#        plt.gcf().clear()
#    for j in range(4, 7):
#        plt.errorbar(x=mdf[c[0]],
#                     y=mdf[c[j]],
#                     yerr=vdf[c[j]],
#                     color=colors[j - 3],
#                     label=c[j])
#        _title = "Analysis B.3 Data " + str(j - 3) + \
#            " : Mean SC by K (Errorbars: Variance)"
#        plt.xlabel('K')
#        plt.ylabel('Mean SC')
#        plt.title(_title)
#        plt.legend(loc='best', title='Legend',
#                   fancybox=True, framealpha=0.5)
#        plt.xticks(mdf[c[0]])
#        plt.grid(linestyle='dotted', alpha=0.5)
#        plt.gcf().savefig(os.path.join(target_dir,
#                                       'B3Data' + str(j - 3) + 'SC.png'),
#                          dpi=1200, bbox_inches='tight')
#        plt.gcf().clear()
