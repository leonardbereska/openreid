import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from os import path as osp
from glob import glob
import cv2
from reid.evaluators import extract_features
from sklearn.manifold import TSNE


def label_to_img(labels):
    imgs = labels  # todo implement
    return imgs


def get_test_features():
    path_to_root = '/Users/leonardbereska/myroot/'
    path = osp.join(path_to_root, 'df')
    imgs_df = glob(path + '/*')
    imgs = imgs_df[0:100]

    imgs_orig = [cv2.imread(i) for i in imgs]
    imgs_orig = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in imgs_orig]

    imgs = [cv2.resize(i, (32, 32), interpolation=cv2.INTER_AREA) for i in imgs_orig]
    imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
    imgs = [np.resize(i, (32 * 32)) for i in imgs]
    X = np.array(imgs)
    return X, imgs_orig


def imscatter(x, y, imgs, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    imgs = [OffsetImage(i, zoom=zoom) for i in imgs]
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im in zip(x, y, imgs):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def plot_tsne(features, images, perplexity=50):

    # features, images = get_test_features()

    # from sklearn.decomposition import PCA
    #
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(X)
    # y = pca_result[:,1]
    # x = pca_result[:,0]

    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=5000)
    tsne_results = tsne.fit_transform(features)

    x = tsne_results[:,0]
    y = tsne_results[:,1]

    ax = plt.axes()
    imscatter(x, y, images, ax, zoom=0.1)
    plt.xticks([])
    plt.yticks([])
    plt.show(block=True)


class Visualize(object):
    def __init__(self, model):
        super(Visualize, self).__init__()
        self.model = model

    def visualize(self, data_loader, query, gallery, metric=None):
        features, labels = extract_features(self.model, data_loader, print_freq=1000)
        print(features)
        print(labels)
        # distmat = pairwise_distance(features, query, gallery, metric=metric)
        # return evaluate_all(distmat, query=query, gallery=gallery)
        imgs = label_to_img(labels)
        plot_tsne(features, imgs)
