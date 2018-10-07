import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from os import path as osp
from glob import glob
import cv2
from reid.evaluators import extract_features
from sklearn.manifold import TSNE
import os


def read_imgs(labels, path_to_data):
    imgs = []
    assert osp.exists(path_to_data)
    for img_path in labels:
        full_path = osp.join(path_to_data,'images', img_path)
        assert osp.exists(full_path)
        img = cv2.imread(full_path)
        imgs.append(img)
    imgs = np.array(imgs)
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


def imscatter(x, y, imgs, ax=None, zoom=1, color=None):
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


def plot_tsne(features, images, labels, perplexity, n_points):

    # features, images = get_test_features()

    # from sklearn.decomposition import PCA
    #
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(X)
    # y = pca_result[:,1]
    # x = pca_result[:,0]
    # n_points = 100
    # perplexity = 5
    # features_ = features
    # images_ = images
    # labels_ = labels
    features = features[0:n_points]
    images = images[0:n_points]
    labels = labels[0:n_points]

    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=5000)
    tsne_results = tsne.fit_transform(features)

    x = tsne_results[:,0]
    y = tsne_results[:,1]
    for i in images:
        cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    ax = plt.axes()
    imscatter(x, y, images, ax, zoom=0.05, color=labels)
    plt.xticks([])
    plt.yticks([])
    # plt.show(block=True)
    plot_dir = osp.join(osp.abspath('.'), 'plots')
    if not osp.exists(plot_dir):
        os.mkdir(plot_dir)
    png = 'png'
    save_path = osp.join(plot_dir, 'tsne-pp{}-n{}.{}'.format(perplexity, n_points, png))
    plt.savefig(save_path, format=png, dpi=1000)


class Visualize(object):
    def __init__(self, model):
        super(Visualize, self).__init__()
        self.model = model

    def visualize(self, data_loader, data_path, n_neighbors, n_points):
        extracted, _ = extract_features(self.model, data_loader, print_freq=1000)
        labels = []
        features = []
        for key in extracted.keys():
            label = int(key.split('_')[0])
            feature = extracted[key].numpy()
            labels.append(label)
            features.append(feature)

        features = np.array(features)
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=1)
        img_paths = list(extracted.keys())

        imgs = read_imgs(img_paths, data_path)
        perplexities = [5, 10, 20]
        n_points = [10, 50, 100, 500, 1000]
        for p in perplexities:
            for n in n_points:
                plot_tsne(features, imgs, labels, perplexity=p, n_points=n)


