import numpy as np
from matplotlib import pyplot as plt
from os import path as osp
from glob import glob
import cv2

path_to_root = '/Users/leonardbereska/myroot/'
path = osp.join(path_to_root, 'df')
imgs_df = glob(path+'/*')
imgs = imgs_df[0:100]

imgs_orig = [cv2.imread(i) for i in imgs]

# im = imgs[0]
# im_small = cv2.resize(im, (16, 16), interpolation=cv2.INTER_AREA)
imgs = [cv2.resize(i, (32, 32), interpolation=cv2.INTER_AREA) for i in imgs_orig]
imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
imgs = [np.resize(i, (32*32)) for i in imgs]
X = np.array(imgs)

# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(X)
# x = pca_result[:,0]
# y = pca_result[:,1]


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
tsne_results = tsne.fit_transform(X)

x = tsne_results[:,0]
y = tsne_results[:,1]


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

imgs_orig = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in imgs_orig]

ax = plt.axes()
imscatter(x, y, imgs_orig, ax, zoom=0.1)
plt.xticks([])
plt.yticks([])
plt.show(block=True)