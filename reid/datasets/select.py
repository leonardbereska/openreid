from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class Select(Dataset):

    def __init__(self, root, from_dir1, from_dir2, to_dir, num_eval, make_test=None, split_id=0, num_val=100, download=True):
        super(Select, self).__init__(root, split_id=split_id)
        self.n_pid = num_eval
        self.from_dir1 = from_dir1
        self.from_dir2 = from_dir2
        self.to_dir = to_dir
        self.root = osp.join(root, 'datasets', self.to_dir)
        self.root_orig = root
        self.build_test = make_test
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " + "You can use download=True to download it.")
        if self.build_test:
            num_val = 0
        self.load(num_val)  # no validation set for testy

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        print('Make dataset')
        from glob import glob
        from scipy.misc import imsave, imread

        mkdir_if_missing(self.root)

        dir_images = osp.join(self.root, 'images')
        mkdir_if_missing(dir_images)

        dir1 = osp.join(self.root_orig, 'raw', self.from_dir1)
        img1 = sorted(glob(dir1 + '/*'))
        dir2 = osp.join(self.root_orig, 'raw', self.from_dir2)
        img2 = sorted(glob(dir2 + '/*'))

        def id(img, full=False):
            id_ = img.split('/')[-1].split('.')[0]
            if not full:
                id_ = id_.split('_')[0]
            return id_

        def cluster_by_id(img_list):
            ids = [id(i) for i in img_list]
            return [[img_list[i] for i in range(len(ids)) if ids[i] == l] for l in set(ids)]

        A = set([id(i) for i in img1])
        B = set([id(i) for i in img2])
        img2 = [i for i in img2 if id(i) in A]

        img1 = cluster_by_id(img1)
        img2 = cluster_by_id(img2)
        assert len(img1) == len(img2)

        all_imgs = zip(img1, img2)

        identities = []
        for pid, (gt, pa) in enumerate(all_imgs):

            images = []
            for i, g in enumerate(gt):
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, i, 0)
                imsave(osp.join(dir_images, fname), imread(g))
                images.append([fname])
            for j, p in enumerate(pa):
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, i+j+1, 1)
                imsave(osp.join(dir_images, fname), imread(p))
                images.append([fname])

            identities.append(images)
            if pid % 100 == 0:
                print('ID {}/{}'.format(pid, len(img1)))
            if pid == self.n_pid-1:
                break

        # Save meta information into a json file
        meta = {'name': 'deepfashion', 'shot': 'single', 'num_cameras': 2, 'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Randomly create ten training and test split
        num = len(identities)
        splits = []
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            if self.build_test is None:
                trainval_pids = sorted(pids[:num // 2])
                test_pids = sorted(pids[num // 2:])
            elif self.build_test:
                trainval_pids = []
                test_pids = sorted(pids)
            else:
                trainval_pids = sorted(pids)
                test_pids = []

            split = {'trainval': trainval_pids, 'query': test_pids, 'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
