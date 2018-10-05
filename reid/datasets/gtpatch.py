from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class GtPatch(Dataset):
    url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'
    md5 = '1c2d9fc1cc800332567a0da25a1ce68c'

    def __init__(self, root, split_id=0, num_val=0, download=True):
        super(GtPatch, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " + "You can use download=True to download it.")

        self.load(num_val)  # no validation set for testy

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import hashlib
        from glob import glob
        from scipy.misc import imsave, imread
        from six.moves import urllib
        from zipfile import ZipFile

        # raw_dir = osp.join(self.root, 'raw')
        # mkdir_if_missing(raw_dir)

        # get lists of img: gen - patch, gt - patch
        dir_images = osp.join(self.root, 'images')
        mkdir_if_missing(dir_images)
        dir_gt = osp.join(self.root, 'gt')
        img_gt = sorted(glob(dir_gt + '/*'))
        dir_patch = osp.join(self.root, 'patch')
        img_patch = sorted(glob(dir_patch + '/*'))

        # for i_gt, i_pa in zip(dir_gt, dir_patch):

        all_imgs = zip(img_gt, img_patch)

        # def get_id(img):
        #     return int(img.split('/')[-1].split('.')[0])  # .split('_')[0])

        # all_id = [get_id(i) for i, _ in all_imgs]
        # from collections import Counter
        # count_id = Counter(all_id)
        # n_id = len(count_id)
        #
        # all_pid_all = [[] for i in range(max(all_id))]
        # for img in all_imgs:
        #     all_pid_all[get_id(img)-1].append(img)
        #
        # all_pid = [pid for pid in all_pid_all if pid]  #

        identities = []
        for pid, (gt, pa) in enumerate(all_imgs):
            # assert len(imgs) == count_id[pid]
            images = []
            # pid = get_id(gt)
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
            imsave(osp.join(dir_images, fname), imread(gt))
            images.append([fname])
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 1, 0)
            imsave(osp.join(dir_images, fname), imread(pa))
            images.append([fname])
            # fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 2, 0)
            # imsave(osp.join(dir_images, fname), imread(imgs[2]))
            # images.append([fname])
            # fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 3, 0)
            # imsave(osp.join(dir_images, fname), imread(imgs[3]))
            # images.append([fname])

            identities.append(images)
            if pid % 100 == 0:
                print('ID {}/{}'.format(pid, len(img_gt)))

        # cameras = [sorted(glob(osp.join(exdir, 'cam_a', '*.bmp'))),
        #            sorted(glob(osp.join(exdir, 'cam_b', '*.bmp')))]
        # assert len(cameras[0]) == len(cameras[1])
        # identities = []
        # for pid, (cam1, cam2) in enumerate(zip(*cameras)):
        #     images = []
        #     view-0
        #     fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
        #     imsave(osp.join(dir_images, fname), imread(cam1))
        #     images.append([fname])
        #     view-1
        #     fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 1, 0)
        #     imsave(osp.join(dir_images, fname), imread(cam2))
        #     images.append([fname])
        #     identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'deepfashion', 'shot': 'single', 'num_cameras': 4, 'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Randomly create ten training and test split
        num = len(identities)
        splits = []
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            # trainval_pids = sorted(pids[:num // 2])
            # test_pids = sorted(pids[num // 2:])

            trainval_pids = []
            test_pids = sorted(pids)
            split = {'trainval': trainval_pids, 'query': test_pids, 'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
