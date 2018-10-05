from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class DeepFashion(Dataset):
    url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'
    md5 = '1c2d9fc1cc800332567a0da25a1ce68c'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(DeepFashion, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import hashlib
        from glob import glob
        from scipy.misc import imsave, imread
        from six.moves import urllib
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)


        # Extract the file
        exdir = osp.join(raw_dir, 'deepfashion')
        if not osp.isdir(exdir):
            print("warning exdir not existent")


        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        all_imgs = glob(exdir + '/*')

        def get_id(img):
            return int(img.split('/')[-1].split('.')[0].split('_')[0])
        all_id = [get_id(i) for i in all_imgs]
        from collections import Counter
        count_id = Counter(all_id)
        n_id = len(count_id)

        all_pid_all = [[] for i in range(max(all_id))]
        for img in all_imgs:
            all_pid_all[get_id(img)-1].append(img)


        all_pid = [pid for pid in all_pid_all if pid]
        all_pid = [pid for pid in all_pid if count_id[get_id(pid[0])]>=4]

        # for imgs in all_pid:
        #     c = count_id[get_id(imgs[0])]
        #     if c == 3:
        #         imgs.append(imgs[0])
        #     if c == 2:
        #         imgs *= 2


        identities = []
        for pid, imgs in enumerate(all_pid):
            # assert len(imgs) == count_id[pid]
            images = []
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
            imsave(osp.join(images_dir, fname), imread(imgs[0]))
            images.append([fname])
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 1, 0)
            imsave(osp.join(images_dir, fname), imread(imgs[1]))
            images.append([fname])
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 2, 0)
            imsave(osp.join(images_dir, fname), imread(imgs[2]))
            images.append([fname])
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 3, 0)
            imsave(osp.join(images_dir, fname), imread(imgs[3]))
            images.append([fname])

            identities.append(images)
            if pid % 100 == 0:
                print('ID {}/{}'.format(pid, len(all_pid)))

        # cameras = [sorted(glob(osp.join(exdir, 'cam_a', '*.bmp'))),
        #            sorted(glob(osp.join(exdir, 'cam_b', '*.bmp')))]
        # assert len(cameras[0]) == len(cameras[1])
        # identities = []
        # for pid, (cam1, cam2) in enumerate(zip(*cameras)):
        #     images = []
        #     view-0
        #     fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
        #     imsave(osp.join(images_dir, fname), imread(cam1))
        #     images.append([fname])
        #     view-1
        #     fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 1, 0)
        #     imsave(osp.join(images_dir, fname), imread(cam2))
        #     images.append([fname])
        #     identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'deepfashion', 'shot': 'single', 'num_cameras': 4,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Randomly create ten training and test split
        num = len(identities)
        splits = []
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            trainval_pids = sorted(pids[:num // 2])
            test_pids = sorted(pids[num // 2:])
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
