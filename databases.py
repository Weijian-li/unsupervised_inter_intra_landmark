import inspect, torch, pickle, cv2, os, numpy as np, scipy.io as sio
from torch.utils.data import Dataset
from utils import process_image, crop, affine_trans
import math


class SuperDB(Dataset):

    def __init__(self, path=None, sigma=1, size=128, step=15, flip=False, angle=0, tight=16, nimages=2, affine=False,
                 db='CelebA', npts=None):
        # - automatically add attributes to the class with the values given by the class
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        del args[0] 
        for k in args:
            setattr(self, k, values[k])
        preparedb(self, db)
        self.db = db

        radius = 14
        number = npts

        rad = np.radians(np.linspace(360.0 / number, 360, number))
        self.init_pts = ((radius * np.array([[math.cos(x), math.sin(x)] for x in rad])).astype(np.float16) + 16) / 32.

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image, points = self.collect(self, idx)
        nimg = len(image)

        if not self.affine:
            sample = dict.fromkeys(['Im', 'ImP', 'init_pts'], None)
            flip = np.random.rand(1)[0] > 0.5 if self.flip else False  # flip both or none

            for j in range(self.nimages):
                out = process_image(image=image[j % nimg],
                                    points=points[j % nimg],
                                    angle=(j+1)*self.angle,
                                    flip=flip,
                                    size=self.size,
                                    tight=self.tight)
                if j == 1:
                   sample['Im'] = out['image']
                else:
                   sample['ImP'] = out['image']

                sample['init_pts'] = self.init_pts
                sample['M'] = out['M'].astype(np.float32)
        else: 
            tmp_angle = np.clip(np.random.randn(1) * self.angle, -40.0, 40.0) if self.angle > 0 else self.angle
            image, points, _ = affine_trans(image[0], points[0], tmp_angle)
            image, points = crop(image, points, 128, tight=self.tight)
            tmp_angle = np.random.randn(1) * self.angle
            imrot, ptsrot, M = affine_trans(image, points, size=128, angle=tmp_angle)
            image = image/255.0
            image = torch.from_numpy(image.swapaxes(2, 1).swapaxes(1, 0))
            image = image.type_as(torch.FloatTensor())
            imrot = torch.from_numpy(imrot/255.0).permute(2, 0, 1).type_as(torch.FloatTensor())
            sample = {'Im': image, 'ImP': imrot,
                      'M': torch.from_numpy(M).type_as(torch.FloatTensor()),
                      'pts': points,
                      'init_pts': self.init_pts}
        return sample


# Define a function that returns the initialisation and the collect function
def preparedb(self, db):
    if db == 'CelebA':
        def init(self):
            txt_file = open('list_landmarks_align_celeba.txt','r')
            lines = txt_file.readlines()[2::]
            names = [l.split()[0] for l in lines]
            coords = [l.split()[1::] for l in lines]
            data = dict(zip(names, coords))
            mafl = [l.replace('\n', '') for l in open('MAFL_test.txt', 'r').readlines()]  # remove MAFL test from training
            files = sorted(list(set(names) - set(mafl)))
            keys = ['files', 'data']

            for k in keys:
                setattr(self, k, eval(k))
            setattr(self, 'len', len(files))

        def collect(self, idx):
            path_to_img = self.path + self.files[idx]
            # print(path_to_img)
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            points = self.data[self.files[idx]]
            points = np.float32(self.data[self.files[idx]]).reshape(-1, 2)
            return [image], [points]

        init(self)
        setattr(self, 'collect', collect)

    if 'AFLW' in db:
        def init(self):
            partition = db.split('-')[1]
            mymat = sio.loadmat(os.path.join(self.path, 'aflw_{}_keypoints.mat'.format(partition)))
            with open(os.path.join(self.path, 'aflw_{}_images.txt'.format(partition)), 'r') as f:
                alllines = f.readlines()
            for i in range(len(alllines)):
                alllines[i] = alllines[i].replace('\n', '')
            files = alllines
            pts = mymat['gt'][:, :, [1, 0]]
            keys = ['files', 'pts']

            for k in keys:
                setattr(self, k, eval(k)) 
            setattr(self, 'len', len(files))

        def collect(self, idx):
            path_to_img = os.path.join(self.path, 'output', self.files[idx])
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            points = self.pts[idx]
            return [image], [points]

        init(self) 
        setattr(self, 'collect', collect)

    if 'MAFL' in db:
        def init(self):
            partition = db.split('-')[1]
            txt_file = open('list_landmarks_align_celeba.txt', 'r')
            lines = txt_file.readlines()[2::]
            names = [l.split()[0] for l in lines]
            coords = [l.split()[1::] for l in lines]
            data = dict(zip(names, coords))
            files = [l.replace('\n', '') for l in open('MAFL_{}.txt'.format(partition), 'r').readlines()]
            notfound = ['031524.jpg', '179577.jpg', '139803.jpg'] if partition == 'train' else []
            for f in notfound:
                idx = files.index(f)
                del files[idx]
            keys = ['files', 'data']
            for k in keys:
                setattr(self, k, eval(k))
            setattr(self, 'len', len(files))

        def collect(self, idx):
            path_to_img = self.path + self.files[idx]
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            points = self.data[self.files[idx]]
            points = np.float32(self.data[self.files[idx]]).reshape(-1, 2)
            return [image], [points]

        init(self)
        setattr(self, 'collect', collect)



