import torch
import numpy as np
import os
from skimage import io
from torch.utils.data import Dataset
from scipy.ndimage.morphology import grey_dilation

NUM_POINTS = 19
HEATMAP_SIZE = (64, 64)
SIGMA = 1

class AFLW(Dataset):
    def __init__(self, annotations_file, img_dir, max_samples=None, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        text = None
        self.img_list = []
        self.points = None
        
        num_samples = 0
        
        with open(annotations_file, 'r') as f:
            text = f.readlines()
            self.points = torch.zeros(len(text), NUM_POINTS, 2)
        if max_samples is not None:
            num_samples = max_samples
        else:
            num_samples = len(text)
        for i in range(num_samples):
            token = text[i].strip().split()
            self.img_list.append(os.path.join(img_dir, token[0]))
            self.points[i] = torch.tensor([float(x) for x in token[1:]], dtype=torch.float32).view(NUM_POINTS, 2)
        self.heatmap_type = 'gaussian'

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        image = io.imread(self.img_list[idx])
        target = torch.zeros(NUM_POINTS, HEATMAP_SIZE[0], HEATMAP_SIZE[1])
        tpts = self.points[idx].clone()
        M = np.zeros((NUM_POINTS, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=np.float32)
        
        for i in range(NUM_POINTS):
            if tpts[i, 1] > 0:
                target[i] = self._generate_target(target[i], tpts[i] * HEATMAP_SIZE[0] - 1, SIGMA)
        if self.transform:
            image = self.transform(image)
            
        for i in range(len(M)):
            # According to the paper https://arxiv.org/pdf/1904.07399 and size of the grey dilation is 3 x 3
            # But I test with 1 x 1 and it works better
            M[i] = grey_dilation(target[i], size=(1, 1))
        M = np.where(M>=0.5, 1, 0)
        
        meta = {'index': idx, 'pts': self.points[idx]}
        return image, target, M, meta
    
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    def _generate_target(self, img, points, sigma, type='Gaussian'):

        h,w = img.shape
        tmp_size = sigma * 3

        # Check that any part of the gaussian is in-bounds
        x1, y1 = int(points[0] - tmp_size), int(points[1] - tmp_size) # Top-left
        x2, y2 = int(points[0] + tmp_size + 1), int(points[1] + tmp_size + 1) # Bottom right
        if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        if type == 'Gaussian':
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        elif type == 'Cauchy':
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

        # Usable gaussian range
        # Determine the bounds of the source gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

        # Image range
        img_x_min, img_x_max = max(0, x1), min(x2, w)
        img_y_min, img_y_max = max(0, y1), min(y2, h)

        img[img_y_min:img_y_max, img_x_min:img_x_max] = \
          torch.from_numpy(g[g_y_min:g_y_max, g_x_min:g_x_max])

        return img
    
    def mean_and_std(self, img_dir, meanstd_file):
        """
        Compute the mean and std of the dataset
        Args:
            img_dir: the directory of the images
            meanstd_file: the file to save the mean and std
        """

        # BEGIN CODE
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)

            for file_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, file_name)
                img = io.read_image(img_path) # HxWxC
                img = torch.tensor(img, dtype=torch.float32) # CxHxW
                img = img.type('torch.FloatTensor')
                img /= 255
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(os.listdir(img_dir))
            std /= len(os.listdir(img_dir))
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        # END CODE