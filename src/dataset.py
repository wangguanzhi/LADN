import os
import os.path
from PIL import Image
import numpy as np
from numpy import asarray
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import pickle
import sys

from utils.train_util import facePartsCoordinatesAPI

# from utils.helpers import FaceCropper
from utils.constants import DLIB_DAT_PATH, DUMMY_BLEND_PATH

class dataset_makeup(data.Dataset):
    def __init__(self, opts, mode ='train'):
        random.seed(9001)
        self.dataroot = opts.dataroot
        self.landmark_file = opts.landmark_file
        self.no_extreme = opts.no_extreme
        self.extreme_only = opts.extreme_only

        self.makeup_weight = 0.8
        self.skin_weight = 0.2
        self.background_weight = 0

        self.resize_size = opts.resize_size
        self.crop_size = opts.crop_size
        self.n_local = opts.n_local

        # Now in the training process, an additional dataset is used for test_forward
        # And thus need one more parameter to control the behavior of the dataset
        self.phase = 'test' if (opts.phase == 'test' or mode == 'test') else 'train'
        self.mode = mode
        self.test_size = opts.test_size

        '''A_all and B_all are used for sampling datapoints for testing forward and interpolation forwarding'''
        # Before makeup
        imgs_before = os.listdir(os.path.join(self.dataroot, "before"))
        self.A_all = [os.path.join(self.dataroot, "before", x) for x in imgs_before]
        self.A = self.A_all

        # After makeup
        imgs_after = os.listdir(os.path.join(self.dataroot, "after"))
        self.B_all = [os.path.join(self.dataroot, "after", x) for x in imgs_after]
        if self.no_extreme:
            imgs_after = [f for f in imgs_after if f[0]=="0" or f[:2] in ["20", "30", "31"]]
        if self.extreme_only:
            imgs_after = [f for f in imgs_after if f[0]=="1" or f[:2] in ["21"]]
        self.B = [os.path.join(self.dataroot, "after", x) for x in imgs_after]

        self.A_size = len(self.A)
        self.B_size = len(self.B)

        # get image pairs for testing purpose
        if self.mode == "test":
            # Sample a set of data points for testing purpose
            self.datapoints = []

            if opts.test_random:
                print("Using random sampled points for test_forward")
                rs = random.getstate()
                # The sampled points should be the same across multiple runs
                random.seed(2018)
                for i in range(self.test_size):
                    path_B = self.B[random.randint(0, self.B_size - 1)]
                    path_A = self.A[random.randint(0, self.A_size - 1)]
                    path_C = self.getPathC(path_A, path_B)
                    self.datapoints.append([path_A, path_B, path_C])
                random.setstate(rs)
            else:
                print("Iterating over all pairs of images and generate results")
                for path_A in self.A:
                    for path_B in self.B:
                        path_C = self.getPathC(path_A, path_B)
                        self.datapoints.append([path_A, path_B, path_C])
                random.shuffle(self.datapoints)
                self.datapoints = self.datapoints[:self.test_size]

            self.dataset_size = len(self.datapoints)
            print("Dataset for testing initialized, including %d data points" % self.dataset_size)
        # interpolate mode, will return one before and two after images for interpolation of makeup styles
        elif self.mode == "interpolate":
            self.A_interpolate = np.random.choice(self.B, opts.test_size)
            self.B_interpolate = np.random.choice(self.B, opts.test_size)
            self.dataset_size = self.test_size
        # normal training dataset
        elif self.mode == "train":
            self.dataset_size = max(self.A_size, self.B_size)
            print('Dataset for training A: %d, B: %d images'%(self.A_size, self.B_size))
        else:
            print("Unknown mode for dataset: %s" % mode)
            assert False

        self.input_dim_A = opts.input_dim_a
        self.input_dim_B = opts.input_dim_b

        # Load api landmars if landmark_file is provided
        if self.landmark_file:
            from utils.api_util import FacePPAPI
            api_landmarks = pickle.load(open(self.landmark_file, 'rb'))
            if len(api_landmarks.keys()) >= self.A_size + self.B_size:
                self.api_landmarks = api_landmarks
                print("API landmarks loaded successfully!")
            else:
                print("Number of landmarks is not enough.")

        # setup image transformation
        # transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        transforms = []
        if self.mode == 'train':
            '''Here, we will do random cropping at __getitem__'''
            pass
        else:
            transforms.append(CenterCrop(opts.crop_size))
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        # Initialize the FaceCropper class
        # self.fc = FaceCropper(predictor_dir = DLIB_DAT_PATH)
        return

    # Return the index to do the random cropping
    def randomCropIndex(self, starting_min, starting_max, size):
        starting_range = [starting_min, starting_max]
        x1 = random.randint(*starting_range)
        y1 = random.randint(*starting_range)
        x2 = x1 + size
        y2 = y1 + size
        return x1, x2, y1, y2

    def getPathC(self, path_A, path_B):
        # If the phase is test, then return a dummy placeholder for blend image
        if self.phase == "test":
            return DUMMY_BLEND_PATH
        # Get the blended images
        imgnum_A = path_A.split("/")[-1].split(".")[0]
        imgnum_B = path_B.split("/")[-1].split(".")[0]
        # path_C = os.path.join(self.dataroot, "blend", "%s_%s.jpg" % (imgnum_A, imgnum_B))
        path_C = os.path.join(self.dataroot, "blend", "%s_%s.jpg" % (imgnum_A, imgnum_B))
        return path_C

    '''For testing purpose: phase should be test'''
    def getDatapointBySeed(self, seed):
        s = random.getstate()
        random.seed(seed)
        path_B = self.B[random.randint(0, self.B_size - 1)]
        path_A = self.A[random.randint(0, self.A_size - 1)]
        random.setstate(s)

        path_C = self.getPathC(path_A, path_B)

        img_A = self.load_img(path_A)
        img_B = self.load_img(path_B)
        img_C = self.load_img(path_C)

        data = {
            "img_A": img_A,
            "img_B": img_B,
            "img_C": img_C,
        }

        return data

    '''For testing purpose: phase should be test'''
    def getDatapointByName(self, path_A, path_B):
        path_C = self.getPathC(path_A, path_B)

        img_A = self.load_img(path_A)
        img_B = self.load_img(path_B)
        img_C = self.load_img(path_C)

        data = {
            "img_A": img_A,
            "img_B": img_B,
            "img_C": img_C,
        }

        return data

    def __getitem__(self, index):
        # Return the specific images path for testing purpose when testing
        if self.mode == "test":
            path_A, path_B, path_C = self.datapoints[index % self.dataset_size]
        elif self.mode == "interpolate":
            path_A = random.choice(self.A)
            path_B = random.choice(self.B)
            path_C = random.choice(self.B)
        elif self.mode == "train":
            if self.dataset_size == self.A_size:
                path_A = self.A[index]
                path_B = self.B[random.randint(0, self.B_size - 1)]
            else:
                path_A = self.A[random.randint(0, self.A_size - 1)]
                path_B = self.B[index]
            path_C = self.getPathC(path_A, path_B)
        else:
            print("Unknown mode: %s" % self.mode)
            assert False

        index_A = path_A.split('/')[-1].split(".")[-2]
        index_B = path_B.split('/')[-1].split(".")[-2]

        # store the current random state for image selection
        rs = random.getstate()

        # Get the images and the masks
        img_A = Image.open(path_A).convert('RGB')
        size_A = img_A.size[0]
        img_A = img_A.resize((self.resize_size, self.resize_size))
        img_B = Image.open(path_B).convert('RGB')
        size_B = img_B.size[0]
        img_B = img_B.resize((self.resize_size, self.resize_size))
        img_C = Image.open(path_C).convert('RGB')
        size_C = img_C.size[0]
        img_C = img_C.resize((self.resize_size, self.resize_size))

        img_A_arr = asarray(img_A)
        img_B_arr = asarray(img_B)
        img_C_arr = asarray(img_C)

        # Extract the landmarks from pre-loaded in api or detect them using Stasm
        use_api_landmark = not self.landmark_file is None
        if use_api_landmark:
            landmark_A_api = self.api_landmarks['/'.join(path_A.split('/')[-2:])]
            landmark_B_api = self.api_landmarks['/'.join(path_B.split('/')[-2:])]
            landmark_C_api = landmark_A_api

        '''
        rects is [left_eye, right_eye, mouth, nose, left_cheek, right_cheek], with each one being [x1, x2, y1, y2]
        '''
        # Get the rects of eyes and mouth
        rects_A = facePartsCoordinatesAPI(img_A_arr, landmark_A_api, n_local = self.n_local, scaling_factor = self.resize_size / size_A)
        rects_B = facePartsCoordinatesAPI(img_B_arr, landmark_B_api, n_local = self.n_local, scaling_factor = self.resize_size / size_B)
        rects_C = rects_A

        # The coordiniates need to be scaled according to the resize_size
        rects_A = np.array(rects_A).astype(int)
        rects_B = np.array(rects_B).astype(int)
        rects_C = np.array(rects_C).astype(int)

        for r in [rects_A, rects_B, rects_C]:
            assert r[:, :2].max() - r[:, :2].min() <= self.crop_size
            assert r[:, 2:].max() - r[:, 2:].min() <= self.crop_size

        # # Conver the numpy array to PIL Image
        # mask_A = self.fc.faceMask(img_A_arr, landmark_A, white = self.makeup_weight, grey = self.skin_weight, black = self.background_weight)
        # mask_B = self.fc.faceMask(img_B_arr, landmark_B, white = self.makeup_weight, grey = self.skin_weight, black = self.background_weight)
        # mask_C = mask_A

        # mask_A_arr = (mask_A * 255).astype(np.uint8)
        # mask_B_arr = (mask_B * 255).astype(np.uint8)
        # mask_C_arr = (mask_C * 255).astype(np.uint8)

        '''if phase is train: Do the random cropping manually'''
        '''If phase is test or interpolate: the cropping will be done in self.transforms and CenterCrop will be used'''
        if self.mode == "train":
            # random crop should include all the rects of A
            starting_max = min(rects_A.min(), self.resize_size - self.crop_size)
            starting_min = max(rects_A.max() - self.crop_size, 0)

            # Random crop should include all the rects of B
            starting_max = min(starting_max, rects_B.min())
            starting_min = max(starting_min, rects_B.max() - self.crop_size)

            x1,x2,y1,y2 = self.randomCropIndex(starting_min, starting_max, self.crop_size)
            img_A_arr = img_A_arr[x1:x2, y1:y2]
            # mask_A_arr = mask_A_arr[x1:x2, y1:y2]
            offset = np.array([[x1,x1,y1,y1]])
            rects_A -= offset

            img_C_arr = img_C_arr[x1:x2, y1:y2]
            # mask_C_arr = mask_C_arr[x1:x2, y1:y2]
            offset = np.array([[x1,x1,y1,y1]])
            rects_C -= offset

            img_B_arr = img_B_arr[x1:x2, y1:y2]
            # mask_B_arr = mask_B_arr[x1:x2, y1:y2]
            offset = np.array([[x1,x1,y1,y1]])
            rects_B -= offset

        # Convert the croped imgs and masks from numpy array to PIL Image
        # mask_A = Image.fromarray(mask_A_arr)
        # mask_B = Image.fromarray(mask_B_arr)
        # mask_C = Image.fromarray(mask_C_arr)
        img_A = Image.fromarray(img_A_arr)
        img_B = Image.fromarray(img_B_arr)
        img_C = Image.fromarray(img_C_arr)

        # img_A, mask_A = self.identityTransform(img_A, mask_A, self.input_dim_A)
        # img_B, mask_B = self.identityTransform(img_B, mask_B, self.input_dim_B)
        # img_C, mask_C = self.identityTransform(img_C, mask_C, self.input_dim_A)
        img_A = self.transforms(img_A)
        img_B = self.transforms(img_B)
        img_C = self.transforms(img_C)

        # recover the random state for image selection
        random.setstate(rs)

        # print("img_A.shape:", img_A.shape)
        # print("img_B.shape:", img_B.shape)
        # print("img_C.shape:", img_C.shape)

        data = {
            "img_A": img_A,
            "img_B": img_B,
            "img_C": img_C,
            # "landmark_A": landmark_A,
            # "landmark_B": landmark_B,
            # "landmark_C": landmark_C,
            "rects_A": rects_A,
            "rects_B": rects_B,
            "rects_C": rects_C,
            "index_A": index_A,
            "index_B": index_B
        }

        return data

    def transformImage(self, img):
        img = self.transforms(img)
        return img

    # to make the transform performed the same way on two images
    def identityTransform(self, imga, imgb, input_dim):
        seed = np.random.randint(0, 2**32)

        random.seed(seed)
        imga = self.transforms(imga)
        random.seed(seed)
        imgb = self.transforms(imgb)

        if input_dim == 1:
            imga = imga[0, ...] * 0.299 + imga[1, ...] * 0.587 + imga[2, ...] * 0.114
            imga = imga.unsqueeze(0)
            imgb = imgb[0, ...] * 0.299 + imgb[1, ...] * 0.587 + imgb[2, ...] * 0.114
            imgb = imgb.unsqueeze(0)
        return imga, imgb

    def load_img(self, img_name, input_dim=3):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.dataset_size
















#
