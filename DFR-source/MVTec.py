import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from skimage.io import imread
from skimage.transform import resize


def get_image_files(path, mode='train'):
    images = []
    ext = {'.jpg', '.png'}
#     path = "/home/jie/Datasets/mvtec-anomaly/bottle/test"
    for root, dirs, files in os.walk(path):
        print('loading image files ' + root)
        for file in files:
            if mode == 'train':
                if os.path.splitext(file)[1] in ext:
                    images.append(os.path.join(root, file))
            else:
                if os.path.splitext(file)[1] in ext and "good" not in root:
                    images.append(os.path.join(root, file))
    return sorted(images)


def get_mask_files(path):
    masks = []
    ext = {'.jpg', '.png'}
#     path = "/home/jie/Datasets/mvtec-anomaly/bottle/ground_truth"
    for root, dirs, files in os.walk(path):
        print('loading mask files ' + root)
        for file in files:
            if os.path.splitext(file)[1] in ext:
                masks.append(os.path.join(root, file))
    return sorted(masks)


"""
MVTec datasets
"""
class NormalDataset(Dataset):

    def __init__(self, path, normalize=True):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.img_files = self._get_image_files(path)
        self.len = len(self.img_files)

        # transformer
        resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
        if normalize:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([resize, transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        """
        img = Image.open(self.img_files[idx])
        if "zipper" in self.img_files[idx] or "screw" in self.img_files[idx] or "grid" in self.img_files[idx]:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _get_image_files(self, path, ext={'.jpg', '.png'}):
        images = []
        for root, dirs, files in os.walk(path):
            print('loading image files ' + root)
            for file in files:
                if os.path.splitext(file)[1] in ext:  # and "good" not in root
                    images.append(os.path.join(root, file))
        return sorted(images)


class AbnormalDataset(Dataset):
    """
    Only getting the abnormal data! ('good' is not included.)

    """
    def __init__(self, path, normalize=True):
        self.img_files = self._get_image_files(path)
        self.len = len(self.img_files)

        # transformer
        resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
        if normalize:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([resize, transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        """
        img = Image.open(self.img_files[idx])
        if "zipper" in self.img_files[idx] or "screw" in self.img_files[idx] or "grid" in self.img_files[idx]:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _get_image_files(self, path, ext={'.jpg', '.png'}):
        images = []
        for root, dirs, files in os.walk(path):
            print('loading image files ' + root)
            for file in files:
                if os.path.splitext(file)[1] in ext and "good" not in root:  # and "good" not in root
                    images.append(os.path.join(root, file))
        return sorted(images)


class MaskDataset(Dataset):
    def __init__(self, path):
        self.mask_files = self._get_mask_files(path)
        self.len = len(self.mask_files)

        # transformer
        resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
        self.transform = transforms.Compose([resize])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        """
        mask = imread(self.mask_files[idx])
        mask = resize(mask, (256, 256))
        mask = np.expand_dims(mask, axis=0)
        return torch.Tensor(mask)

    def _get_mask_files(self, path, ext={'.jpg', '.png'}):
        masks = []
        #         path = "/home/jie/Datasets/mvtec-anomaly/bottle/ground_truth"
        for root, dirs, files in os.walk(path):
            print('loading mask files ' + root)
            for file in files:
                if os.path.splitext(file)[1] in ext:
                    masks.append(os.path.join(root, file))
        return sorted(masks)


class TestDataset(Dataset):

    def __init__(self, path, normalize=True):
        self.img_files = self._get_image_files(path)
        self.len = len(self.img_files)

        # transformer
        resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
        if normalize:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([resize, transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        """
        img = Image.open(self.img_files[idx])
        if "zipper" in self.img_files[idx] or "screw" in self.img_files[idx] or "grid" in self.img_files[idx]:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img_name = self.img_files[idx]

        # mask
        # h, w, _ = img.shape
        if img_name.split('/')[-2] == "good":
            mask = np.zeros((256, 256))
        else:
            if "wine" in self.img_files[idx]:
                mask_path = img_name.replace("test", "ground_truth").split(".")[-2] + ".png"
                mask = imread(mask_path, as_gray=True)
                mask = resize(mask, (256, 256))
            else:
                mask_path = img_name.replace("test", "ground_truth").split(".")[-2] + "_mask.png"
                mask = imread(mask_path, as_gray=True)
                mask = resize(mask, (256, 256))
        return img, mask, img_name

    def _get_image_files(self, path, ext={'.jpg', '.png'}):
        images = []
        for root, dirs, files in os.walk(path):
            if "check" in root:
                continue
            print('loading image files ' + root)
            for file in files:
                if os.path.splitext(file)[-1] in ext and 'checkpoint' not in file:
                    images.append(os.path.join(root, file))
        return sorted(images)

# np.pad()

# ---------------------------------------------------------------------------- #
# For supervised learning and performance boundary analysis.
# ---------------------------------------------------------------------------- #

# for sklearn ann (one-shot)
class ValTestDataset(Dataset):

    def __init__(self, path, val=True, val_size=0.4, seed=0):
        self.seed = 0
        self.total_img_files = self._get_image_files(path)
        self.len = len(self.total_img_files)

        np.random.seed(self.seed)
        idx = np.random.permutation(range(self.len))
        val_idx = idx[:int(self.len*val_size)]
        test_idx = idx[int(self.len*val_size):]

        if val:
            self.img_files = np.array(self.total_img_files)[val_idx].tolist()
            self.len = len(self.img_files)
        else:
            self.img_files = np.array(self.total_img_files)[test_idx].tolist()
            self.len = len(self.img_files)

        # transformer
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
        self.transform = transforms.Compose([resize, transforms.ToTensor(), normalize])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        """
        img = Image.open(self.img_files[idx])
        if "zipper" in self.img_files[idx] or "screw" in self.img_files[idx] or "grid" in self.img_files[idx]:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img_name = self.img_files[idx]

        # mask
        # h, w, _ = img.shape
        if img_name.split('/')[-2] == "good":
            mask = np.zeros((256, 256))
        else:
            mask_path = img_name.replace("test", "ground_truth").split(".")[-2] + "_mask.png"
            mask = imread(mask_path)
            mask = resize(mask, (256, 256))
        return img, mask, img_name

    def _get_image_files(self, path, ext={'.jpg', '.png'}):
        images = []
        for root, dirs, files in os.walk(path):
            print('loading image files ' + root)
            for file in files:
                if os.path.splitext(file)[1] in ext:
                    images.append(os.path.join(root, file))
        return sorted(images)


# for pytorch
class TrainTestDataset(Dataset):
    """
    For reading the abnormal image and its mask, and normal image simutaneously
    """

    def __init__(self, abnormal_path, normal_path=None, is_train=True, train_size=0.4, seed=0, normalize=True):
        self.seed = seed
        self.is_train = is_train
        self.total_img_files = self._get_image_files(abnormal_path, is_normal=False)
        self.len = len(self.total_img_files)

        np.random.seed(self.seed)
        idx = np.random.permutation(range(self.len))
        val_idx = idx[:int(self.len*train_size)]
        test_idx = idx[int(self.len*train_size):]
        # print("val idx", val_idx)
        if is_train:
            self.img_files = np.array(self.total_img_files)[val_idx].tolist()
            self.len = len(self.img_files)
            # normal image
            self.img_normal_files = self._get_image_files(normal_path, is_normal=True)
            self.img_normal_files = self.img_normal_files[0:self.len]
        else:
            self.img_files = np.array(self.total_img_files)[test_idx].tolist()
            self.len = len(self.img_files)

        # transformer
        # t_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # t_resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
        # self.transform = transforms.Compose([t_resize, transforms.ToTensor(), t_normalize])
        # transformer
        t_resize = transforms.Resize(size=(256, 256), interpolation=Image.NEAREST)
        if normalize:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([t_resize, transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([t_resize, transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        """
        # abnormal image
        img = Image.open(self.img_files[idx])
        if "zipper" in self.img_files[idx] or "screw" in self.img_files[idx] or "grid" in self.img_files[idx]:
            img = np.expand_dims(np.array(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img_name = self.img_files[idx]

        # mask corresponds to abnormal image
        mask_path = img_name.replace("test", "ground_truth").split(".")[-2] + "_mask.png"
        mask = imread(mask_path)
        mask = resize(mask, (256, 256), mode='constant')

        # normal image
        if self.is_train:
            normal_img = Image.open(self.img_normal_files[idx])
            if "zipper" in self.img_files[idx] or "screw" in self.img_files[idx] or "grid" in self.img_files[idx]:
                normal_img = np.expand_dims(np.array(normal_img), axis=2)
                normal_img = np.concatenate([normal_img, normal_img, normal_img], axis=2)
                normal_img = Image.fromarray(normal_img.astype('uint8')).convert('RGB')
            if self.transform is not None:
                normal_img = self.transform(normal_img)
            return img, mask, normal_img, img_name
        else:
            return img, mask, img_name

    def _get_image_files(self, path, is_normal=True, ext={'.jpg', '.png'}):
        images = []
        if is_normal:
            for root, dirs, files in os.walk(path):
                print('loading image files ' + root)
                for file in files:
                    images.append(os.path.join(root, file))
        else:
            for root, dirs, files in os.walk(path):
                print('loading image files ' + root)
                for file in files:
                    if os.path.splitext(file)[1] in ext and "good" not in root:  # and "good" not in root
                        images.append(os.path.join(root, file))
        return sorted(images)


def build_dataset_from_featmap(x, mask=None, ksize=5, stride=5, agg_type='avg', device='cpu'):
    r""" implementation in pytorch
    Testing Supervised learning.

    Returns:
      featmap patches xx as training data: [b, num_patch, c, 1, 1]
      reduced masks mm as labels: [b, num_patch, 1, 1, 1]
    """
    # get shapes
    b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

    # extract patches from high-level feature maps
    xx = torch.nn.functional.unfold(x, (ksize, ksize), stride=(stride, stride))  # 'same' padding
    xx = xx.contiguous().view(b, c, ksize, ksize, -1)  # w: (b, c, k, k, h*w）
    xx = xx.permute(0, 4, 1, 2, 3)  # w: (b, h*w, c, k, k)
    if agg_type == 'avg':
        xx = xx.mean(dim=(3, 4), keepdim=True)

    # resize and extract patches from masks
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    m = torch.nn.functional.unfold(mask, (ksize, ksize), stride=(stride, stride))
    m = m.contiguous().view(b, 1, ksize, ksize, -1)  # m: (b, 1, k, k, h*w)
    m = m.permute(0, 4, 1, 2, 3)                     # m: (b, h*w, 1 , k, k)
    mm = torch.eq(m.mean(dim=(2, 3, 4), keepdim=True), torch.tensor(1.).to(device)).to(torch.float32)

    return xx, mm

if __name__ == "__main__":
    data_name = "bottle"
    train_data_path = "/home/jovyan/work/dataset/MVAomaly/"+ data_name + "/train"
    test_data_path = "/home/jovyan/work/dataset/MVAomaly/" + data_name + "/test"
    train_data = NormalDataset(path=train_data_path)
    test_data = TestDataset(path=test_data_path)
    # print(train_data.img_files)
    for img_file in test_data.img_files:
        print(img_file)

    # # data loader
    # from torch.utils.data import DataLoader
    # # train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    # # for normal_img in train_data_loader:
    # #     print("#############")
    # #     print(normal_img.shape)


    # test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    # for abnormal_img, mask, abnormal_img_name in test_data_loader:
    #     print("#############")
    #     print(abnormal_img_name)
    #     print(abnormal_img.shape)
    #     print(mask.shape)
    #     print(mask.max(), mask.min())





