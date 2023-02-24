from torchvision import transforms
from PIL import Image
from torchvision.datasets import DatasetFolder
from randaugment import RandAugmentMC
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Seeds
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class TransformFixMatch(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if self.dataset_name == 'pacs':
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=224,
                                      padding=int(224*0.125),
                                      pad_if_needed=True,
                                      padding_mode='reflect')])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=224,
                                      padding=int(224*0.125),
                                      pad_if_needed=True,
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.dataset_name in ['vlcs', 'office_home']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=224,
                                      padding=int(224*0.125),
                                      pad_if_needed=True,
                                      padding_mode='reflect')])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=224,
                                      padding=int(224*0.125),
                                      pad_if_needed=True,
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


DATA = "/home/maha.agro/FixMatch-pytorch/dataset/seed2/VOC2007/"
DATA_TRAIN_SET = (DATA + "train")
DATA_UNLABELED_SET = (DATA + "unlabeled")
DATA_TEST_SET = (DATA + "test")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_dataset(dataset_name):
    train_set = DatasetFolder(DATA_TRAIN_SET, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                              transform=train_tfm)

    test_set = DatasetFolder(DATA_TEST_SET, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                             transform=test_tfm)

    unlabeled_dataset = DatasetFolder(DATA_UNLABELED_SET, loader=lambda x: Image.open(x).convert('RGB'),
                                      extensions=IMG_EXTENSIONS, transform=TransformFixMatch(dataset_name))

    val_dataset = DatasetFolder(DATA_UNLABELED_SET, loader=lambda x: Image.open(x).convert('RGB'),
                                      extensions=IMG_EXTENSIONS, transform=test_tfm)

    targets = unlabeled_dataset.targets

    unlabeled_idx, valid_idx= train_test_split(
    np.arange(len(targets)),
    test_size=0.1,
    shuffle=True,
    stratify=targets)
    un_dataset = torch.utils.data.Subset(unlabeled_dataset,unlabeled_idx)
    v_dataset = torch.utils.data.Subset(val_dataset, valid_idx)

    print("Size of unlabeled data: ",len(unlabeled_dataset))
    return train_set, un_dataset, v_dataset, test_set 