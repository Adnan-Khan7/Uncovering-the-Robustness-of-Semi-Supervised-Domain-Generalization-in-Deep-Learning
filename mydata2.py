from torchvision import transforms
from PIL import Image
from torchvision.datasets import DatasetFolder
from randaugment import RandAugmentMC
import torch
import numpy as np
import random
from torch.utils.data import Subset, ConcatDataset
from sklearn.model_selection import train_test_split

# Seeds
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# art, cartoon, photo, sketch
# python train2.py --expand-labels --resume /l/users/adnan.khan/logs/new_pacs/labels_10_seed2/sketch/e_40/mc_10/var_07_avg/checkpoint.pth.tar --out /l/users/adnan.khan/logs/new_pacs/labels_10_seed2/sketch/e_40/mc_10/var_07_avg
# python train.py --expand-labels --resume /l/users/adnan.khan/logs/pacx/art/var_07_avg/checkpoint.pth.tar --out /l/users/adnan.khan/logs/pacx/art/var_07_avg

DATA = "/l/users/adnan.khan/pacs/labels_10/pacs_ssdg/seed2/sketch/"

Data_A_path = "/l/users/adnan.khan/pacs/labels_10/Original/art"
Data_B_path = "/l/users/adnan.khan/pacs/labels_10/Original/cartoon"
Data_C_path = "/l/users/adnan.khan/pacs/labels_10/Original/photo"


DATA_TRAIN_SET = (DATA + "train")
# DATA_UNLABELED_SET = (DATA + "unlabeled")
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

class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            #adding resize
            #transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            #adding resize
            #transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

train_set = DatasetFolder(DATA_TRAIN_SET, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                          transform=train_tfm)

Data_A = DatasetFolder(Data_A_path, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                          transform=TransformFixMatch())
Data_B = DatasetFolder(Data_B_path, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                          transform=TransformFixMatch())
Data_C = DatasetFolder(Data_C_path, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                          transform=TransformFixMatch())

test_set = DatasetFolder(DATA_TEST_SET, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                         transform=test_tfm)

# unlabeled_dataset = DatasetFolder(DATA_UNLABELED_SET, loader=lambda x: Image.open(x).convert('RGB'),
#                                       extensions=IMG_EXTENSIONS, transform=TransformFixMatch())

# val_dataset = DatasetFolder(DATA_UNLABELED_SET, loader=lambda x: Image.open(x).convert('RGB'),
#                                       extensions=IMG_EXTENSIONS, transform=test_tfm)

# targets = unlabeled_dataset.targets

# unlabeled_idx, valid_idx= train_test_split(
#     np.arange(len(targets)),
#     test_size=0.1,
#     shuffle=True,
#     stratify=targets)

# un_dataset = torch.utils.data.Subset(unlabeled_dataset,unlabeled_idx)
# v_dataset = torch.utils.data.Subset(val_dataset, valid_idx)

# print("Number of unlabeled samples: ", len(un_dataset))
Data_a = DatasetFolder(Data_A_path, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                          transform=train_tfm)
Data_b = DatasetFolder(Data_B_path, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                          transform=train_tfm)
Data_c = DatasetFolder(Data_C_path, loader=lambda x: Image.open(x).convert('RGB'), extensions=IMG_EXTENSIONS,
                          transform=train_tfm)


# create validation data split for each of the folders
a_targets = Data_a.targets
a_unlabeled_idx, a_valid_idx = train_test_split(np.arange(len(a_targets)), test_size=0.1, shuffle=True, stratify=a_targets)
b_targets = Data_b.targets
b_unlabeled_idx,b_valid_idx = train_test_split(np.arange(len(b_targets)), test_size=0.1, shuffle=True, stratify=b_targets)
c_targets = Data_c.targets
c_unlabeled_idx, c_valid_idx = train_test_split(np.arange(len(c_targets)), test_size=0.1, shuffle=True, stratify=c_targets)

# create the final validation dataset by concatenating the 3 separate validation datasets together
a_valid_dataset = Subset(Data_a, a_valid_idx)
b_valid_dataset = Subset(Data_b, b_valid_idx)
c_valid_dataset = Subset(Data_c, c_valid_idx)
v_dataset = ConcatDataset([a_valid_dataset, b_valid_dataset, c_valid_dataset])