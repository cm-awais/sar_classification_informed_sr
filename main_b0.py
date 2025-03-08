from cvpr_utils import *
from cvpr_models import *
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

batch_size = 32

# fusar_transform = Compose([Resize((256, 256)), ToTensor()])

# fusar_train_dataset = ImageFolder(root="data/classificiation/c_fusar_splited/train", transform=fusar_transform)
# fusar_val_dataset = ImageFolder(root="data/classificiation/c_fusar_splited/val", transform=fusar_transform)
# fusar_test_dataset = ImageFolder(root="data/classificiation/c_fusar_splited/test", transform=fusar_transform)

# fusar_test_loader = DataLoader(fusar_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
# fusar_train_loader = DataLoader(fusar_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)
# fusar_val_loader = DataLoader(fusar_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)

# opensar_transform = Compose([Resize((64, 64)), ToTensor()])

# opensar_train_dataset = ImageFolder(root="data/classificiation/c_opensar_splited/train", transform=opensar_transform)
# opensar_val_dataset = ImageFolder(root="data/classificiation/c_opensar_splited/val", transform=opensar_transform)
# opensar_test_dataset = ImageFolder(root="data/classificiation/c_opensar_splited/test", transform=opensar_transform)

# opensar_train_loader = DataLoader(opensar_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
# opensar_val_loader = DataLoader(opensar_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)
# opensar_test_loader = DataLoader(opensar_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)

mstar_transform = Compose([Resize((128, 128)), ToTensor()])

mstar_train_dataset = ImageFolder(root="data/super_resolution/mstar_split/train", transform=mstar_transform)
mstar_val_dataset = ImageFolder(root="data/super_resolution/mstar_split/val", transform=mstar_transform)
mstar_test_dataset = ImageFolder(root="data/super_resolution/mstar_split/test", transform=mstar_transform)

mstar_train_loader = DataLoader(mstar_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
mstar_val_loader = DataLoader(mstar_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)
mstar_test_loader = DataLoader(mstar_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)


# mstar_train_dataset = SRDataset(hr_dir="data/super_resolution/mstar_split/train", ds_type="mstar")
# mstar_val_dataset = SRDataset(hr_dir="data/super_resolution/mstar_split/val", ds_type="mstar")
# mstar_test_dataset = SRDataset(hr_dir="data/super_resolution/mstar_split/test", ds_type="mstar")

# mstar_train_loader = DataLoader(mstar_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
# mstar_val_loader = DataLoader(mstar_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)
# mstar_test_loader = DataLoader(mstar_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)

# hrsid_train_dataset = SRDataset(hr_dir="data/super_resolution/hrsid_split/train", ds_type="hrsid")
# hrsid_val_dataset = SRDataset(hr_dir="data/super_resolution/hrsid_split/val", ds_type="hrsid")
# hrsid_test_dataset = SRDataset(hr_dir="data/super_resolution/hrsid_split/test", ds_type="hrsid")

# hrsid_test_loader = DataLoader(hrsid_test_dataset, batch_size=32, shuffle=True, num_workers=4)
# hrsid_train_loader = DataLoader(hrsid_train_dataset, batch_size=32, shuffle=True, num_workers=4)
# hrsid_val_loader = DataLoader(hrsid_val_dataset, batch_size=32, shuffle=True, num_workers=4)

# test_dataset = ConcatDataset([fusar_test_dataset, opensar_test_dataset, mstar_test_dataset, hrsid_test_dataset])
# train_dataset = ConcatDataset([fusar_train_dataset, opensar_train_dataset, mstar_train_dataset, hrsid_train_dataset])
# val_dataset = ConcatDataset([fusar_valid_dataset, opensar_valid_dataset, mstar_valid_dataset, hrsid_valid_dataset])

# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

loaders = { 
            # "fusar":[fusar_train_loader, fusar_val_loader, fusar_test_loader, 9], 
            # "opensar":[opensar_train_loader, opensar_val_loader, opensar_test_loader,  6], 
            "mstar":[mstar_train_loader, mstar_val_loader, mstar_test_loader, 11]
            }

scores = []

# scores.extend(train_base(loaders=loaders,scale=2))
# non-baselines
# classification scores without sr
b3_classification_no_sr(loaders=loaders, scale=2)