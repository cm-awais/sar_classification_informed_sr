from cvpr_utils import *
from cvpr_models import *
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

batch_size = 24

fusar_train_dataset = SRDataset(hr_dir="data/super_resolution/fusar_split/train", ds_type="fusar")
fusar_val_dataset = SRDataset(hr_dir="data/super_resolution/fusar_split/val", ds_type="fusar")
fusar_test_dataset = SRDataset(hr_dir="data/super_resolution/fusar_split/test", ds_type="fusar")

fusar_test_loader = DataLoader(fusar_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
fusar_train_loader = DataLoader(fusar_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)
fusar_val_loader = DataLoader(fusar_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)

opensar_train_dataset = SRDataset(hr_dir="data/super_resolution/opensar_split/train", ds_type="opensar")
opensar_val_dataset = SRDataset(hr_dir="data/super_resolution/opensar_split/val", ds_type="opensar")
opensar_test_dataset = SRDataset(hr_dir="data/super_resolution/opensar_split/test", ds_type="opensar")

opensar_train_loader = DataLoader(opensar_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
opensar_val_loader = DataLoader(opensar_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)
opensar_test_loader = DataLoader(opensar_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)

mstar_train_dataset = SRDataset(hr_dir="data/super_resolution/mstar_split/train", ds_type="mstar")
mstar_val_dataset = SRDataset(hr_dir="data/super_resolution/mstar_split/val", ds_type="mstar")
mstar_test_dataset = SRDataset(hr_dir="data/super_resolution/mstar_split/test", ds_type="mstar")

mstar_train_loader = DataLoader(mstar_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
mstar_val_loader = DataLoader(mstar_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)
mstar_test_loader = DataLoader(mstar_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,  drop_last=False, pin_memory=True)

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

loaders = { "fusar":[fusar_train_loader, fusar_val_loader, fusar_test_loader, 15], 
            "opensar":[opensar_train_loader, opensar_val_loader, opensar_test_loader,  17], 
            "mstar":[mstar_train_loader, mstar_val_loader, mstar_test_loader, 11]
            }

scores = []

# scores.extend(train_base(loaders=loaders,scale=2))
# non-baselines
scale = 2

models = {
        # "edsr" : lambda: edsr(scale=scale, pretrained=True).to(device),
        "rcan" : lambda: rcan(scale=scale, pretrained=True).to(device),
        # "carn" : lambda: carn(scale=scale, pretrained=True).to(device),
        }

scores.extend(finetune_w_classification2(loaders=loaders, models=models, scale=2))

os.makedirs('cvpr', exist_ok=True)

headers = ["train_psnr", "train_ssim", "valid_ssim", "test_psnr", "test_ssim"]
with open('cvpr/b2_rcan2.csv', 'w', encoding="utf-8") as myfile:
    writer = csv.DictWriter(myfile, fieldnames=headers)  # Use DictWriter
    writer.writeheader()  # Write the header row
    writer.writerows(scores)
# o means only the model mentioned, b2 means block 2
# b2_rcan122_o.csv is for rcan and b2_carn1_o.csv is for carn