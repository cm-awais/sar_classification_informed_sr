import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch
import torch.nn.functional as F
from torchsr.models import edsr, rcan, carn
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import gc
import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from cvpr_models import *



def get_classification_metrics(true_labels, predicted_labels):
    """
    Compute accuracy, precision, recall, and F1-score for each class.

    Args:
        true_labels (list): True class labels.
        predicted_labels (list): Predicted class labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score per class.
    """
    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Accuracy per class
    class_accuracies = (conf_matrix.diagonal() / conf_matrix.sum(axis=1)).tolist()

    # Precision, Recall, F1-score per class
    precision = precision_score(true_labels, predicted_labels, average=None).tolist()
    recall = recall_score(true_labels, predicted_labels, average=None).tolist()
    f1 = f1_score(true_labels, predicted_labels, average=None).tolist()

    # Prepare dictionary
    metrics = {
        "accuracy": class_accuracies,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    metrics_list = []
    for j in range(len(class_accuracies)):
        metrics_list.extend([class_accuracies[j], precision[j], recall[j], f1[j]])

    # metrics_list = [class_accuracies, precision, recall, f1]

    return metrics, metrics_list

def cal_scores(targets, predictions, check=False):
  if check:
    true_labels = [int(t.item()) for t in targets]  # Extract integer values
    predicted_labels = [int(p.item()) for p in predictions]

  scores = []
  # Generate classification report
  report, metrics_list = get_classification_metrics(true_labels, predicted_labels)

  overall_accuracy = round(accuracy_score(true_labels, predicted_labels), 4)* 100

  # Calculate overall precision, recall, and F1-score (weighted average)
  overall_precision = round(precision_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  overall_recall = round(recall_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  overall_f1 = round(f1_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  scores.extend([overall_accuracy, overall_precision, overall_recall, overall_f1])
  scores.extend(metrics_list)

  return {
      'class_metrics' : report,
      'overall_accuracy': overall_accuracy,
      'overall_precision': overall_precision,
      'overall_recall': overall_recall,
      'overall_f1': overall_f1
  }, scores


# Evaluation function 
def evaluate_model_cls(model, data_loader, device, feature_extractor=None):
    model.eval()
    model.to(device)
    saving_string = ""
    correct = 0
    total = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            # print(data.shape, target.shape)
            features = data
            if feature_extractor is not None:
                features = feature_extractor(data)  # uncomment
            
            output = model(features)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted)
            targets.extend(target)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    saving_string += f"Accuracy: {accuracy:.2f}% \n"
    # print(f"Accuracy: {accuracy:.2f}%")
    print()
    dicrt, scores = cal_scores(predictions=predictions, targets=targets, check=True)
    # print(dicrt)
    saving_string += json.dumps(dicrt, indent=4)
    return saving_string, scores, dicrt

# weights initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def save_model(model, scale, model_name, dataset_name, loss_name, output_dir="saved_models"):
    """
    Saves the trained model with a unique file name.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model to save.
        model_name (str): The name of the model (e.g., "edsr", "rcan").
        dataset_name (str): The name of the dataset used for training.
        loss_name (str): The name of the loss function used during training.
        output_dir (str): The directory where the model will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "cvpr"
    file_name = f"{model_name}_{str(scale)}_{dataset_name}_{loss_name}_{ext}.pth"
    file_path = os.path.join(output_dir, file_name)
    
    # Save the model
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to: {file_path}")

def load_model(model, scale, model_name, dataset_name, loss_name, output_dir="saved_models"):
    """
    Saves the trained model with a unique file name.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model to save.
        model_name (str): The name of the model (e.g., "edsr", "rcan").
        dataset_name (str): The name of the dataset used for training.
        loss_name (str): The name of the loss function used during training.
        output_dir (str): The directory where the model will be saved.
    """
    # Ensure the output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "cvpr"
    file_name = f"{model_name}_{str(scale)}_{dataset_name}_{loss_name}_{ext}.pth"
    file_path = os.path.join(output_dir, file_name)
    model.load_state_dict(torch.load(file_path, map_location=device))
    return model
    # # Save the model
    # torch.load(model.state_dict(), file_path)
    # print(f"Model saved to: {file_path}")

class SRDataset(Dataset):
    def __init__(self, hr_dir, ds_type="fusar"):
        """
        Custom Dataset for Super-Resolution tasks.
        Args:
            lr_dir (str): Directory containing low-resolution images.
            hr_dir (str): Directory containing high-resolution images.
            ds_type (bool): True for OpenSAR, False for Fusar.
        """
        self.hr_dir = hr_dir
        # self.lr_images = []
        self.hr_images = []
        self.ds_type = ds_type  # Dataset type flag

        # Walk through LR and HR directories to collect valid file paths
        for class_name in os.listdir(hr_dir):
            hr_class_dir = os.path.join(hr_dir, class_name)

            if not os.path.isdir(hr_class_dir):
                # print(f"Skipping non-directory: {hr_class_dir}")
                continue

            for img_name in os.listdir(hr_class_dir):
                hr_path = os.path.join(hr_class_dir, img_name)

                if self._is_valid_image(hr_path):
                    self.hr_images.append(hr_path)
                # else:
                #     print(f"Invalid image skipped: {hr_path}")

        print(f"Total valid HR images: {len(self.hr_images)}")

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        """
        Load and return a single LR and HR image pair.
        Args:
            idx (int): Index of the image pair to load.
        Returns:
            dict: A dictionary with 'lr' and 'hr' tensors.
        """
        # while True:  # Retry logic to handle problematic images
        hr_path = self.hr_images[idx]

        try:
            # Open images
            hr_img = Image.open(hr_path).convert("RGB")

            # Check image dimensions
            if hr_img.size[0] < 2 or hr_img.size[1] < 2:
                print(f"Image too small: {hr_path}")
                raise ValueError("Image dimensions too small")

            # Define transformations
            if self.ds_type == "opensar":  # OpenSAR
                lr_transform = Compose([Resize((32, 32)), ToTensor()])
                hr_transform = Compose([Resize((64, 64)), ToTensor()])
            elif self.ds_type == "fusar":  # Fusar 512x512
                lr_transform = Compose([Resize((128, 128)), ToTensor()])
                hr_transform = Compose([Resize((256, 256)), ToTensor()])
            elif self.ds_type == "hrsid":  # hrsid 800x800
                lr_transform = Compose([Resize((128, 128)), ToTensor()])
                hr_transform = Compose([Resize((256, 256)), ToTensor()])
            elif self.ds_type == "mstar":  # mstar
                lr_transform = Compose([Resize((64, 64)), ToTensor()])
                hr_transform = Compose([Resize((128, 128)), ToTensor()])
            
            # Pre processing add to overleaf!!

            # Apply transformations
            lr_img = lr_transform(hr_img)
            hr_img = hr_transform(hr_img)

            return {"lr": lr_img, "hr": hr_img}

        except (OSError, IOError, ValueError) as e:
            print(f"Skipping problematic image: {hr_path} due to error: {e}")
            # Remove problematic image paths
            # del self.lr_images[idx]
            # del self.hr_images[idx]

            # # Adjust index if necessary
            # if idx >= len(self.hr_images):
            #     idx -= 1

    @staticmethod
    def _is_valid_image(file_path):
        """
        Check if a file is a valid image.
        Args:
            file_path (str): Path to the image file.
        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify the image is valid
            with Image.open(file_path) as img:
                if img.size[0] < 2 or img.size[1] < 2:
                    print(f"Image too small: {file_path}")
                    return False
            return True
        except (OSError, IOError) as e:
            # print(f"Invalid image file: {file_path} - {e}")
            return False
        


def normalized_loss(l1_value, ssim_value, psnr_value, max_l1=10):
    alpha, beta, gamma = 0.7, 0.2, 0.1  # Tuned weights
    # l1_loss = l1_value / max_l1  # Normalize L1
    # ssim_loss = 1.0001 - ssim_value
    # psnr_loss = (40.0001 - psnr_value) / 40  # Normalize PSNR

    max_psnr = 40.0001

    l1_norm = l1_value / max_l1

    l1_loss = torch.clamp(l1_norm, 0, 10)
    ssim_value = (1.00001 - ssim_value) * 10
    ssim_loss = torch.clamp(ssim_value, 0, 10)  # Correct SSIM loss
    psnr_norma = ((max_psnr - psnr_value) / max_psnr) * 10
    psnr_loss = torch.clamp(psnr_norma, 0, 10)

    total_loss = alpha * l1_loss + beta * ssim_loss + gamma * psnr_loss

    return total_loss

class CombinedLoss(nn.Module):
    def __init__(self, psnr_weight=0.5, ssim_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight

    def forward(self, sr, hr):
        # PSNR Loss
        mse = F.mse_loss(sr, hr)
        # psnr = -10 * torch.log10(1 / (mse + 1e-8))  # PSNR as loss

        ssim_loss = (1.00001 - ssim(sr, hr, data_range=1.0)) * 10

        # # Combined loss
        # loss = (
        #     self.psnr_weight * (40.0001 - psnr) + self.ssim_weight * ssim_loss
        # )

        max_psnr = 40.00001

        # Robust and Normalized PSNR Loss (0-1 range)
        psnr = -10 * torch.log10(1 / (mse + 1e-8))
        normalized_psnr = ((max_psnr - psnr) / max_psnr) * 10
        normalized_psnr = torch.clamp(normalized_psnr, 0, 10)  # Crucial for stability

        # SSIM Loss (Standard, 0-1 range)
        # ssim_loss = 1 - SSIM()(sr, hr)
        ssim_loss = torch.clamp(ssim_loss, 0, 10) # Clipping

        # # MSE Loss (Normalized if needed)
        # normalized_mse = mse / (torch.max(mse) + 1e-8) if mse_weight > 0 else mse # Normalization to prevent domination.
        # normalized_mse = torch.clamp(normalized_mse, 0, 1)

        # Combined loss (weighted average of normalized losses)
        loss = (self.psnr_weight * normalized_psnr + 
                self.ssim_weight * ssim_loss)
                #  + mse_weight * normalized_mse)

        return loss

class CombinedLoss2(nn.Module):
    def __init__(self, psnr_weight=0.5, ssim_weight=0.5):
        super(CombinedLoss2, self).__init__()
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight

    def forward(self, sr, hr):
        # PSNR Loss
        mse = F.mse_loss(sr, hr)
        psnr = -10 * torch.log10(1 / (mse + 1e-8))  # PSNR as loss

        # SSIM Loss: Compute per image in the batch
        batch_size = sr.size(0)
        ssim_loss = 0
        for i in range(batch_size):
            # sr_np = sr[i].permute(1, 2, 0).cpu().numpy()  # Convert PyTorch tensor to NumPy (H, W, C)
            sr_np = sr[i].permute(1, 2, 0).cpu().detach().numpy()
            hr_np = hr[i].permute(1, 2, 0).cpu().numpy()  # Convert PyTorch tensor to NumPy (H, W, C)
            
            # Compute SSIM between the two images using ssim_1
            min_dim = min(sr_np.shape[0], sr_np.shape[1])  # Get the smallest image dimension
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  # Ensure `win_size` is odd
            ssim_value = ssim_1(
                sr_np, 
                hr_np, 
                data_range=1.0, 
                multichannel=True, 
                win_size=win_size, 
                channel_axis=-1
            )
            # ssim_value = ssim_1(sr_np, hr_np, data_range=1.0, multichannel=True)
            
            ssim_loss += 1 - ssim_value  # Invert SSIM to compute the loss
        ssim_loss /= batch_size  # Average SSIM loss over the batch
        # for i in range(batch_size):
        #     ssim_loss += 1 - ssim(
        #         sr[i].unsqueeze(0),  # Take the i-th image and keep batch dimension
        #         hr[i].unsqueeze(0),
        #         data_range=1.0,
        #     )
        # ssim_loss /= batch_size  # Average SSIM loss over the batch

        # Combined loss
        loss = (
            self.psnr_weight * psnr + self.ssim_weight * ssim_loss
        )
        return loss

class HybridLoss2(nn.Module):
    def __init__(self, l1_weight=1.0, psnr_weight=0.5, ssim_weight=0.5):
        super(HybridLoss2, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight

    def forward(self, sr, hr):
        # L1 Loss
        l1 = self.l1_loss(sr, hr)

        # PSNR Loss
        mse = F.mse_loss(sr, hr)
        psnr = -10 * torch.log10(1 / (mse + 1e-8))  # PSNR as loss

        # SSIM Loss: Compute per image in the batch
        batch_size = sr.size(0)
        ssim_loss = 0
        for i in range(batch_size):
            # sr_np = sr[i].permute(1, 2, 0).cpu().numpy()  # Convert PyTorch tensor to NumPy (H, W, C)
            sr_np = sr[i].permute(1, 2, 0).cpu().detach().numpy()
            hr_np = hr[i].permute(1, 2, 0).cpu().numpy()  # Convert PyTorch tensor to NumPy (H, W, C)
            
            # Compute SSIM between the two images using ssim_1
            min_dim = min(sr_np.shape[0], sr_np.shape[1])  # Get the smallest image dimension
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  # Ensure `win_size` is odd
            ssim_value = ssim_1(
                sr_np, 
                hr_np, 
                data_range=1.0, 
                multichannel=True, 
                win_size=win_size, 
                channel_axis=-1
            )

            # ssim_value = ssim_1(sr_np, hr_np, data_range=1.0, multichannel=True)
            
            ssim_loss += 1 - ssim_value  # Invert SSIM to compute the loss
        ssim_loss /= batch_size  # Average SSIM loss over the batch
        
        # ssim_loss = 0
        # for i in range(batch_size):
        #     ssim_loss += 1 - ssim(
        #         sr[i].unsqueeze(0),  # Take the i-th image and keep batch dimension
        #         hr[i].unsqueeze(0),
        #         data_range=1.0,
        #     )
        # ssim_loss /= batch_size  # Average SSIM loss over the batch

        # Combined loss
        loss = (
            self.l1_weight * l1
            + self.psnr_weight * psnr
            + self.ssim_weight * ssim_loss
        )
        return loss


class HybridLoss(nn.Module):
    def __init__(self, l1_weight=1.0, psnr_weight=0.5, ssim_weight=0.5):
        super(HybridLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight

    def forward(self, sr, hr):
        # L1 Loss
        l1 = self.l1_loss(sr, hr)

        # PSNR Loss
        mse = F.mse_loss(sr, hr)
        psnr = -10 * torch.log10(1 / (mse + 1e-8))  # PSNR as loss

        ssim_loss = ssim(sr, hr, data_range=1.0)

        # Combined loss
        loss = normalized_loss(l1_value=l1, psnr_value=psnr, ssim_value=ssim_loss)
        return loss

def train_base(loaders, scale=2):
    print("baseline")
    scores = []
    models = {
        "edsr" : lambda: edsr(scale=scale, pretrained=True),
        # "rcan" : lambda: rcan(scale=scale, pretrained=True),
        "carn" : lambda: carn(scale=scale, pretrained=True),
        }
    for sr_model_name, sr_model in models.items():
        print("Model: ",sr_model_name)
        for ds, loader in loaders.items():
            print("Dataset: ", ds)
            train_loaders = loader[0]
            valid_loaders = loader[1]
            test_loaders = loader[2]
            base_line_model = sr_model()
            
            base_line_model = base_line_model.to(device)
            ds = ["fusar", "opensar", "mstar", "hrsid"]
            for inde in range(len(train_loaders)):
                train_psnr, train_ssim = evaluate_model(base_line_model, train_loaders[inde])
                valid_psnr, valid_ssim = evaluate_model(base_line_model, valid_loaders[inde])
                test_psnr, test_ssim = evaluate_model(base_line_model, test_loaders[inde])

                print(f"{ds[inde]},{train_psnr:.4f}, {train_ssim:.4f}, {valid_psnr:.4f}, {valid_ssim:.4f}, {test_psnr:.4f}, {test_ssim:.4f},")

                scores.append([round(train_psnr, 4), round(train_ssim, 4), round(valid_psnr, 4), round(valid_ssim, 4), round(test_psnr, 4), round(test_ssim, 4)])
    print()

    return scores

def train_method(loaders, scale=2, b1=True):
    print("sr_training")
    scores = []
    if b1:
        # mn = ""
        # Initialize combined loss
        combined_loss = CombinedLoss(psnr_weight=0.5, ssim_weight=0.5).to(device)
        # Initialize hybrid loss
        hybrid_loss = HybridLoss(l1_weight=0.5, psnr_weight=0.3, ssim_weight=0.2).to(device)
    else:
        # mn = "1"
        # Initialize combined loss
        combined_loss = CombinedLoss2(psnr_weight=0.5, ssim_weight=0.5).to(device)
        # Initialize hybrid loss
        hybrid_loss = HybridLoss2(l1_weight=0.5, psnr_weight=0.3, ssim_weight=0.2).to(device)

    losses = {
            "L1":nn.L1Loss().to(device),
            "PNSR+SSIM": combined_loss,
            "hybrid": hybrid_loss}
    
    models = {
        # "edsr" : lambda: edsr(scale=scale, pretrained=True).to(device),
        # "rcan" : lambda: rcan(scale=scale, pretrained=True).to(device),
        "carn" : lambda: carn(scale=scale, pretrained=True).to(device),
        }
    dl_load = ["opensar", "mstar", "hrsid", "fusar"]
    scaler = GradScaler()

    for sr_model_name, sr_model_fn in models.items():
        print("Model: ",sr_model_name)
        for ln, loss in losses.items():
            print("Loss Function: ",ln)
            # ln += mn
            criterion = loss
            for ds, loader in loaders.items():
                # print("Dataset: ", ds)
                train_loaders, valid_loaders, test_loaders = loader
                # valid_loaders = loader[1]
                # test_loaders = loader[2]
                model = None
                model = sr_model_fn()  # For 2x upscaling
                # model.apply(initialize_weights)
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # model = model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-4)

                # Training loop with validation
                epochs = 10  # Adjust as needed
                # with tqdm(total=epochs, desc="Training Progress", unit="epoch") as epoch_pbar:
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0

                    for index in range(len(train_loaders)):
                        # print(dl_load[index])
                        train_loader = train_loaders[index]

                        # Wrap the training loop in tqdm
                        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                            for batch in train_loader:
                                lr = batch["lr"].to(device, non_blocking=True)
                                hr = batch["hr"].to(device, non_blocking=True)
                                
                                optimizer.zero_grad()

                                with autocast():
                                    sr = model(lr)
                                
                                with torch.cuda.amp.autocast(enabled=False):
                                    loss = criterion(sr.float(), hr.float())
                                # loss = combined_loss(sr, hr)
                                # loss = hybrid_loss(sr, hr)
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                                # loss.backward()
                                # optimizer.step()

                                train_loss += loss.item()

                                # Update tqdm progress bar
                                pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1)})
                                pbar.update(1)
                
                for index in range(len(valid_loaders)):
                    # print(dl_load[index])
                    train_loader = train_loaders[index]
                    # print(train_loader)
                    valid_loader = valid_loaders[index]
                    # print(len(valid_loader))
                    test_loader = test_loaders[index]
                    # print(test_loader)

                    # Evaluate on each subset
                    train_psnr, train_ssim = evaluate_model(model, train_loader)
                    valid_psnr, valid_ssim = evaluate_model(model, valid_loader)
                    test_psnr, test_ssim = evaluate_model(model, test_loader)

                    print(f"{dl_load[index]} , {ln} , {train_psnr:.4f}, {train_ssim:.4f}, {valid_psnr:.4f}, {valid_ssim:.4f}, {test_psnr:.4f}, {test_ssim:.4f},")
                    scores.append([round(train_psnr, 4), round(train_ssim, 4), round(valid_psnr, 4), round(valid_ssim, 4), round(test_psnr, 4), round(test_ssim, 4)])

                save_model(model, scale, sr_model_name, ds, ln)
                # # Cleanup
                # del model, optimizer, criterion, sr, loss
                # gc.collect()
                # torch.cuda.empty_cache()

                # Display results
                # print(f"{ln} , {ds} Train - PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f}")
                # print(f"{ln} , {ds} Valid - PSNR: {valid_psnr:.4f}, SSIM: {valid_ssim:.4f}")
                # print(f"{ln} , {ds} Test  - PSNR: {test_psnr:.4f}, SSIM: {test_ssim:.4f}")

    return scores

def finetune_w_classification(loaders, scale=2):
    print("sr_training")
    scores = []
    combined_loss = CombinedLoss2(psnr_weight=0.5, ssim_weight=0.5).to(device)
    hybrid_loss = HybridLoss2(l1_weight=0.5, psnr_weight=0.3, ssim_weight=0.2).to(device)

    losses = {
            "L1":nn.L1Loss().to(device),
            "PNSR+SSIM": combined_loss,
            "hybrid": hybrid_loss
            }
    
    models = {
        # "edsr" : lambda: edsr(scale=scale, pretrained=True).to(device),
        # "rcan" : lambda: rcan(scale=scale, pretrained=True).to(device),
        "carn" : lambda: carn(scale=scale, pretrained=True).to(device),
        }
    scaler = GradScaler()

    for sr_model_name, sr_model in models.items():
        model_factories = load_model_factory()
        print("Model: ",sr_model_name)
        for ln, loss in losses.items():
            print("Loss Function: ",ln)
            criterion = loss
            for ds, loader in loaders.items():
                print("Dataset: ", ds)
                train_loader = loader[0]
                valid_loader = loader[1]
                test_loader = loader[2]
                num_classes = loader[3]
                ds += "_hrsid_f_"
                sr_model_l = None
                sr_model_l = sr_model()  # For 2x upscaling
                sr_model_l = load_model(sr_model_l, scale, sr_model_name, "Concatenated", ln)
                
                sr_model_l = sr_model_l.to(device)
                optimizer = optim.Adam(sr_model_l.parameters(), lr=1e-4)
                comb_model_name = sr_model_name + "_ft_"

                # Preload classification models for the current dataset
                classifier_models = {}
                for cls_model_name, cls_model_factory in model_factories.items():
                    cls_model = cls_model_factory()
                    freeze_model_features(cls_model)
                    modify_classification_head(cls_model, num_classes)
                    cls_model.to(device)
                    cls_model.eval()  # Freeze evaluation mode since they are not being trained
                    classifier_models[cls_model_name] = cls_model
                
                
                # cls_criterion = nn.CrossEntropyLoss()
                cls_criterion = nn.MSELoss()


                # Training loop with validation
                epochs = 10  # Adjust as needed
                for epoch in range(epochs):
                    sr_model_l.train()
                    train_loss = 0
                    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                        for batch in train_loader:
                            lr = batch["lr"].to(device, non_blocking=True)
                            hr = batch["hr"].to(device, non_blocking=True)
                            # label = batch["label"].to(device)
                            cls_loss = 0.0 

                            optimizer.zero_grad()

                            # Forward pass in mixed precision
                            with autocast():
                                sr = sr_model_l(lr)
                                
                                # Accumulate classification losses over each preloaded classifier model
                                for cls_name, cls_model in classifier_models.items():
                                    cls_out_sr = cls_model(sr)
                                    cls_out_hr = cls_model(hr)
                                    # print("SR: ", cls_out_sr)
                                    # print("HR: ", cls_out_sr)
                                    cl_loss = cls_criterion(cls_out_sr, cls_out_hr)
                                    cls_loss += cl_loss
                                    # print("Classification Loss: ",cls_name , cl_loss.item())
        
                            # Average the classification loss dynamically
                            cls_loss /= len(classifier_models)

                            with torch.cuda.amp.autocast(enabled=False):
                                loss = criterion(sr.float(), hr.float())

                            # print("Classificaiton Loss: ", cls_loss.item())    
                            # print("SR Loss: ", loss.item())    
                            combo_loss = cls_loss + loss
                            # combo_loss = loss
                            scaler.scale(combo_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            train_loss += combo_loss.item()

                            # Update tqdm progress bar
                            pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1)})
                            pbar.update(1)

                # Evaluate on each subset
                train_psnr, train_ssim = evaluate_model(sr_model_l, train_loader)
                valid_psnr, valid_ssim = evaluate_model(sr_model_l, valid_loader)
                test_psnr, test_ssim = evaluate_model(sr_model_l, test_loader)

                print(f"{ds} , {comb_model_name} ,{ln} , {train_psnr:.4f}, {train_ssim:.4f}, {valid_psnr:.4f}, {valid_ssim:.4f}, {test_psnr:.4f}, {test_ssim:.4f},")
                scores.append([round(train_psnr, 4), round(train_ssim, 4), round(valid_psnr, 4), round(valid_ssim, 4), round(test_psnr, 4), round(test_ssim, 4)])

                save_model(sr_model_l, scale, comb_model_name, ds, ln)
    return scores

def finetune_w_classification2(loaders, models, scale=2):
    print("sr_training_2")
    scores = []
    combined_loss = CombinedLoss2(psnr_weight=0.5, ssim_weight=0.5).to(device)
    hybrid_loss = HybridLoss2(l1_weight=0.5, psnr_weight=0.3, ssim_weight=0.2).to(device)

    losses = {
            "L1":nn.L1Loss().to(device),
            "PNSR+SSIM": combined_loss,
            "hybrid": hybrid_loss
            }
    
    # models = {
    #     # "edsr" : lambda: edsr(scale=scale, pretrained=True).to(device),
    #     # "rcan" : lambda: rcan(scale=scale, pretrained=True).to(device),
    #     "carn" : lambda: carn(scale=scale, pretrained=True).to(device),
    #     }
    scaler = GradScaler()

    train_loaders = []
    test_loaders = []
    valid_loaders = []

    for ds, loader in loaders.items():
        # print("Dataset: ", ds)
        num_classes = loader[3]
        train_loaders.append([ds, num_classes, loader[0]])
        valid_loaders.append(loader[1])
        test_loaders.append(loader[2])
    
    classifier_models = {}
    cls_model_factories = load_model_factory()
    for cls_model_name, cls_model_factory in cls_model_factories.items():
        cls_model = cls_model_factory()
        freeze_model_features(cls_model)
        classifier_models[cls_model_name] = cls_model                        

    for sr_model_name, sr_model in models.items():
        print("Model: ",sr_model_name)
        for ln, loss in losses.items():
            print("Loss Function: ",ln)
            criterion = loss
            sr_model_l = None
            sr_model_l = sr_model()  # For 2x upscaling
            sr_model_l = load_model(sr_model_l, scale, sr_model_name, "Concatenated", ln)
            
            sr_model_l = sr_model_l.to(device)
            optimizer = optim.Adam(sr_model_l.parameters(), lr=1e-4)
            comb_model_name = sr_model_name + "_ft2_"
            
            
            # cls_criterion = nn.CrossEntropyLoss()
            cls_criterion = nn.MSELoss()

            # Training loop with validation
            epochs = 10  # Adjust as needed
            # val_loaders = 
            for epoch in range(epochs):
                sr_model_l.train()
                train_loss = 0
                for ds, num_classes, train_loader in train_loaders:
                    print("Dataset: ", ds)
                    ds += "_hrsid_f_"
                        
                    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                        for batch in train_loader:
                            lr = batch["lr"].to(device, non_blocking=True)
                            hr = batch["hr"].to(device, non_blocking=True)
                            cls_loss = 0.0 

                            optimizer.zero_grad()

                            # Forward pass in mixed precision
                            with autocast():
                                sr = sr_model_l(lr)
                                
                                # Accumulate classification losses over each preloaded classifier model
                                for cls_name, cls_model in classifier_models.items():
                                    modify_classification_head(cls_model, num_classes)
                                    cls_model.to(device)
                                    cls_model.eval()  # Use in evaluation mode since they are not being trained
                                    cls_out_sr = cls_model(sr)
                                    cls_out_hr = cls_model(hr)
                                    cl_loss = cls_criterion(cls_out_sr, cls_out_hr)
                                    cls_loss += cl_loss
        
                            # Average the classification loss dynamically
                            cls_loss /= len(classifier_models)

                            with torch.cuda.amp.autocast(enabled=False):
                                loss = criterion(sr.float(), hr.float())
                            
                            combo_loss = cls_loss + loss
                            # combo_loss = loss
                            scaler.scale(combo_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            train_loss += combo_loss.item()

                            # Update tqdm progress bar
                            pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1)})
                            pbar.update(1)
                    # valid_psnr, valid_ssim = evaluate_model(sr_model_l, valid_loaders[index])

            for index in range(len(valid_loaders)):
                # Evaluate on each subset
                train_psnr, train_ssim = evaluate_model(sr_model_l, train_loaders[index][2])
                valid_psnr, valid_ssim = evaluate_model(sr_model_l, valid_loaders[index])
                test_psnr, test_ssim = evaluate_model(sr_model_l, test_loaders[index])

                print(f"{ds} , {comb_model_name} ,{ln} , {train_psnr:.4f}, {train_ssim:.4f}, {valid_psnr:.4f}, {valid_ssim:.4f}, {test_psnr:.4f}, {test_ssim:.4f},")
                scores.append([round(train_psnr, 4), round(train_ssim, 4), round(valid_psnr, 4), round(valid_ssim, 4), round(test_psnr, 4), round(test_ssim, 4)])

            save_model(sr_model_l, scale, comb_model_name, "ft", ln)
    return scores


def b3_classification(loaders, models, scale=2): # trained models
    print("Pretrained Models")
    scores = []
    combined_loss = CombinedLoss2(psnr_weight=0.5, ssim_weight=0.5).to(device)
    hybrid_loss = HybridLoss2(l1_weight=0.5, psnr_weight=0.3, ssim_weight=0.2).to(device)

    losses = {
            "L1":nn.L1Loss().to(device),
            "PNSR+SSIM": combined_loss,
            "hybrid": hybrid_loss
            }
    csv_path = "cvpr/"
    scaler = GradScaler()
    csv1 = []

    for sr_model_name, sr_model in models.items():
        model_factories = load_model_factory()
        print("Model: ",sr_model_name)
        csv_path += sr_model_name + "_"
        for ln, _ in losses.items():
            print("Loss Function: ",ln)
            # criterion = loss
            sr_model_l = None
            sr_model_l = sr_model()  # For 2x upscaling
            # sr_model_l = load_model(sr_model_l, scale, sr_model_name, "Concatenated", ln)
            # loading fine tuned "_ft2_"
            sr_model_l = load_model(sr_model_l, scale, sr_model_name + "_ft2_", "ft", ln)

            
            sr_model_l = sr_model_l.to(device)
            sr_model_l.eval()
            
            for ds, loader in loaders.items():
                print("Dataset: ", ds)
                csv_path += ds + "_"
                train_loader = loader[0]
                valid_loader = loader[1]
                test_loader = loader[2]
                num_classes = loader[3]
                # ds += "_hrsid_f_"
                # optimizer = optim.Adam(sr_model_l.parameters(), lr=1e-4)
                # comb_model_name = sr_model_name + "_ft_"

                for cls_model_name, cls_model_factory in model_factories.items():
                    cls_model = cls_model_factory()
                    freeze_model_features(cls_model)
                    modify_classification_head_o3(cls_model, num_classes)
                    cls_model.to(device)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cls_model.parameters()), lr=0.0001)
                    cls_criterion = nn.CrossEntropyLoss()
                    comb_model_name = sr_model_name + "_" + cls_model_name
                    csv_head = comb_model_name + "_" + ds + "_" + ln + "_"

                    # Training loop with validation
                    epochs = 10  # Adjust as needed
                    for epoch in range(epochs):
                        cls_model.train()
                        train_loss = 0
                        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                            for inputs, labels in train_loader:
                                inputs, labels = inputs.to(device), labels.to(device)
                                # label = batch["label"].to(device)
                                cls_loss = 0.0 

                                optimizer.zero_grad()

                                # Forward pass in mixed precision
                                with autocast():
                                    sr = sr_model_l(inputs)
                                    cls_out_sr = cls_model(sr)

                                with torch.cuda.amp.autocast(enabled=False):
                                    cls_loss = cls_criterion(cls_out_sr, labels)
                                # combo_loss = loss
                                scaler.scale(cls_loss).backward()
                                scaler.step(optimizer)
                                scaler.update()

                                train_loss += cls_loss.item()

                                # Update tqdm progress bar
                                pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1)})
                                pbar.update(1)

                        _, _, d_scrs = evaluate_model_cls(cls_model, valid_loader, device)
                        # print("Evaluation: ",d_scrs)

                    # Evaluate on each subset
                    _, csv_scores, dict_scores = evaluate_model_cls(cls_model, test_loader, device)

                    csv_scores = [csv_head] + csv_scores

                    print(f"{ds} , {comb_model_name} ,{ln} , {dict_scores}")
                    csv1.append(csv_scores)
    fields = ["Header"] + ["Accuracy", "Precision", "Recall", "F1"] * (num_classes +1)

    # Write results to CSV
    # csv_path = "cvpr/cls_scores_b1.csv" # classification scores on trained sr models
    csv_path += "b1_2x_new.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(csv1)
    # return csv1

def b3_classification_ft(loaders, models, scale=2):  # fine tuned models
    print("Fine tuned models")
    scores = []
    combined_loss = CombinedLoss2(psnr_weight=0.5, ssim_weight=0.5).to(device)
    hybrid_loss = HybridLoss2(l1_weight=0.5, psnr_weight=0.3, ssim_weight=0.2).to(device)

    losses = {
            "L1":nn.L1Loss().to(device),
            "PNSR+SSIM": combined_loss,
            "hybrid": hybrid_loss
            }
    
    scaler = GradScaler()
    csv_path = "cvpr/"
    csv1 = []

    for sr_model_name, sr_model in models.items():
        model_factories = load_model_factory()
        print("Model: ",sr_model_name)
        csv_path += sr_model_name + "_"
        for ln, _ in losses.items():
            print("Loss Function: ",ln)
            # criterion = loss
            
            for ds, loader in loaders.items():
                print("Dataset: ", ds)
                csv_path += ds + "_"
                train_loader = loader[0]
                valid_loader = loader[1]
                test_loader = loader[2]
                num_classes = loader[3]
                ds += "_hrsid_f_"
                # optimizer = optim.Adam(sr_model_l.parameters(), lr=1e-4)
                comb_model_name_1 = sr_model_name + "_ft_"

                sr_model_l = None
                sr_model_l = sr_model()  # For 2x upscaling
                sr_model_l = load_model(sr_model_l, scale, comb_model_name_1, ds, ln)
                
                sr_model_l = sr_model_l.to(device)
                sr_model_l.eval()

                for cls_model_name, cls_model_factory in model_factories.items():
                    cls_model = cls_model_factory()
                    freeze_model_features(cls_model)
                    modify_classification_head_o3(cls_model, num_classes)
                    cls_model.to(device)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cls_model.parameters()), lr=0.0001)
                    cls_criterion = nn.CrossEntropyLoss()
                    comb_model_name = sr_model_name + "_" + cls_model_name
                    csv_head = comb_model_name + "_" + ds + "_" + ln + "_"

                    # Training loop with validation
                    epochs = 10  # Adjust as needed
                    for epoch in range(epochs):
                        cls_model.train()
                        train_loss = 0
                        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                            for inputs, labels in train_loader:
                                inputs, labels = inputs.to(device), labels.to(device)
                                # label = batch["label"].to(device)
                                cls_loss = 0.0 

                                optimizer.zero_grad()

                                # Forward pass in mixed precision
                                with autocast():
                                    sr = sr_model_l(inputs)
                                    cls_out_sr = cls_model(sr)

                                with torch.cuda.amp.autocast(enabled=False):
                                    cls_loss = cls_criterion(cls_out_sr, labels)
                                # combo_loss = loss
                                scaler.scale(cls_loss).backward()
                                scaler.step(optimizer)
                                scaler.update()

                                train_loss += cls_loss.item()

                                # Update tqdm progress bar
                                pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1)})
                                pbar.update(1)

                        _, _, d_scrs = evaluate_model_cls(cls_model, valid_loader, device)
                        # print("Evaluation: ",d_scrs)

                    # Evaluate on each subset
                    _, csv_scores, dict_scores = evaluate_model_cls(cls_model, test_loader, device)

                    csv_scores = [csv_head] + csv_scores

                    print(f"{ds} , {comb_model_name} ,{ln} , {dict_scores}")
                    csv1.append(csv_scores)
    fields = ["Header"] + ["Accuracy", "Precision", "Recall", "F1"] * (num_classes +1)

    # Write results to CSV
    # csv_path = "cvpr/cls_scores_b2.csv" # classification scores on fine tuned models
    csv_path += "b2_4x.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(csv1)


def b3_classification_orig(loaders, models, scale=2):    
    scaler = GradScaler()
    csv_path = "cvpr/" # classification scores on original models
    csv1 = []
    print("Original Models without pretraining!")

    for sr_model_name, sr_model in models.items():
        model_factories = load_model_factory()
        print("Model: ",sr_model_name)
        csv_path += sr_model_name + "_"
        for ds, loader in loaders.items():
            print("Dataset: ", ds)
            csv_path += ds + "_"
            train_loader = loader[0]
            valid_loader = loader[1]
            test_loader = loader[2]
            num_classes = loader[3]
            csv_path += ds + "_"

            sr_model_l = None
            sr_model_l = sr_model()  # For 2x upscaling
            
            sr_model_l = sr_model_l.to(device)
            sr_model_l.eval()

            for cls_model_name, cls_model_factory in model_factories.items():
                cls_model = cls_model_factory()
                freeze_model_features(cls_model)
                modify_classification_head_o3(cls_model, num_classes)
                cls_model.to(device)
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cls_model.parameters()), lr=0.0001)
                cls_criterion = nn.CrossEntropyLoss()
                comb_model_name = sr_model_name + "_" + cls_model_name
                csv_head = comb_model_name + "_" + ds + "_"

                # Training loop with validation
                epochs = 10  # Adjust as needed
                for epoch in range(epochs):
                    cls_model.train()
                    train_loss = 0
                    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                        for inputs, labels in train_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            # label = batch["label"].to(device)
                            cls_loss = 0.0 

                            optimizer.zero_grad()

                            # Forward pass in mixed precision
                            with autocast():
                                sr = sr_model_l(inputs)
                                cls_out_sr = cls_model(sr)

                            with torch.cuda.amp.autocast(enabled=False):
                                cls_loss = cls_criterion(cls_out_sr, labels)
                            # combo_loss = loss
                            scaler.scale(cls_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            train_loss += cls_loss.item()

                            # Update tqdm progress bar
                            pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1)})
                            pbar.update(1)

                    _, _, d_scrs = evaluate_model_cls(cls_model, valid_loader, device)
                    # print("Evaluation: ",d_scrs)

                # Evaluate on each subset
                _, csv_scores, dict_scores = evaluate_model_cls(cls_model, test_loader, device)

                csv_scores = [csv_head] + csv_scores

                print(f"{ds} , {comb_model_name}, {dict_scores}")
                csv1.append(csv_scores)
    fields = ["Header"] + ["Accuracy", "Precision", "Recall", "F1"] * (num_classes +1)

    # Write results to CSV
    csv_path += "b0_4x.csv"
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(csv1)

def b3_classification_no_sr(loaders, scale=2):    
    scaler = GradScaler()
    csv1 = []
    for ds, loader in loaders.items():
        print("Dataset: ", ds)
        train_loader = loader[0]
        valid_loader = loader[1]
        test_loader = loader[2]
        num_classes = loader[3]
        model_factories = load_model_factory()

        for cls_model_name, cls_model_factory in model_factories.items():
            cls_model = cls_model_factory()
            freeze_model_features(cls_model)
            modify_classification_head_o3(cls_model, num_classes)
            cls_model.to(device)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cls_model.parameters()), lr=0.0001)
            cls_criterion = nn.CrossEntropyLoss()
            comb_model_name = cls_model_name
            csv_head = comb_model_name + "_" + ds + "_"

            # Training loop with validation
            epochs = 10  # Adjust as needed
            for epoch in range(epochs):
                cls_model.train()
                train_loss = 0
                with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        # label = batch["label"].to(device)
                        cls_loss = 0.0 

                        optimizer.zero_grad()

                        # Forward pass in mixed precision
                        with autocast():
                            cls_out_sr = cls_model(inputs)

                        with torch.cuda.amp.autocast(enabled=False):
                            cls_loss = cls_criterion(cls_out_sr, labels)
                        # combo_loss = loss
                        scaler.scale(cls_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        train_loss += cls_loss.item()

                        # Update tqdm progress bar
                        pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1)})
                        pbar.update(1)

                _, _, d_scrs = evaluate_model_cls(cls_model, valid_loader, device)
                # print("Evaluation: ",d_scrs)

            # Evaluate on each subset
            _, csv_scores, dict_scores = evaluate_model_cls(cls_model, test_loader, device)

            csv_scores = [csv_head] + csv_scores

            print(f"{ds} , {comb_model_name}, {dict_scores}")
            csv1.append(csv_scores)
    fields = ["Header"] + ["Accuracy", "Precision", "Recall", "F1"] * (num_classes +1)

    # Write results to CSV
    csv_path = "cvpr/cls_scores_no_sr_2.csv" # classification scores on original models
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(csv1)
