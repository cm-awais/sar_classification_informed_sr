import torch
import numpy as np
import warnings
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import csv

from skimage.metrics import structural_similarity as ssim_1
from skimage.metrics import peak_signal_noise_ratio as psnr_1
warnings.filterwarnings("ignore")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

def calculate_metrics(sr_image, hr_image):
    """
    Calculate PSNR and SSIM between super-resolved and high-resolution images.
    Args:
        sr_image (numpy array): Super-resolved image.
        hr_image (numpy array): High-resolution ground truth image.
    Returns:
        psnr_value (float): Peak Signal-to-Noise Ratio.
        ssim_value (float): Structural Similarity Index.
    """
    sr_image = np.array(sr_image, dtype=np.float32) / 255.0
    hr_image = np.array(hr_image, dtype=np.float32) / 255.0

    psnr_value = psnr_1(hr_image, sr_image, data_range=1.0)
    ssim_value = ssim_1(hr_image, sr_image, multichannel=True, data_range=1.0, win_size=3)

    return psnr_value, ssim_value

def validate_model(model, dataloader, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss += loss.item()

    return val_loss / len(dataloader)

def evaluate_model(model, dataloader):
    """
    Evaluate the model and calculate average PSNR and SSIM.
    Args:
        model: The super-resolution model.
        dataloader: DataLoader containing the dataset to evaluate.
    Returns:
        avg_psnr (float): Average PSNR over the dataset.
        avg_ssim (float): Average SSIM over the dataset.
    """
    model.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    model = model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            # print()

            sr = model(lr).cpu() #.squeeze().permute(0, 2, 3, 1)  # Convert to HWC
            # print("SR-Shape: ",sr.shape)
            # print("SR- Size(): ",sr.size())
            hr = hr.cpu() #.squeeze().permute(0, 2, 3, 1)  # Convert to HWC
            # print("HR-shape, size", hr.shape, hr.size())

            for i in range(sr.size(0)):
                sr_img = sr[i].numpy()
                hr_img = hr[i].numpy()

                sr_img = (sr_img * 255).clip(0, 255).astype(np.uint8)
                hr_img = (hr_img * 255).clip(0, 255).astype(np.uint8)

                psnr_value, ssim_value = calculate_metrics(sr_img, hr_img)
                total_psnr += psnr_value
                total_ssim += ssim_value
                count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim


# =====================
# Utility Functions
# =====================

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_factory():
    """Create a dictionary of model factories."""
    return {
        "resnet18": lambda: torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True),
        "vgg16": lambda: torch.hub.load("pytorch/vision:v0.10.0", "vgg16", pretrained=True),
        "mobilenet_v2": lambda: torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True),
        "resnet50": lambda: torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
        "densenet121": lambda: torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=True),
        # "efficientnet_b0": lambda: torch.hub.load("pytorch/vision:v0.10.0", "efficientnet_b0", pretrained=True),
    }


def freeze_model_features(model):
    """Freeze all feature extraction layers in the model."""
    for param in model.parameters():
        param.requires_grad = False


def modify_classification_head_o(model, num_classes): # old
    """Modify the classification head of a model to match the dataset's number of classes."""
    if hasattr(model, "fc"):  # For models like ResNet
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):  # For models like VGG, DenseNet
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)

def modify_classification_head_o1(model, num_classes):
    """
    Modify the classification head of a model to use a new sequential block:
    
        Linear(in_features, 4096) -> ReLU -> Dropout -> Linear(4096, num_classes)
    
    where in_features is determined from the existing classifier.
    """
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            first_layer = model.classifier[0]
            if isinstance(first_layer, nn.Linear):
                in_features = first_layer.in_features
            else:
                raise ValueError("The first module in model.classifier is not Linear; cannot extract in_features.")
        else:
            in_features = model.classifier.in_features
    else:
        raise ValueError("The model does not have a recognized classification head (fc or classifier).")
    
    new_classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, num_classes)
    )
    
    if hasattr(model, 'classifier'):
        model.classifier = new_classifier
    elif hasattr(model, 'fc'):
        model.fc = new_classifier

def modify_classification_head_o3(model, num_classes):
    """
    Modify the classification head of a model to use a new sequential block:
    
        Linear(in_features, 4096) -> ReLU -> Dropout -> Linear(4096, num_classes)
    
    where in_features is determined from the existing classifier.
    """
    if hasattr(model, "fc"):  # For models like ResNet
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(4096, num_classes)
                    )
    elif hasattr(model, "classifier"):  # For models like VGG, DenseNet
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            model.classifier[-1] = nn.Sequential(
                        nn.Linear(model.classifier[-1].in_features, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(4096, num_classes)
                    )
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            model.classifier = nn.Sequential(
                        nn.Linear(model.classifier.in_features, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(4096, num_classes)
                    )

def modify_classification_head(model, num_classes):
    if hasattr(model, "fc"):  # For models like ResNet
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # If you want to train the new head, ensure its parameters are trainable:
        for param in model.fc.parameters():
            param.requires_grad = False
    elif hasattr(model, "classifier"):  # For models like VGG, DenseNet
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            for param in model.classifier[-1].parameters():
                param.requires_grad = False
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            for param in model.classifier.parameters():
                param.requires_grad = False


def save_results_to_csv(results, output_file):
    """Save results to a CSV file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(results[0].keys())  # Write header
        writer.writerows([result.values() for result in results])
    print(f"Results saved to {output_file}")


# =====================
# Testing Loop
# =====================

def test_model(model, test_loader, device):
    """Test the model on the test dataset and compute metrics."""
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    precision_micro = precision_score(all_labels, all_preds, average="micro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    recall_micro = recall_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }


# =====================
# Main Training and Testing
# =====================

def main():
    # Set seed and device
    set_seed(42)
    device = get_device()

    # Load model factories and DataLoaders
    model_factories = load_model_factory()
    num_classes = 10  # Example: Adjust based on your dataset
    num_epochs = 10
    learning_rate = 1e-4

    # Assume `dataloaders_dict` is already defined
    # Structure: { "DS1": {"train": train_loader, "valid": valid_loader, "test": test_loader}, ... }

    training_results = []  # Store training and validation results
    testing_results = []   # Store testing results for all models and datasets

    for model_name, model_factory in model_factories.items():
        for dataset_name, splits in dataloaders_dict.items():
            print(f"Training {model_name} on {dataset_name}")

            # Load and prepare model
            model = model_factory()
            freeze_model_features(model)
            modify_classification_head(model, num_classes)
            model.to(device)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            train_loader = splits["train"]
            valid_loader = splits["valid"]
            test_loader = splits["test"]

            # Training loop
            for epoch in range(num_epochs):
                train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
                valid_loss, valid_accuracy = validate_one_epoch(model, valid_loader, criterion, device)

                training_results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy,
                })

                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                      f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%")

            # Testing loop
            print(f"Testing {model_name} on {dataset_name}")
            test_metrics = test_model(model, test_loader, device)
            test_metrics.update({"model": model_name, "dataset": dataset_name})
            testing_results.append(test_metrics)

    # Save results
    save_results_to_csv(training_results, "training_results.csv")
    save_results_to_csv(testing_results, "testing_results.csv")


if __name__ == "__main__":
    # main()
    print("Models")
