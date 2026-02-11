# full_pipeline_cnn_lnn_figures_v3.py
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from PIL import Image

pl.seed_everything(42)

# ============================
# 1. LNN MODEL
# ============================

class LiquidCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x, h):
        tau = torch.sigmoid(self.tau(x)) + 1e-3
        dh = (-h + self.activation(self.W_in(x) + self.W_rec(h))) / tau
        h = h + dh
        return h

class LiquidLayer(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len=6):
        super().__init__()
        self.cell = LiquidCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.seq_len = seq_len

    def forward(self, x):
        B = x.size(0)
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for _ in range(self.seq_len):
            h = self.cell(x, h)
        return h

class CNN_LNN(nn.Module):
    def __init__(self, backbone_name, num_classes, seq_len=6):
        super().__init__()
        # Select backbone
        if backbone_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            cnn_out = 1280
        elif backbone_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            cnn_out = 2048
        elif backbone_name == "densenet121":
            base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            cnn_out = 1024
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        # Extract feature layers only
        if "resnet" in backbone_name:
            self.cnn = nn.Sequential(*list(base.children())[:-1])
        elif "efficientnet" in backbone_name:
            self.cnn = nn.Sequential(*list(base.children())[:-1])
        elif "densenet" in backbone_name:
            self.cnn = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
        
        self.liquid = LiquidLayer(cnn_out, 256, seq_len)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(-1).squeeze(-1)
        h = self.liquid(x)
        return self.fc(h)

class LitModel(pl.LightningModule):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN_LNN(backbone_name, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# ============================
# 2. DATA LOADING
# ============================

def load_dataset(data_dir, batch_size=16):
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(data_dir, transform=train_transform)
    class_names = dataset.classes
    train_size = int(0.8*len(dataset))
    val_size = len(dataset)-train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, val_dataset, class_names, val_transform

# ============================
# 3. EVALUATION & FIGURE GENERATION
# ============================

def evaluate_model(model, loader, class_names, device):
    model.eval()
    all_preds, all_labels, all_outputs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)
            all_outputs.append(outputs.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_outputs = torch.cat(all_outputs).numpy()
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    df_metrics = pd.DataFrame({"Class":class_names, "Precision":precision, "Recall":recall, "F1 Score":f1})
    print(df_metrics)
    return all_preds, all_labels, all_outputs, df_metrics


def get_target_cnn_layer(model, backbone_name):
    """
    Selects a spatially rich CNN layer for CAM methods.
    """
    cnn = model.model.cnn  # unwrap Lightning → CNN_LNN → cnn

    if backbone_name == "resnet50":
        # cnn = [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
        return cnn[6]  # layer3

    elif backbone_name == "efficientnet_b0":
        # cnn[0] = EfficientNet.features
        return cnn[0][4]  # earlier MBConv block

    elif backbone_name == "densenet121":
        # cnn[0] = DenseNet.features
        return cnn[0][-2]  # last dense block (before norm)

    else:
        raise ValueError("Unsupported backbone for CAM")


import torch.nn.functional as F

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def generate(self, x, class_idx):
        self.model.zero_grad()

        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        grads = self.gradients          # [B, C, H, W]
        acts  = self.activations        # [B, C, H, W]

        grads2 = grads ** 2
        grads3 = grads ** 3

        eps = 1e-8
        alpha = grads2 / (
            2 * grads2 +
            (acts * grads3).sum(dim=(2,3), keepdim=True) +
            eps
        )

        positive_grads = F.relu(grads)
        weights = (alpha * positive_grads).sum(dim=(2,3), keepdim=True)

        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)

        # Upsample to input resolution
        cam = cam.unsqueeze(1)
        cam = F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        cam = cam.squeeze(1)

        return cam



def save_gradcampp_samples(model, dataset, all_preds, all_labels,
                           class_names, device, save_dir,
                           correct=True, num_samples=6):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # -------------------------
    # Backbone-specific layers
    # -------------------------
    backbone = model.hparams.backbone_name
    cnn = model.model.cnn

    if backbone == "efficientnet_b0":
        target_layer = cnn[0][4]          # early MBConv
    elif backbone == "resnet50":
        target_layer = cnn[-2]            # layer4
    elif backbone == "densenet121":
        # Select target CNN layer safely for Grad-CAM++
        target_layer = get_target_cnn_layer(model, backbone)
    else:
        raise ValueError("Unsupported backbone for Grad-CAM++")

    campp = GradCAMPlusPlus(model, target_layer)

    indices = [
        i for i in range(len(all_preds))
        if (all_preds[i] == all_labels[i]) == correct
    ]

    if len(indices) == 0:
        campp.remove_hooks()
        return

    chosen = np.random.choice(indices, min(num_samples, len(indices)), replace=False)

    for idx in chosen:
        img, label = dataset[idx]
        x = img.unsqueeze(0).to(device)
        pred = all_preds[idx]

        cam = campp.generate(x, pred)[0].detach().cpu().numpy()
        img_np = img.permute(1,2,0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        plt.figure(figsize=(9,3))

        plt.subplot(1,3,1)
        plt.imshow(img_np)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(cam, cmap="jet")
        plt.title("Grad-CAM++")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(img_np)
        plt.imshow(cam, cmap="jet", alpha=0.5)
        plt.title(f"T:{class_names[label]}\nP:{class_names[pred]}")
        plt.axis("off")

        fname = f"{'correct' if correct else 'wrong'}_gradcampp_{idx}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()

    campp.remove_hooks()



def save_figures(backbone_dir, model, loader, dataset, all_preds, all_labels, all_outputs,
                 class_names, device, val_transform, logger_dir,backbone_name):
    
    os.makedirs(backbone_dir, exist_ok=True)

    # -------------------------------
    # 1. Training & Validation Curves
    # -------------------------------
    metrics_file = os.path.join(logger_dir, "metrics.csv")
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)

        # Handle different column names used by Lightning
        train_loss_col = "train_loss" if "train_loss" in metrics_df.columns else "train_loss_step"
        train_acc_col = "train_acc" if "train_acc" in metrics_df.columns else "train_acc_step"
        val_loss_col = "val_loss_epoch" if "val_loss_epoch" in metrics_df.columns else "val_loss"
        val_acc_col = "val_acc_epoch" if "val_acc_epoch" in metrics_df.columns else "val_acc"

        # Loss Curve
        plt.figure(figsize=(8,5))
        plt.plot(metrics_df[train_loss_col], label="Train Loss")
        plt.plot(metrics_df[val_loss_col], label="Val Loss")
        plt.xlabel("Step / Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(backbone_dir,"train_val_loss.png"))
        plt.close()

        # Accuracy Curve
        plt.figure(figsize=(8,5))
        plt.plot(metrics_df[train_acc_col], label="Train Acc")
        plt.plot(metrics_df[val_acc_col], label="Val Acc")
        plt.xlabel("Step / Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training & Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(backbone_dir,"train_val_acc.png"))
        plt.close()

    # -------------------------------
    # 2 & 3. Confusion Matrices
    # -------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(backbone_dir,"confusion_matrix.png"))
    plt.close()

    cm_norm = cm.astype(float)/cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.savefig(os.path.join(backbone_dir,"confusion_matrix_normalized.png"))
    plt.close()

    # -------------------------------
    # 4. Per-Class Error Rate
    # -------------------------------
    error_rate = 1 - np.diag(cm)/cm.sum(axis=1)
    plt.figure(figsize=(6,4))
    sns.barplot(x=class_names, y=error_rate)
    plt.ylabel("Error Rate")
    plt.title("Per-Class Error Rate")
    plt.savefig(os.path.join(backbone_dir,"per_class_error_rate.png"))
    plt.close()

    # -------------------------------
    # 5. ROC Curves
    # -------------------------------
    y_true_bin = label_binarize(all_labels, classes=range(len(class_names)))
    plt.figure(figsize=(6,5))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_outputs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(os.path.join(backbone_dir,"roc_curve.png"))
    plt.close()

    # -------------------------------
    # 6. Sample Predictions
    # -------------------------------
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    indices = np.random.choice(len(dataset), 9, replace=False)
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        pred = all_preds[idx]
        plt.subplot(3,3,i+1)
        plt.imshow(img.permute(1,2,0))
        plt.title(f"T:{class_names[label]}\nP:{class_names[pred]}")
        plt.axis("off")
    plt.savefig(os.path.join(backbone_dir,"sample_predictions.png"))
    plt.close()

    # -------------------------------
    # 7. t-SNE Features
    # -------------------------------
    features, labels_tsne = [], []
    with torch.no_grad():
        for imgs, labels_batch in loader:
            imgs = imgs.to(device)
            feat = model.model.cnn(imgs).squeeze(-1).squeeze(-1)
            features.append(feat.cpu())
            labels_tsne.append(labels_batch)
    features = torch.cat(features).numpy()
    labels_tsne = torch.cat(labels_tsne).numpy()
    tsne_proj = TSNE(n_components=2, random_state=42).fit_transform(features)
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=tsne_proj[:,0], y=tsne_proj[:,1], hue=[class_names[i] for i in labels_tsne], palette="tab10")
    plt.title("t-SNE Feature Embedding")
    plt.savefig(os.path.join(backbone_dir,"tsne_features.png"))
    plt.close()

    # -------------------------------
    # 8. Misclassified Samples
    # -------------------------------
    misclassified = [(dataset[i][0], dataset[i][1], all_preds[i]) for i in range(len(all_preds)) if all_preds[i] != dataset[i][1]]
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    for i, (img,label,pred) in enumerate(misclassified[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(img.permute(1,2,0))
        plt.title(f"T:{class_names[label]}\nP:{class_names[pred]}")
        plt.axis("off")
    plt.savefig(os.path.join(backbone_dir,"misclassified_samples.png"))
    plt.close()

    # -------------------------------
    # 9. Dataset Distribution
    # -------------------------------
    plt.figure(figsize=(6,4))
    class_counts = [sum([1 for i in range(len(dataset)) if dataset[i][1]==c]) for c in range(len(class_names))]
    sns.barplot(x=class_names, y=class_counts)
    plt.ylabel("Number of Images")
    plt.title("Dataset Distribution")
    plt.savefig(os.path.join(backbone_dir,"dataset_distribution.png"))
    plt.close()
    
    
        # -------------------------------
    # 10. Grad-CAM++ (Correct)
    # -------------------------------
    save_gradcampp_samples(
        model=model,
        dataset=dataset,
        all_preds=all_preds,
        all_labels=all_labels,
        class_names=class_names,
        device=device,
        save_dir=os.path.join(backbone_dir, "gradcampp_correct"),
        correct=True
    )

    # -------------------------------
    # 11. Grad-CAM++ (Misclassified)
    # -------------------------------
    save_gradcampp_samples(
        model=model,
        dataset=dataset,
        all_preds=all_preds,
        all_labels=all_labels,
        class_names=class_names,
        device=device,
        save_dir=os.path.join(backbone_dir, "gradcampp_wrong"),
        correct=False
    )




# ============================
# 4. MAIN PIPELINE
# ============================

if __name__ == "__main__":
    data_dir = "dataset"
    output_dir = "figures"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbones = ["efficientnet_b0","densenet121","resnet50"]

    # Load data
    train_loader, val_loader, val_dataset, class_names, val_transform = load_dataset(data_dir)
    
    comparison_metrics = []

    for backbone in backbones:
        print(f"\n=== Training backbone: {backbone} ===")
        model = LitModel(backbone, num_classes=len(class_names))
        
        logger = CSVLogger(save_dir="logs", name=backbone)
        checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1,
                                        filename=f"best_model_{backbone}")
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        trainer = pl.Trainer(max_epochs=10, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             callbacks=[checkpoint_cb, lr_monitor],
                             logger=logger)
        print("gpu" if torch.cuda.is_available() else "cpu")
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        # Load best model
        best_model = LitModel.load_from_checkpoint(checkpoint_cb.best_model_path,map_location=device)
        best_model.to(device)
        
        # Evaluate
        all_preds, all_labels, all_outputs, df_metrics = evaluate_model(best_model, val_loader, class_names, device)
        df_metrics["Backbone"] = backbone
        comparison_metrics.append(df_metrics)
        
        # Generate figures for this backbone
        backbone_dir = os.path.join(output_dir, backbone)
        save_figures(backbone_dir, best_model, val_loader, val_dataset, all_preds, all_labels, all_outputs, class_names, device, val_transform, logger.log_dir,backbone)
    
    # Merge all metrics
    all_metrics = pd.concat(comparison_metrics, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    all_metrics.to_csv(os.path.join(output_dir,"comparison_metrics.csv"), index=False)
    
    # F1 score comparison bar chart
    plt.figure(figsize=(8,5))
    sns.barplot(data=all_metrics, x="Class", y="F1 Score", hue="Backbone")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Comparison Across Backbones")
    plt.savefig(os.path.join(output_dir,"f1_comparison.png"))
    plt.close()
    
    print(f"All metrics and figures saved in '{output_dir}' folder.")
