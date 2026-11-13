"""
Step 3: 이원화 모델 학습 v2 (A40 서버용)
==========================================
Model A: EfficientNet-B7 + CBAM  (이목구비 미세 변화 + Attention)
Model B: Swin Transformer Base   (전체 맥락 + 고해상도 강건성)

설치:
    pip install torch torchvision timm tqdm scikit-learn matplotlib seaborn

실행:
    python train_v2.py --model efficientnet   # Model A만
    python train_v2.py --model swin           # Model B만
    python train_v2.py --model both           # 둘 다
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ================= 설정 =================
DATA_DIR    = "/workspace/user5/img_augmented"
OUTPUT_DIR  = "/workspace/user5/model_output_v2"
CLASSES     = ["anger", "blank", "happy", "panic", "sadness"]
NUM_CLASSES = len(CLASSES)

# A40 48GB VRAM 기준 하이퍼파라미터
BATCH_SIZE   = 64      # EfficientNet-B7은 메모리 많이 씀 → 64로 설정
NUM_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 8
IMG_SIZE     = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class EmotionDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(data_dir, split)
        for emotion in CLASSES:
            folder = os.path.join(split_dir, emotion)
            if not os.path.isdir(folder):
                continue
            label = CLASS_TO_IDX[emotion]
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, fname), label))

        print(f"  {split}: {len(self.samples)}장 로드")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])


# ══════════════════════════════════════════════════════════════
# CBAM (Convolutional Block Attention Module)
# 채널 + 공간 Attention을 결합해 이목구비 영역에 집중
# ══════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        mx  = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        att = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x,  dim=1, keepdim=True)[0]
        att = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Channel + Spatial Attention 결합"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ══════════════════════════════════════════════════════════════
# Model A: EfficientNet-B7 + CBAM
# 정확도/연산량 균형 + Attention으로 이목구비 집중
# ══════════════════════════════════════════════════════════════

class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import timm
        # EfficientNet-B7 백본
        backbone        = timm.create_model("efficientnet_b7",
                                             pretrained=True, num_classes=0)
        self.features   = backbone
        feat_dim        = backbone.num_features  # 2560

        # CBAM Attention
        self.cbam       = CBAM(feat_dim, reduction=32)

        # 분류 헤드
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.bn         = nn.BatchNorm1d(feat_dim)
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # 특징 추출 (마지막 pooling 전 feature map)
        x = self.features.forward_features(x)  # (B, 2560, H, W)
        x = self.cbam(x)                        # Attention 적용
        x = self.pool(x).flatten(1)             # (B, 2560)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ══════════════════════════════════════════════════════════════
# Model B: Swin Transformer Base
# 계층적 Attention으로 전체 맥락 + 고해상도 강건성
# ══════════════════════════════════════════════════════════════

def build_swin(num_classes):
    import timm
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.2,
        attn_drop_rate=0.1,
        drop_path_rate=0.2,
    )
    return model


# ══════════════════════════════════════════════════════════════
# 학습 함수
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Mixed Precision (A40 활용 극대화)
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.cuda.amp.autocast():
            outputs  = model(imgs)
            loss     = criterion(outputs, labels)

        total_loss  += loss.item() * imgs.size(0)
        preds        = outputs.argmax(1)
        correct     += preds.eq(labels).sum().item()
        total       += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ══════════════════════════════════════════════════════════════
# 결과 시각화
# ══════════════════════════════════════════════════════════════

def save_results(history, preds, labels, model_name, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} 학습 결과", fontsize=14, fontweight="bold")

    # Loss 곡선
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#4895EF")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#F94144")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 곡선
    axes[1].plot(epochs, [a*100 for a in history["train_acc"]],
                 label="Train", color="#4895EF")
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],
                 label="Val",   color="#F94144")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[2])
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "results.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 결과 그래프 저장: {save_dir}/results.png")


# ══════════════════════════════════════════════════════════════
# 학습 메인
# ══════════════════════════════════════════════════════════════

def train_model(model_name):
    model_label = {
        "efficientnet": "Model A: EfficientNet-B7 + CBAM",
        "swin":         "Model B: Swin Transformer Base"
    }[model_name]

    print(f"\n{'='*55}")
    print(f"  {model_label}")
    print(f"  디바이스: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"{'='*55}\n")

    # 데이터 로드
    print("데이터 로드 중...")
    train_ds = EmotionDataset(DATA_DIR, "train", get_transforms(True))
    val_ds   = EmotionDataset(DATA_DIR, "val",   get_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 모델 생성
    print(f"\n모델 로드 중: {model_name}")
    try:
        if model_name == "efficientnet":
            model = EfficientNetCBAM(NUM_CLASSES).to(DEVICE)
        else:
            model = build_swin(NUM_CLASSES).to(DEVICE)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("   pip install timm 확인하세요")
        return None

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  파라미터 수: {total_params:.1f}M")

    # 클래스 불균형 가중치
    counts  = [0] * NUM_CLASSES
    for _, l in train_ds.samples:
        counts[l] += 1
    weights = torch.tensor(
        [max(counts) / c for c in counts], dtype=torch.float
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    # Mixed Precision Scaler (A40 속도 향상)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # 학습 루프
    save_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    history  = {"train_loss": [], "val_loss": [],
                 "train_acc":  [], "val_acc":  []}
    final_preds, final_labels = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:>3}/{NUM_EPOCHS}  lr={lr_now:.2e}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, preds, labels = evaluate(
            model, val_loader, criterion)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        final_preds, final_labels = preds, labels

        print(f"  Train  Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%")
        print(f"  Val    Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")

        # Best 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "classes":    CLASSES,
                "class_to_idx": CLASS_TO_IDX,
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"  ✅ Best 저장 → val_acc: {best_acc*100:.2f}%")

        # 10 에폭마다 체크포인트
        if epoch % 10 == 0:
            torch.save({
                "epoch":     epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
            }, os.path.join(save_dir, f"ckpt_epoch{epoch}.pth"))

    # 최종 리포트
    print(f"\n{'='*55}")
    print(f"  학습 완료 — Best Val Acc: {best_acc*100:.2f}%")
    print(f"{'='*55}\n")
    print(classification_report(final_labels, final_preds, target_names=CLASSES))

    # 결과 시각화
    save_results(history, final_preds, final_labels, model_name, save_dir)

    # 히스토리 저장
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n📁 저장 위치: {save_dir}")
    return best_acc


# ══════════════════════════════════════════════════════════════
# 두 모델 결과 비교
# ══════════════════════════════════════════════════════════════

def compare_models():
    """두 모델의 학습 곡선을 한 그래프에 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"efficientnet": "#4895EF", "swin": "#F94144"}
    labels = {"efficientnet": "EfficientNet-B7+CBAM",
              "swin":         "Swin Transformer"}

    for model_name in ["efficientnet", "swin"]:
        hist_path = os.path.join(OUTPUT_DIR, model_name, "history.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            h = json.load(f)
        epochs = range(1, len(h["val_acc"]) + 1)
        color  = colors[model_name]
        label  = labels[model_name]

        axes[0].plot(epochs, h["val_loss"], label=label, color=color)
        axes[1].plot(epochs, [a*100 for a in h["val_acc"]],
                     label=label, color=color)

    for ax, title in zip(axes, ["Val Loss 비교", "Val Accuracy 비교 (%)"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Model A vs Model B 비교", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 모델 비교 그래프 저장: {save_path}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both",
                        choices=["efficientnet", "swin", "both"])
    args = parser.parse_args()

    print(f"디바이스: {DEVICE}")
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 없음: {DATA_DIR}")
        print("   augmentation.py 먼저 실행하세요.")
        exit(1)

    results = {}

    if args.model in ("efficientnet", "both"):
        acc = train_model("efficientnet")
        if acc:
            results["EfficientNet-B7+CBAM"] = acc

    if args.model in ("swin", "both"):
        acc = train_model("swin")
        if acc:
            results["Swin Transformer"] = acc

    if len(results) > 1:
        compare_models()
        print(f"\n{'='*45}")
        print("  최종 모델 비교")
        print(f"{'='*45}")
        for name, acc in results.items():
            print(f"  {name:<25}: {acc*100:.2f}%")
        best = max(results, key=results.get)
        print(f"\n  🏆 Best: {best} ({results[best]*100:.2f}%)")
