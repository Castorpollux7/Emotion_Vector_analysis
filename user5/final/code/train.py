"""
Step 3: 이원화 모델 학습 (A40 서버용)
=======================================
Model A: ResNet50 + DAN (이목구비 미세 변화 포착)
Model B: ViT-Base      (이미지 전체 맥락 파악)

설치:
    pip install torch torchvision timm tqdm scikit-learn matplotlib

실행:
    # Model A 학습
    python train.py --model resnet_dan

    # Model B 학습
    python train.py --model vit

    # 둘 다 순서대로 학습
    python train.py --model both
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ================= 설정 =================
DATA_DIR    = "/workspace/user5/img_augmented"   # Augmentation 결과 폴더
OUTPUT_DIR  = "/workspace/user5/model_output"    # 모델 저장 폴더
CLASSES     = ["anger", "blank", "happy", "panic", "sadness"]
NUM_CLASSES = len(CLASSES)

# 학습 하이퍼파라미터 (A40 48GB VRAM 기준)
BATCH_SIZE  = 128     # VRAM 넉넉해서 크게 설정
NUM_EPOCHS  = 50
LR          = 1e-4
WEIGHT_DECAY= 1e-4
NUM_WORKERS = 8
IMG_SIZE    = 224

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


# ── Transform 정의 ─────────────────────────────────────────────
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
# Model A: ResNet50 + DAN (Distract Your Attention Network)
# 이목구비의 미세한 변화를 Attention으로 포착
# ══════════════════════════════════════════════════════════════

class AttentionModule(nn.Module):
    """채널 + 공간 Attention 결합"""
    def __init__(self, channels):
        super().__init__()
        # 채널 Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )
        # 공간 Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 채널 Attention
        ca  = self.channel_att(x).view(x.size(0), x.size(1), 1, 1)
        x   = x * ca

        # 공간 Attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x,  dim=1, keepdim=True)[0]
        sa  = self.spatial_att(torch.cat([avg, mx], dim=1))
        x   = x * sa

        return x


class ResNetDAN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ResNet50 백본 로드
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 백본 레이어 분리
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                     backbone.relu, backbone.maxpool)
        self.layer1  = backbone.layer1   # 256ch
        self.layer2  = backbone.layer2   # 512ch
        self.layer3  = backbone.layer3   # 1024ch
        self.layer4  = backbone.layer4   # 2048ch

        # DAN Attention 모듈 (layer3, layer4에 적용)
        self.att3 = AttentionModule(1024)
        self.att4 = AttentionModule(2048)

        # 분류 헤드
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Linear(2048, num_classes)

        # BatchNorm 정규화
        self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.att3(x)       # Attention 적용
        x = self.layer4(x)
        x = self.att4(x)       # Attention 적용
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ══════════════════════════════════════════════════════════════
# Model B: ViT-Base (Vision Transformer)
# 이미지 전체의 맥락과 분위기 파악
# ══════════════════════════════════════════════════════════════

def build_vit(num_classes):
    try:
        import timm
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.1,
            attn_drop_rate=0.1
        )
        return model
    except ImportError:
        print("❌ timm 설치 필요: pip install timm")
        return None


# ══════════════════════════════════════════════════════════════
# 학습 함수
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient Clipping (안정적 학습)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += imgs.size(0)

    if scheduler:
        scheduler.step()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs      = model(imgs)
        loss         = criterion(outputs, labels)

        total_loss  += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct     += predicted.eq(labels).sum().item()
        total       += imgs.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ══════════════════════════════════════════════════════════════
# 학습 메인
# ══════════════════════════════════════════════════════════════

def train_model(model_name):
    print(f"\n{'='*55}")
    print(f"  Model {'A: ResNet50 + DAN' if model_name == 'resnet_dan' else 'B: ViT-Base'} 학습 시작")
    print(f"  디바이스: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"{'='*55}\n")

    # 데이터 로드
    print("데이터 로드 중...")
    train_dataset = EmotionDataset(DATA_DIR, "train", get_transforms(True))
    val_dataset   = EmotionDataset(DATA_DIR, "val",   get_transforms(False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 모델 생성
    if model_name == "resnet_dan":
        model = ResNetDAN(NUM_CLASSES).to(DEVICE)
    else:
        model = build_vit(NUM_CLASSES)
        if model is None:
            return
        model = model.to(DEVICE)

    # 클래스 불균형 처리 (Loss 가중치)
    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    class_weights = torch.tensor(
        [max(class_counts) / c for c in class_counts],
        dtype=torch.float
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 학습 루프
    save_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch:>3}/{NUM_EPOCHS}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc, preds, labels = evaluate(
            model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")

        # Best 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "classes":    CLASSES,
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"  ✅ Best 모델 저장 (val_acc: {best_acc*100:.2f}%)")

        # 10 에폭마다 중간 저장
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model_name": model_name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth"))

    # 최종 리포트
    print(f"\n{'='*55}")
    print(f"  학습 완료 — Best Val Acc: {best_acc*100:.2f}%")
    print(f"{'='*55}\n")

    # Classification Report
    print(classification_report(labels, preds, target_names=CLASSES))

    # 학습 곡선 저장
    _save_training_curve(history, model_name, save_dir)

    # 히스토리 저장
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"📁 모델 저장 위치: {save_dir}")
    return best_acc


def _save_training_curve(history, model_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150)
    plt.close()
    print(f"  📈 학습 곡선 저장: {save_dir}/training_curve.png")


# ══════════════════════════════════════════════════════════════
# CLI 실행
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both",
                        choices=["resnet_dan", "vit", "both"],
                        help="학습할 모델 선택")
    args = parser.parse_args()

    print(f"디바이스: {DEVICE}")
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 폴더 없음: {DATA_DIR}")
        print("   augmentation.py 먼저 실행하세요.")
        exit(1)

    results = {}

    if args.model in ("resnet_dan", "both"):
        acc = train_model("resnet_dan")
        results["resnet_dan"] = acc

    if args.model in ("vit", "both"):
        acc = train_model("vit")
        results["vit"] = acc

    if len(results) > 1:
        print(f"\n{'='*40}")
        print("  최종 모델 비교")
        print(f"{'='*40}")
        for name, acc in results.items():
            print(f"  {name:<15}: {acc*100:.2f}%")
        best = max(results, key=results.get)
        print(f"\n  🏆 Best 모델: {best} ({results[best]*100:.2f}%)")
