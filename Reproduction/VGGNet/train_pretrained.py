import torch
import torch.nn as nn
from torchvision.models import vgg16,VGG16_Weights

from pathlib import Path

import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#配置路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "catVSdog"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TXT = DATA_DIR / "train.txt"
TEST_TXT = DATA_DIR / "test.txt"
VAL_RATIO = 0.1
NUM_WORKERS = 4
SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BEST_MODEL_PATH = MODEL_DIR / "vgg16_catvsdog_head_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2 
BATCH_SIZE = 32 
NUM_EPOCHS = 5  
LEARNING_RATE = 0.001 

def build_model():
    #加载预训练的VGG16模型
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights = weights)

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    #冻结前面提取特征的部分，这些部分反向传播训练时不用求导、更新
    for param in model.features.parameters():
        param.requires_grad = False
    
    #模型和参数移动到GPU上（如果有）
    model = model.to(DEVICE)
    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class CatDogDataset(Dataset):
    def __init__(self,samples,transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)#样本数量

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label#返回一个样本索引

def read_samples(txt_file):
    samples = []
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            relative_path, label = line.split()
            image_path = BASE_DIR / relative_path

            if not image_path.exists():
                raise FileNotFoundError(f"图片不存在: {image_path}")

            samples.append((image_path, int(label)))  # 将标签转换为整数

    return samples

#划分训练集和验证集
def stratified_split(samples,val_ratio,seed):
    rng = random.Random(seed)

    samples_by_class = {}

    for sample in samples:
        label = sample[1]
        if label not in samples_by_class:
            samples_by_class[label] = []
        samples_by_class[label].append(sample)

    train_samples = []
    val_samples = []

    for label in sorted(samples_by_class.keys()):
        class_samples = samples_by_class[label]
        rng.shuffle(class_samples)

        val_size = int(len(class_samples) * val_ratio)
        val_size = max(val_size, 1)  # 确保至少有一个样本用于验证集

        val_samples.extend(class_samples[:val_size])
        train_samples.extend(class_samples[val_size:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples

#构建训练集和验证集的两套预处理
def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),#随机裁剪
        transforms.RandomHorizontalFlip(),#随机水平翻转图像
        transforms.ColorJitter(#随机扰动
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),#归一化
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),#缩放
        transforms.CenterCrop(224),#裁剪
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform

def build_dataloaders():
    all_train_samples = read_samples(TRAIN_TXT)
    train_samples, val_samples = stratified_split(
        all_train_samples,
        VAL_RATIO,
        SEED
    )

    train_transform, val_transform = build_transforms()

    train_dataset = CatDogDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = CatDogDataset(
        val_samples,
        transform=val_transform
    )

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    return train_loader, val_loader

#创建AdamW优化器
def build_optimizer(model):
    trainable_params = [
        param
        for param in model.parameters()
        if param.requires_grad#参数更新刚才没有冻结的部分
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    return optimizer

#完成一个epoch的训练
def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    epoch
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            current_loss = running_loss / total
            current_acc = correct / total

            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {current_loss:.4f} "
                f"Accuracy: {current_acc:.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc

def main():
    print("device:", DEVICE)

    train_loader, val_loader = build_dataloaders()

    train_labels = [
        label
        for _, label in train_loader.dataset.samples
    ]
    val_labels = [
        label
        for _, label in val_loader.dataset.samples
    ]

    print("train images:", len(train_loader.dataset))
    print("val images:", len(val_loader.dataset))
    print("train cat:", train_labels.count(0))
    print("train dog:", train_labels.count(1))
    print("val cat:", val_labels.count(0))
    print("val dog:", val_labels.count(1))

    model = build_model()
    total_params, trainable_params = count_parameters(model)

    print("total parameters:", total_params)
    print("trainable parameters:", trainable_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch
        )

        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion
        )

        print(
            f"Epoch {epoch} finished | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }

            torch.save(checkpoint, BEST_MODEL_PATH)

            print(
                f"Best model saved: {BEST_MODEL_PATH} | "
                f"val acc: {best_val_acc:.4f}"
            )

    print("\nTraining finished")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()






