from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import vgg16, VGG16_Weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 加载原始 ImageNet VGG16，不替换分类层
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)
    model = model.to(device)
    model.eval()

  
    cat_indices = list(range(281, 286))#猫的类型
    dog_indices = list(range(151, 269))#狗的类型
    cat_dog_indices = cat_indices + dog_indices

    categories = weights.meta["categories"]
    print("cat indices:", cat_indices)
    print("cat classes:", [categories[i] for i in cat_indices])
    print("dog class count:", len(dog_indices))

    # 测试集位置，不依赖当前工作目录
    base_dir = Path(__file__).resolve().parent
    test_dir = base_dir / "data" / "catVSdog" / "test_data"

    if not test_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

    # 使用官方预训练权重对应的预处理
    test_transform = weights.transforms()

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform,
    )

    print("classes:", test_dataset.classes)
    print("class_to_idx:", test_dataset.class_to_idx)
    print("test images:", len(test_dataset))

    # ImageFolder 会按照文件夹名称排序，应该是 cat=0, dog=1
    assert test_dataset.class_to_idx == {"cat": 0, "dog": 1}

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    total = 0
    correct = 0

    cat_total = 0
    cat_correct = 0
    dog_total = 0
    dog_correct = 0

    top1_in_cat_dog = 0
    group_total = 0
    group_correct = 0

    cat_dog_indices_tensor = torch.tensor(
        cat_dog_indices,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            # 合并所有猫类概率和所有狗类概率
            cat_score = probs[:, cat_indices].sum(dim=1)
            dog_score = probs[:, dog_indices].sum(dim=1)

            # cat=0, dog=1，与 ImageFolder 标签一致
            predictions = (dog_score > cat_score).long()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            cat_mask = labels == 0
            dog_mask = labels == 1

            cat_total += cat_mask.sum().item()
            dog_total += dog_mask.sum().item()

            cat_correct += (
                (predictions[cat_mask] == labels[cat_mask]).sum().item()
            )
            dog_correct += (
                (predictions[dog_mask] == labels[dog_mask]).sum().item()
            )

            top1 = logits.argmax(dim=1)
            in_group = torch.isin(top1, cat_dog_indices_tensor)

            top1_in_cat_dog += in_group.sum().item()

            if in_group.any():
                group_total += in_group.sum().item()
                group_correct += (
                    (predictions[in_group] == labels[in_group]).sum().item()
                )

    accuracy = 100.0 * correct / total
    cat_accuracy = 100.0 * cat_correct / cat_total
    dog_accuracy = 100.0 * dog_correct / dog_total
    group_ratio = 100.0 * top1_in_cat_dog / total

    if group_total > 0:
        group_accuracy = 100.0 * group_correct / group_total
    else:
        group_accuracy = 0.0

    print("\n========== Evaluation Result ==========")
    print(f"Overall cat/dog accuracy: {accuracy:.4f}%")
    print(f"Cat accuracy: {cat_accuracy:.4f}%")
    print(f"Dog accuracy: {dog_accuracy:.4f}%")
    print(f"Top-1 in cat/dog classes: {group_ratio:.4f}%")
    print(f"Accuracy when top-1 in cat/dog classes: {group_accuracy:.4f}%")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print("=======================================")


if __name__ == "__main__":
    main()