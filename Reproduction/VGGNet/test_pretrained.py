from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16


BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "data" / "catVSdog" / "test_data"
MODEL_PATH = BASE_DIR / "models" / "vgg16_catvsdog_head_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_CLASSES = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model():
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        NUM_CLASSES
    )
    return model


def main():
    print("device:", DEVICE)
    print("checkpoint:", MODEL_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint val loss:", checkpoint.get("val_loss"))
    print("checkpoint val acc:", checkpoint.get("val_acc"))

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    print("classes:", test_dataset.classes)
    print("class_to_idx:", test_dataset.class_to_idx)
    print("test images:", len(test_dataset))

    assert test_dataset.class_to_idx == {"cat": 0, "dog": 1}

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda"
    )

    correct = 0
    total = 0
    cat_correct = 0
    cat_total = 0
    dog_correct = 0
    dog_total = 0

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            cat_mask = labels == 0
            dog_mask = labels == 1

            cat_total += cat_mask.sum().item()
            dog_total += dog_mask.sum().item()

            cat_correct += (
                predictions[cat_mask] == labels[cat_mask]
            ).sum().item()

            dog_correct += (
                predictions[dog_mask] == labels[dog_mask]
            ).sum().item()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    accuracy = 100.0 * correct / total
    cat_accuracy = 100.0 * cat_correct / cat_total
    dog_accuracy = 100.0 * dog_correct / dog_total

    print("\n========== Final Test Result ==========")
    print(f"Overall accuracy: {accuracy:.4f}%")
    print(f"Cat accuracy: {cat_accuracy:.4f}%")
    print(f"Dog accuracy: {dog_accuracy:.4f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average time per image: {elapsed / total * 1000:.2f} ms")
    print("=======================================")


if __name__ == "__main__":
    main()