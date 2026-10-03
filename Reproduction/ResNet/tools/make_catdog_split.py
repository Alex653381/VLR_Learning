from pathlib import Path
import random


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE_DIR / "data" / "catVSdog"

SEED = 42
VAL_RATIO = 0.1


def normalize_path(raw_path):
    path = Path(raw_path.replace("\\", "/"))

    try:
        path = path.relative_to(Path("data/catVSdog"))
    except ValueError:
        pass

    return path.as_posix()


def read_samples(txt_path):
    samples = []

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            raw_path, label = line.split()
            samples.append((normalize_path(raw_path), int(label)))

    return samples


def stratified_split(samples, val_ratio, seed):
    rng = random.Random(seed)
    samples_by_class = {}

    for sample in samples:
        label = sample[1]
        samples_by_class.setdefault(label, []).append(sample)

    train_samples = []
    val_samples = []

    for label in sorted(samples_by_class):
        class_samples = samples_by_class[label]
        rng.shuffle(class_samples)

        val_size = max(1, int(len(class_samples) * val_ratio))

        val_samples.extend(class_samples[:val_size])
        train_samples.extend(class_samples[val_size:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


def write_annotations(samples, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        for image_path, label in samples:
            f.write(f"{image_path} {label}\n")


def count_labels(samples):
    labels = [label for _, label in samples]
    return labels.count(0), labels.count(1)


def main():
    all_train_samples = read_samples(DATA_ROOT / "train.txt")
    test_samples = read_samples(DATA_ROOT / "test.txt")

    train_samples, val_samples = stratified_split(
        all_train_samples,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    write_annotations(train_samples, DATA_ROOT / "train_split.txt")
    write_annotations(val_samples, DATA_ROOT / "val_split.txt")
    write_annotations(test_samples, DATA_ROOT / "test_split.txt")

    train_cat, train_dog = count_labels(train_samples)
    val_cat, val_dog = count_labels(val_samples)
    test_cat, test_dog = count_labels(test_samples)

    print("train:", len(train_samples), "cat:", train_cat, "dog:", train_dog)
    print("val:", len(val_samples), "cat:", val_cat, "dog:", val_dog)
    print("test:", len(test_samples), "cat:", test_cat, "dog:", test_dog)


if __name__ == "__main__":
    main()