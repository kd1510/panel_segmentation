from datasets import Dataset, Image, DatasetDict
import os


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

def generate_train_val_ds():
    folder = os.environ['IMAGE_FOLDER_PATH']
    data = [os.path.join(folder, x) for x in os.listdir(folder)]

    images = sorted([x for x in data if 'label' not in x])
    labels = sorted([x for x in data if 'label' in x])

    split = int(len(images) * 0.8)
    image_paths_train = images[:split]
    image_paths_val = images[split:]
    label_paths_train = labels[:split]
    label_paths_val = labels[split:]

    train_dataset = create_dataset(image_paths_train[:50], label_paths_train[:50])
    val_dataset = create_dataset(image_paths_val[:10], label_paths_val[:10])

    return train_dataset, val_dataset