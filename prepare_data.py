from datasets import Dataset, Image, DatasetDict
import os


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

# load dataset
# folder = '/home/krish/Downloads/zenodo/extracted/full/PV03/PV03_Ground_Cropland'
folder = os.environ['IMAGE_FOLDER_PATH']
data = [os.path.join(folder, x) for x in os.listdir(folder)]

images = sorted([x for x in data if 'label' not in x])
labels = sorted([x for x in data if 'label' in x])

# split into train and validation
split = int(len(images) * 0.8)
image_paths_train = images[:split]
image_paths_val = images[split:]
label_paths_train = labels[:split]
label_paths_val = labels[split:]


train_dataset = create_dataset(image_paths_train, label_paths_train)
val_dataset = create_dataset(image_paths_val, label_paths_val)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
  }
)