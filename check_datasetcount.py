import os

dataset_path = "dataset"

for cls in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, cls)

    image_count = len(os.listdir(class_path))

    print(f"{cls}: {image_count} images")