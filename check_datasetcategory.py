import os

dataset_path = "dataset"

classes = os.listdir(dataset_path)

print("Classes Found:")
print()

for cls in classes:
    print(cls)

print("\nTotal Classes:", len(classes))