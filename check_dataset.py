import os

train_path = "dataset/asl_alphabet_train/asl_alphabet_train"


print("Checking dataset path...\n")

if not os.path.exists(train_path):
    print("❌ Path not found:", train_path)
    exit()

folders = sorted(os.listdir(train_path))
print("Folders found:", folders)
print("\nTotal classes:", len(folders))

for f in folders:
    img_count = len(os.listdir(os.path.join(train_path, f)))
    print(f"{f}: {img_count} images")
