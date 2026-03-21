import kagglehub
import os

path = kagglehub.dataset_download("dharshan0025/skin-cancer-dataset")
print("Path to dataset files:", path)

for f in os.listdir(path):
    print(f)
