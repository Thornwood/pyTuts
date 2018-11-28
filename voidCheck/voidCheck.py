# Imports
import os

imgs = []
path = input("Give the path where the x-ray images are:")
extensions = [".jpg", ".png", ".tga"]

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(tuple(extensions)):
            imgs.append(os.path.join(root, file))

print("Images read: " + str(imgs.__len__()))
print(*imgs, sep="\n")
