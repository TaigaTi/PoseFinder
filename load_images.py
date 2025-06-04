import cv2
import os

folder = "data/sitting"

images = []

# Load sitting images
for filename in os.listdir(folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load {filename}")
            
print(f"Loaded {len(images)} images from {folder}")

for i, img in enumerate(images):
    cv2.imshow(f"Image {i+1}", img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()