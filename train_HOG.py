from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import imutils
import cv2
import pickle


print("[INFO] extracting features...")
data = []
labels = []

for imagePath in paths.list_images("data3\\"):
  make = imagePath.split("\\")[-2]
  img = cv2.imread(imagePath)
  if img is None:
    print(imagePath)
    continue
  img = cv2.resize(img, (40, 40), cv2.INTER_NEAREST)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  img_list = (img, cv2.flip(img, 1))
  for t_img in img_list:
    H = feature.hog(t_img, orientations=9, pixels_per_cell=(20, 20),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

    data.append(H)
    labels.append(make)

print("[INFO] training classifier...")

model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)

with open('hand_up.dat', "wb") as f:
  pickle.dump(model, f)
