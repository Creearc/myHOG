from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import imutils
import cv2
import pickle
import numpy as np
from scipy.spatial import distance

import time

def pyramid(img, scale=1.2, window_size=30):
  out = img.copy()
  multi = 1
  yield out, multi 
  while True:
    h, w = out.shape[:2]
    w = int(w / scale)
    multi = multi * scale
    out = imutils.resize(out, width=w)
    if h < window_size or w < window_size:
      break
    yield out, multi

def sliding_window(img, window_size, step):
  h, w = img.shape[:2]
  for x in range(0, w - window_size, step):
    for y in range(0, h - window_size, step):
      yield (x, y, img[y : y + window_size, x : x + window_size])

def draw_obj(img, objs, save=False):
  out = img.copy()
  for (x, y, window_size) in objs:
    if save:
      cv2.imwrite('out/{}.bmp'.format(time.time()), img[y : y + window_size, x : x + window_size])
      time.sleep(0.01)
    out = cv2.rectangle(out, (x, y), (x + window_size, y + window_size), 255, 3)
  return out

def neighbors(objs, min_dist=10):
  out = []
  for (x, y, _)  in objs:
    i=0
    while i:
      x1, y1 = objs[i][0], objs[i][1]
      if (x==x1 and y==y1) or window_size == -1:
        continue
      if distance.euclidean((x1, y1), (x, y)) < min_dist:
        objs.pop(i)
      else:
        out.append(objs[i])
  return out
  

debug = True
debug = False

t = time.time()
print('Loading HOG model')
with open('hand_up.dat', 'rb') as f:
  model = pickle.load(f)
print('Model was loaded. {}sec'.format(time.time() - t))

window_size = 100
step = 25


for imagePath in paths.list_images("test_h/"):
  t = time.time()
  objs = []
  img = cv2.imread(imagePath)
  h0, w0 = img.shape[:2]
  if h0 > w0:
    img = imutils.resize(img, height=480, inter=cv2.INTER_NEAREST)
  else:
    img = imutils.resize(img, width=640, inter=cv2.INTER_NEAREST)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  for (resized_img, multi) in pyramid(img, scale=1.2, window_size=window_size):
    window_sizes =int(window_size * multi)
    for (x, y, inp) in sliding_window(resized_img, window_size, step):
      inp = imutils.resize(inp, width=40, height=40, inter=cv2.INTER_NEAREST)
      H = feature.hog(inp,
                      orientations=9, pixels_per_cell=(20, 20),
                      cells_per_block=(2, 2),
                      transform_sqrt=True, block_norm="L1")
      if model.predict(H.reshape(1, -1))[0] == '1':
        xs, ys = int(x * multi), int(y * multi)
        objs.append((xs, ys, window_sizes))
      if debug:
        out = img.copy()
        out = cv2.rectangle(out, (x, y), (x + multi, y + multi), 150, 2)
        cv2.imshow("out", out)
        cv2.waitKey(1)
      
  print(time.time() - t)
  #objs = neighbors(objs, min_dist=10)
  cv2.imshow("img", draw_obj(img, objs, True))
  cv2.waitKey(1)
    


