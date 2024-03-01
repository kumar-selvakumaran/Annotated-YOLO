import os
import torch
import numpy as np
import random
import shutil
from IPython.display import Image as im
from IPython.display import display as dis
import cv2

def getFullPaths(fileDir, fileNames):
  fullPaths = []
  for fp in fileNames:
    fullPaths.append(fileDir + fp)
  return fullPaths


def printGridDets(x,y, numGrids):
  gridLenW, gridLenH = 480, 300
  gridRow, gridCol = y//gridLenH, x//gridLenW
  print(f"grid row : {gridRow}, grid col : {gridCol}, x% : {x/(gridCol*numGrids)}, y% : {y/(gridRow*numGrids)}")


def getDataPaths():
  imDir = "./sampleset/images/"
  labelDir = "./sampleset/labels/"
  imPaths = os.listdir(imDir)
  imPaths = getFullPaths(imDir, imPaths)

  labelPaths = os.listdir(labelDir)
  labelPaths = getFullPaths(labelDir, labelPaths)

  return imPaths, labelPaths

def get_coords(labels, image):
    imh, imw = image.shape[:2]
    coords = []
    boxes_per_im = []
    cls_ids = []
    i = 0

    for i in range(len(labels)):

        lpath = labels[i]

        num_boxes = 0

        imCoords = []
        imClsIds = []

        with open(lpath, 'r') as rf:
            for line in rf.readlines():
                cls_id, x, y, w, h = list(map(float, line.split(" ")))
                x *= imw
                y *= imh
                w *= imw
                h *= imh
                imCoords.append([x, y, w, h])
                imClsIds.append(int(cls_id))
                num_boxes += 1

            boxes_per_im.append(num_boxes)

        coords.append(np.array(imCoords))
        cls_ids.append(np.array(imClsIds))

    # coords = np.array(coords)
    # cls_ids = np.array(cls_ids)

    return coords, cls_ids, boxes_per_im


def get_colors(num_classes = 11, normed = False, min_brightness = 0.25):
    colorvals = []
    for i in range(3):
        if normed:
            colorvals+=list(((np.array(random.sample(range(num_classes), num_classes))*((256*(1-min_brightness))/num_classes)).astype(int) + (min_brightness*256))/256) #num_classes levels of brightness, 20 classes
        else:
            colorvals+=list((np.array(random.sample(range(num_classes), num_classes))*((256*(1-min_brightness))/num_classes)).astype(int) + (min_brightness*256))
    if not normed:
        colors = [(int(colorvals[x]), int(colorvals[x+1]), int(colorvals[x+2])) for x in range(0,len(colorvals), 3)]
    else:
        colors = [(colorvals[x], colorvals[x+1], colorvals[x+2]) for x in range(0,len(colorvals), 3)]
    return colors

def dot_formatter(cls_ids, packing = 4):
    unique_ids, counts = np.unique(cls_ids, return_counts = True)
    a = counts ** (1/packing)
    a = np.round(a.max() - a + a.min())
    a/=50
    colors = get_colors(len(unique_ids), normed = True)
    dot_size_dict = dict(zip(unique_ids, list(zip(a, colors[:len(a)]))))
    return dot_size_dict

def rgb_to_hex(x):
    clr = []
    for ch in x:
        clr.append(int(round(ch*256)))
    r, g, b = tuple(clr)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)



def xyxy_to_xywh(coords, scale = True):
#     coords_scaled = np.zeros_like(coords)

    coords[:, 2] = coords[:, 2] - coords[:, 0]
    coords[:, 3] = coords[:, 3] - coords[:, 1]
    coords[:, 0] += (coords[:, 2] / 2)
    coords[:, 1] += (coords[:, 3] / 2)

    if scale:
        coords = np.round(coords).astype(int)

    return coords


def xywh_to_xyxy(coords, scale = True):
    coords_scaled = np.zeros_like(coords)

    coords_scaled[:, 0] = coords[:, 0] - (coords[:, 2] * 0.5)
    coords_scaled[:, 1] = coords[:, 1] - (coords[:, 3] * 0.5)
    coords_scaled[:, 2] = coords[:, 0] + (coords[:, 2] * 0.5)
    coords_scaled[:, 3] = coords[:, 1] + (coords[:, 3] * 0.5)

    if scale:
        coords_scaled = np.round(coords_scaled).astype(int)

    return coords_scaled

imPaths, labelPaths = getDataPaths()

def vizImBoxes(iPath, lPath, width, height):
  colors = get_colors(num_classes = 11, min_brightness = 0.1)
  i = 0
  img = cv2.imread(iPath)
  imh, imw = img.shape[:2]

  coords = []
  cls_ids = []

  with open(lPath, 'r') as rf:

      for line in rf.readlines():
          cls_id, x, y, w, h = list(map(float, line.split(" ")))
          x *= imw
          y *= imh
          w *= imw
          h *= imh
          coords.append([x, y, w, h])
          cls_ids.append(int(cls_id))

      coords = np.array(coords)
      cls_ids = np.array(cls_ids)

      if len(coords) <1 :
          print("NO LABELS FOR THIS IMAGE\n")
          return
      coords = xywh_to_xyxy(coords)

  for ind, coord in enumerate(coords):
      color = colors[cls_ids[ind]]
      img = cv2.rectangle(img, (coord[0], coord[1]), (coord[2], coord[3]), color = color, thickness = 6)

  img = cv2.resize(img, (width, height))
  cv2.imwrite("temp.jpg", img)
  dis(im("temp.jpg"))


def getPathComponents(x):
  fileDir = "/".join(x.split("/")[:-1])
  fileName = ".".join(x.split("/")[-1].split('.')[:-1])
  extension = "." + x.split("/")[-1].split('.')[-1]
  return fileDir, fileName, extension

def getfileId(x):
  fileId = x.split("/")[-1].split("_")[-1].split(".")[0]
  return int(fileId)

def orderFiles(imPaths, labelPaths):
  newImPaths = []
  newlabelPaths = []

  nameDict = dict()

  for ind, lp in enumerate(labelPaths):
    lDir, lName, lExtension = getPathComponents(lp)

    newLabelPath = lDir + "/" + lName + "_" + str(ind) + lExtension

    shutil.move(lp, newLabelPath)

    newlabelPaths.append(newLabelPath)
    nameDict[lName] = ind

  for ind, ip in enumerate(imPaths):
    iDir, iName, iExtension = getPathComponents(ip)

    newImPath = iDir + "/" + iName + "_" + str(nameDict[iName]) + iExtension

    newImPaths.append(newImPath)

    shutil.move(ip, newImPath)

  newImPaths.sort(key = getfileId)

  return newImPaths, newlabelPaths

#______________________________________________________________________________
##################3_______  MAKE GROUNDTRUTHS ______________###################
##########  IMPLEMENTED SINGLE MEMBERSSHIP, i.e.  THE BOX BELONGS TO THE GRID 
# THAT CONTAINS THE THE BOX'S CENTER.  #########################################
#______________________________________________________________________________

def getGridMembership(allCoords, numGrids, imW, imH):
  gridLenW = imW // numGrids #Finding the number of grid cols
  gridLenH = imH // numGrids  #finding the number of grid rows

  gridMems = []

  for imCoord in allCoords:
    print(f"imCoord : {imCoord}")
    if len(imCoord)<1:  
      gridMems.append(torch.tensor([])) #when image has no boxes
      continue
    centerX = imCoord[:,0]  
    centerY = imCoord[:,1]
    gridX = centerX // gridLenW #finding the col grid where the center lies 
    gridY = centerY // gridLenH #finding the row grid where the center lies

    # appending grid positions as a num_boxes x 2 tensor. (actually grid pos 
    # array for image image is hetergenous, so a list "gridMems" of len 
    # Batch_size is used to store the heterogenous list of grid positions 
    # corresponding to the boxes in images )) 
    gridMems.append(torch.cat((gridX[..., None], gridY[..., None]), 1))

  return gridMems

"""
From the grid positions, this obtains the one hot vector of size #grids

eg: if a box is in grid (3,2) when the image is split as 4x4 grid, 
    the one hot vector is [[0, 0, 0, 1][0, 0, 1, 0]]
"""

def getGridMembershipMask(gridMems, numGrids):
  if gridMems == None:
    print("\n GRID MEMBERSHIPS IS NONE, IMAGE SHOULD NOT BE HAVING BBs\n")
    return

  if isinstance(gridMems, list):
    templist = []
    for gridMem in gridMems:
      if len(gridMem)<1:
        templist.append(torch.tensor([]))
        continue
      templist.append(nn.functional.one_hot(gridMem.to(int), numGrids))
    gridMems = templist
  else:
    print(f"""\n expected grid membership as a list of tensors or np.arrays of
              of bbox memberships per image in the following format:\n
              [[bbox1im1mem -> bboxnim1mem], [bbox1im2mem -> bboxnim2mem] ... imn]""")
    return

  return gridMems

"""
getting boolean mask from the onehot vectors , depending on the grid query.

eg: if the 3rd box/GT in a image is in grid (3,2) denoted by the output of
    getGridMemberships() as [[0, 0, 0, 1][0, 0, 1, 0,]]
                            AND
    you want boxes boxes in grid (1,1), then the output of this function will be
    [- , - , False,  ... -] (size = numBBoxes), if it lies in the box you wanted, 
    it would have been True instead.
"""
def getGridBoxes(gridMemsOneHot, gridX, gridY):

  if gridMemsOneHot == None:
    print("\n GRID MEMBERSHIPS IS NONE, IMAGE SHOULD NOT BE HAVING BBs\n")
    return

  gridMasks = []
  for imGridMem in gridMemsOneHot:
    if len(imGridMem) < 1:
      gridMasks.append(torch.tensor([]))
      continue
    gridMask = torch.logical_and(imGridMem[:, 0, gridX] == 1, imGridMem[:, 1, gridY] == 1)
    # print(f"from getgridBoxes : {(gridMems[0][:, 0, gridX]).shape}")
    gridMasks.append(gridMask)

  return gridMasks


def vizImBoxesRaw(img, coords, cls_ids, width, height):

  colors = get_colors(num_classes = 11, min_brightness = 0.1)

  coords = xywh_to_xyxy(coords)

  for ind, coord in enumerate(coords):
    color = colors[cls_ids[ind]]
    img = cv2.rectangle(img, (coord[0], coord[1]), (coord[2], coord[3]), color = color, thickness = 6)

  img = cv2.resize(img, (width, height))
  cv2.imwrite("temp.jpg", img)
  dis(im("temp.jpg"))
