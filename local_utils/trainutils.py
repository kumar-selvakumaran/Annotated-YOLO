import torch
from torch import nn
import numpy as np

def printGridDets(x,y, numGrids):
  print(f"\n{x,y}")
  imWidth, imHeight = 1920, 1200
  gridLenW = imWidth//numGrids
  gridLenH = imHeight//numGrids
  gridRow, gridCol = y//gridLenH, x//gridLenW
  xper = round((x-(gridCol*gridLenW))/gridLenW, 2)
  yper = round((y-(gridRow*gridLenH))/gridLenH, 2)
  print(f"2d : {(gridRow, gridCol)}, 1d : {(gridRow*numGrids) + gridCol} x% : {xper}, y% : {yper}")
  if xper < 0.5:
    gridRowTemp = gridRow 
    gridColTemp = gridCol - 1
    print(f"other valid : 2d : {(gridRowTemp, gridColTemp)}, 1d : {(gridRowTemp*numGrids) + gridColTemp} ")
  
  if xper > 0.5:
    gridRowTemp = gridRow  
    gridColTemp = gridCol + 1
    print(f"other valid : 2d : {(gridRowTemp, gridColTemp)}, 1d : {(gridRowTemp*numGrids) + gridColTemp} ")

  if yper < 0.5:
    gridRowTemp = gridRow - 1
    gridColTemp = gridCol 
    print(f"other valid : 2d : {(gridRowTemp, gridColTemp)}, 1d : {(gridRowTemp*numGrids) + gridColTemp} ")

  if yper > 0.5:
    gridRowTemp = gridRow + 1
    gridColTemp = gridCol
    print(f"other valid : 2d : {(gridRowTemp, gridColTemp)}, 1d : {(gridRowTemp*numGrids) + gridColTemp}\n")

#_______________________________________________________________________________
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#                    P R E D S    F O R M A T T I N G
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""
FORMATTING POSITS (preds)

STRUCTURE:

[Pclass1, Pclass2, Pclass3.... Pclass11, Conf, x, y, w, h] //NOT YET, SHOULD DO THE REGRESSION NORMALIZATION
"""
def formatPredsForLossComp(preds, numClasses):
  positSize, numGrids, _ = preds.shape
  predSize = numClasses + 5
  predInds = list(range(0, positSize, predSize))
  preds = preds.flatten(-2,-1)
  anchPreds = torch.stack(list(torch.tensor_split(preds, predInds, dim = 0)[1:]))#.shape
  
  out = anchPreds.permute(2, 0, 1)
  return out

"""
implementing posits to boxes based on https://docs.ultralytics.com/yolov5/tutorials/architecture_description/#43-eliminate-grid-sensitivity
but making boxes in IMAGE COORDINATES UNLIKE OG IMPLEMENTATION. check why they didnt use image coordinates and get back.
"""
def positsToBBoxes(posits,
                   imWidth,
                   imHeight,
                   anchorBoxes):
  totalNumGrids, numAnchors, predSize = posits.shape
  numGrids = totalNumGrids ** (1/2)
  gridWidth = imWidth / numGrids
  gridHeight = imHeight / numGrids

  vertGridInds, horGridInds = torch.meshgrid(torch.arange(numGrids), torch.arange(numGrids))
  vertGridInds, horGridInds = vertGridInds.flatten(), horGridInds.flatten()

  #posits to bbox coordinates(x) (image (pixels))
  posits[:,:,-4] = (((2 * torch.sigmoid(posits[:,:,-4])) - 0.5) * gridWidth) #scaling
  posits[:,:,-4] += (gridWidth * horGridInds[None, ...].tile(numAnchors,1).T) #global positioning

  #posits to bbox coordinates(y) (image (pixesls))
  posits[:, :, -3] = (((2 * torch.sigmoid(posits[:, :, -3])) - 0.5) * gridHeight) #scaling
  posits[:, :, -3] += (gridHeight * vertGridInds[None, ...].tile(numAnchors,1).T) #global positioning

  #posits to bbox coordinates(wh) (image (pixesls))
  posits[:, :, -2:] = anchorBoxes[:, -2:] * ( 2 * torch.sigmoid(posits[:, :, -2:] ) **2) 
  return posits

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#_______________________________________________________________________________




#_______________________________________________________________________________
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                #   B U I L D I N G   T A R G E T S
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx                
#______________________________________________________________________________

"""_______  MAKE GRID MEMBERSHIPS ______________################
 MULTI MEMBERSHIP AS SEEN IN : https://github.com/ultralytics/yolov5/issues/6998#44 
______________________________________________________________________________

STRUCTURE :  
let pos = membership flattened index. i.e , if image into 2x2 grids.
[[1, 2][3, 4]] would be the indices of the grids, (row major).

[[x, y, w, h, pos] ... numBoxes] -> boxes with membership (pos)
NOTE : we shold add duplicaties for each valid pos. eg: if x < 0.5*gridW and 
y < 0.5*gridH, then then for the curr pos (p1,p2), other valid poses are 
(p1-1, p2), (p1, p2-1), so duplicates must be made with
each of them as a pose. for more info : refer : https://github.com/ultralytics/yolov5/issues/6998#44

accessing memberships while "building targets" : boxes[boxes[:, 4] == reqPos1d] 
 
    where reqPos is the grid position that is required. 

"""


"""
- init numBoxes*3 array 'Aall'. assuming iteself + up/down + left/right border valid.
- init array with possibly valid poses (may exceed image dims) and
  keep poses within the image alone,
- use the above mask to select only the valid boxes from 'Aall'.
- This will be final set of membership boxes accessed by build Targets.

  SUPPORTS ONLY SINGLE IMAGE AT A TIME, NOT BATCHED.
"""

"""
creates a [eps, eps, 1-eps, eps ...] class Label vector with length nC (numClasses)
1-eps value's position corresponds to the class label's position. typically, eps
should be close to 0, when epsilon = 0, there is no label smooth and the result
is just oneHot Encording.  

takes input (currCoords) of the format : 
  [numBoxes x 5(xywh + classLabel)]

returns labels of the format:
  [numBoxes x 15(11classes + 4 (xywh))]
"""
def genSmoothenedLabels(currCoords,
                        epsilon,
                        numClasses):
  gtMask = nn.functional.one_hot(currCoords[:,-1].to(int), num_classes = numClasses)
  labels = (gtMask - epsilon) + ((1- gtMask) * 2 * epsilon)
  currCoords = torch.concat([labels, currCoords[:, :-1]], 1)
  return currCoords

"""
takes input (currCoords) of the format : 
  [numBoxes x 5(xywh + classLabel)]
"""
def genGridMemberships(currCoords : torch.tensor,
                       imHeight : int,
                       imWidth : int,
                       numGrids : int,
                       numClasses : int,
                       verbose : bool):
  insertInd = 0

  currCoords = genSmoothenedLabels(currCoords, 0.1, numClasses)

  gridLenW = imWidth//numGrids
  gridLenH = imHeight//numGrids

  gridRow = currCoords[:, -1] // gridLenH
  gridCol = currCoords[:, -2] // gridLenW


  mems1d = gridRow * numGrids + gridCol
  mems2d = torch.concat([gridRow[..., None], gridCol[..., None]],1)


  gridX = currCoords[:, -4] - gridCol * gridLenW
  gridY = currCoords[:, -3] - gridRow * gridLenH

  numBoxes, _ = currCoords.shape
  memBoxes = torch.zeros(tuple([numBoxes*3, 16]))
  validMems2d = torch.ones(tuple([numBoxes*3, 2])) * -1

  validMems2d[insertInd:numBoxes] = mems2d
  memBoxes[insertInd:numBoxes, :-1]  = currCoords
  insertInd += numBoxes

  tempMask = torch.logical_and(gridX < 0.5*gridLenW, mems2d[:, 1] - 1 > -1)
  numTemp = len(torch.nonzero(tempMask))
  validMems2d[insertInd:insertInd+numTemp] = mems2d[tempMask]
  validMems2d[insertInd:insertInd+numTemp, 1] -= 1
  memBoxes[insertInd:insertInd+numTemp, :-1]  = currCoords[tempMask]
  insertInd += numTemp

  tempMask = torch.logical_and(gridX > 0.5*gridLenW, mems2d[:, 1] + 1 < numGrids)
  numTemp = len(torch.nonzero(tempMask))
  validMems2d[insertInd:insertInd+numTemp] = mems2d[tempMask]
  validMems2d[insertInd:insertInd+numTemp, 1] += 1
  memBoxes[insertInd:insertInd+numTemp, :-1]  = currCoords[tempMask]
  insertInd += numTemp

  tempMask = torch.logical_and(gridY < 0.5*gridLenH, mems2d[:, 0] - 1 > -1)
  numTemp = len(torch.nonzero(tempMask))
  validMems2d[insertInd:insertInd+numTemp] = mems2d[tempMask]
  validMems2d[insertInd:insertInd+numTemp, 0] -= 1
  memBoxes[insertInd:insertInd+numTemp, :-1]  = currCoords[tempMask]
  insertInd += numTemp

  tempMask = torch.logical_and(gridY > 0.5*gridLenH, mems2d[:, 0] + 1 < numGrids)
  numTemp = len(torch.nonzero(tempMask))
  validMems2d[insertInd:insertInd+numTemp] = mems2d[tempMask]
  validMems2d[insertInd:insertInd+numTemp, 0] += 1
  memBoxes[insertInd:insertInd+numTemp, :-1]  = currCoords[tempMask]
  insertInd += numTemp

  memBoxes = memBoxes[validMems2d[:,0] != -1]
  validMems2d = validMems2d[validMems2d[:,0] != -1]
  memBoxes[:,-1] = validMems2d[:,0]*numGrids + validMems2d[:,1]

  if verbose == True:
    print(f"currCoords : {currCoords.shape}")
    for i in currCoords:
      printGridDets(i[0].item(),i[1].item(),numGrids)#REMOVE TRAIN UTILS FROM THIS LINE.
    print(f"grid row length : {gridLenH}, grid col length : {gridLenW}")
    print(f"memBoxes duplications : {memBoxes[numBoxes:]}")

  return memBoxes

def matchAnchors(predBoxes, anchors, thresh):
  numAnchors = len(anchors)
  numPreds = len(predBoxes)
  print(f"num anchors : {numAnchors}, num boxes : {numPreds}")
  whAnchors = anchors[:, 2:]
  whPredBoxes = predBoxes[:, 2:]

  ratioW = whAnchors.tile(numPreds)[:,::2].T / whPredBoxes.tile(numAnchors)[:,::2]
  ratioH = whAnchors.tile(numPreds)[:,1::2].T / whPredBoxes.tile(numAnchors)[:,1::2]

  ratioWMax = torch.max(ratioW, 1 / ratioW)
  ratioHMax = torch.max(ratioH, 1 / ratioH)
  ratioMax = torch.max(ratioWMax, ratioHMax)
  # anchorMask = ratioMax < thresh
  return ratioMax

def buildTargets(currCoords,
                 imWidth,
                 imHeight,
                 numGrids,
                 classes,
                 numClasses,
                 anchorBoxes):
                 
  totalNumGrids = numGrids * numGrids
  numAnchors = len(anchorBoxes)
  numBoxes = len(currCoords)

  currCoords = torch.concat([currCoords, classes[..., None]], 1)

  # Obtaining a getting the boxes with valid Grid memberships.
  # shape : numBoxes*3 x 6 
  """
  include class probabilities as well as it is used in label smothing,
  and class loss computation. that is now the struture will be:

  numBoxes*3 x (numClasses + 4(xywh) + 1(membership) 

  """
  memBoxes =  genGridMemberships(currCoords,
                                 imHeight,
                                 imWidth,
                                 numGrids,
                                 numClasses,
                                 False)

  # getting anchor memberships after computing the defining ratio : 
  # shape : numBoxes*3 x 4
  ratioMax = matchAnchors(memBoxes[:,-4:], torch.tensor(anchorBoxes), 4)

  boxInds, anchorInds = torch.meshgrid(torch.arange(ratioMax.shape[0]),
                                  torch.arange(ratioMax.shape[1]),
                                  indexing='ij')

  memInds = memBoxes[boxInds.flatten()][:,-1].to(torch.int)

  boxesToLoad = memBoxes[boxInds.flatten()][:,:-1]

  # converting the defining ratio and the grid membership Boxes tensor to 
  # shape : numGrids x numAnchors x numBoxes 
  targetInds = torch.ones(totalNumGrids, numAnchors, numBoxes) * np.inf

  targets = torch.ones(totalNumGrids, numAnchors, numBoxes, 15) * np.inf

  tailTracker = np.zeros(targets.shape[:2]).astype(int)
  memTailInds = []
  for memInd, AnchorInd in zip(memInds.tolist(),
                              anchorInds.flatten().to(torch.int).tolist()):
    memTailInds.append(tailTracker[memInd, AnchorInd])
    tailTracker[memInd, AnchorInd] +=1

  targetInds[memInds, anchorInds.flatten(), memTailInds] = ratioMax.flatten()

  #loading boxes into the targets tensor
  targets[memInds, anchorInds.flatten(), memTailInds] = boxesToLoad

  gridInds, anchorInds = torch.meshgrid(torch.arange(totalNumGrids),
                                  torch.arange(numAnchors),
                                  indexing='ij')

  finalTargets = torch.zeros([totalNumGrids, numAnchors, 15])

  # Making a tensor with the box with the closest ratio to each anchor box 
  # for each grid
  finalTargets[gridInds.flatten(), anchorInds.flatten()] = targets[gridInds.flatten(),
                                                                    anchorInds.flatten(),
                                                                    torch.argmin(targetInds, 2).flatten()]
  return finalTargets, targets, targetInds

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#_______________________________________________________________________________
