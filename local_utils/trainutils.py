import torch
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



#______________________________________________________________________________
##################3_______  MAKE GRID MEMBERSHIPS ______________################
##########  MULTI MEMBERSHIP AS SEEN IN : https://github.com/ultralytics/yolov5/issues/6998#44 
#______________________________________________________________________________
"""
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
- init numBoxes*3 array 'Aall'. assuming 1 + 2 border valid.
- init array with possibly valid poses (may exceed image dims) and 
  keep poses within the image alone, 
- use the above mask to select only the valid boxes from 'Aall'.
- This will be final set of membership boxes accessed by build Targets.

  SUPPORTS ONLY SINGLE IMAGE AT A TIME, NOT BATCHED.
"""
def genGridMemberships(currCoords : torch.tensor,
                   imHeight : int,
                   imWidth : int,
                   numGrids : int,
                   verbose : bool):
  insertInd = 0

  gridLenW = imWidth//numGrids
  gridLenH = imHeight//numGrids
  
  gridRow = currCoords[:, 1] // gridLenH
  gridCol = currCoords[:, 0] // gridLenW


  mems1d = gridRow * numGrids + gridCol
  mems2d = torch.concat([gridRow[..., None], gridCol[..., None]],1)


  gridX = currCoords[:, 0] - gridCol * gridLenW
  gridY = currCoords[:, 1] - gridRow * gridLenH

  numBoxes, _ = currCoords.shape
  memBoxes = torch.zeros(tuple([numBoxes*3, 5]))
  validMems2d = torch.ones(tuple([numBoxes*3, 2])) * -1

  validMems2d[insertInd:numBoxes] = mems2d
  memBoxes[insertInd:numBoxes, :4]  = currCoords
  insertInd += numBoxes

  tempMask = torch.logical_and(gridX < 0.5*gridLenW, mems2d[:, 1] - 1 > -1)
  numTemp = len(torch.nonzero(tempMask))
  validMems2d[insertInd:insertInd+numTemp] = mems2d[tempMask]
  validMems2d[insertInd:insertInd+numTemp, 1] -= 1
  memBoxes[insertInd:insertInd+numTemp, :4]  = currCoords[tempMask]
  insertInd += numTemp

  tempMask = torch.logical_and(gridX > 0.5*gridLenW, mems2d[:, 1] + 1 < numGrids)
  numTemp = len(torch.nonzero(tempMask))
  validMems2d[insertInd:insertInd+numTemp] = mems2d[tempMask]
  validMems2d[insertInd:insertInd+numTemp, 1] += 1
  memBoxes[insertInd:insertInd+numTemp, :4]  = currCoords[tempMask]
  insertInd += numTemp

  tempMask = torch.logical_and(gridY < 0.5*gridLenH, mems2d[:, 0] - 1 > -1)
  numTemp = len(torch.nonzero(tempMask))
  validMems2d[insertInd:insertInd+numTemp] = mems2d[tempMask]
  validMems2d[insertInd:insertInd+numTemp, 0] -= 1
  memBoxes[insertInd:insertInd+numTemp, :4]  = currCoords[tempMask]
  insertInd += numTemp

  tempMask = torch.logical_and(gridY > 0.5*gridLenH, mems2d[:, 0] + 1 < numGrids)
  numTemp = len(torch.nonzero(tempMask))
  validMems2d[insertInd:insertInd+numTemp] = mems2d[tempMask]
  validMems2d[insertInd:insertInd+numTemp, 0] += 1
  memBoxes[insertInd:insertInd+numTemp, :4]  = currCoords[tempMask]
  insertInd += numTemp

  memBoxes = memBoxes[validMems2d[:,0] != -1]
  validMems2d = validMems2d[validMems2d[:,0] != -1]
  memBoxes[:,4] = validMems2d[:,0]*numGrids + validMems2d[:,1]

  if verbose == True:
    print(f"currCoords : {currCoords.shape}")
    for i in currCoords:
      printGridDets(i[0].item(),i[1].item(),numGrids)
    print(f"grid row length : {gridLenH}, grid col length : {gridLenW}")
    print(f"memBoxes duplications : {memBoxes[numBoxes:]}")

  return memBoxes
