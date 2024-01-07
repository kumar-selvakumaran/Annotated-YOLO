import torch
from torch import nn
from collections import OrderedDict

"""
- The custom yolo model is implemented as seen in the architecture diagrams
- the parts , as in the diagram are:
  1. backbone
  2. neck
  3-7. heads 1-5

- head numbers are defined as in the architecture diagram
"""

class normed_Conv2d(nn.Conv2d):
  def __init__(self, do_norm = True, **kwargs):
    super(normed_Conv2d, self).__init__(**kwargs)
    self.normed = do_norm
    self.norm_layer = nn.BatchNorm2d(num_features = self.out_channels)
    self.activation = nn.SiLU()

  def forward(self, x):
    x = super(normed_Conv2d, self).forward(x)
    if self.normed:
      x = self.norm_layer(x)
    x = self.activation(x)
    return x




""" NN.SEQUENTIAL CAN BE USED INSTEAD OF MODULELIST, BUT IT TAKES THE SAME TIME AS LOOPING OVER LAYERS IN MODULELIST
DEFINITION SHOULD BE DONE BY MAKING AN ORDEREDDICT OF THE LAYERS"""
class bottleneck(nn.Module):
  def __init__(self,
               in_channels,
               num_1x1_filters_per_rep,
               num_3x3_filters_per_rep,
               num_reps,
               block_number = None,
               downsample = False
               ):
    super(bottleneck, self).__init__()

    self.num_1x1_filters_per_rep = num_1x1_filters_per_rep
    self.num_3x3_filters_per_rep = num_3x3_filters_per_rep
    self.num_reps = num_reps

    layer_block = nn.ModuleList()

    if block_number == None:
        block_number = "NA"

    if downsample:
      padding_3x3 = 0
    else:
      padding_3x3 = 1

    for rep_no in range(self.num_reps):

      rep = OrderedDict()

      rep[f"block_{block_number}_rep_{rep_no}_ConvBNSiLU_1x1"] = normed_Conv2d(
          in_channels =  in_channels,
          out_channels = self.num_1x1_filters_per_rep,
          kernel_size = 1,
          # padding = 0 #same
          )

      rep[f"block_{block_number}_rep_{rep_no}_ConvBNSiLU_3x3"] = normed_Conv2d(
          in_channels = self.num_1x1_filters_per_rep,
          out_channels = self.num_3x3_filters_per_rep,
          kernel_size = 3,
          padding = padding_3x3 # same
          )

      rep = nn.Sequential(rep)

      layer_block.add_module(name = f"block_{block_number}_rep_{rep_no}",
                             module = rep)

      in_channels = self.num_3x3_filters_per_rep

    self.layer_block = layer_block


  def forward(self, x):
    init_layer = True
    for layer in self.layer_block:
      if init_layer:
        x = layer(x)
      else:
        x = layer(x) + x

    return x



class backbone(nn.Module):
  def __init__(self):
    super(backbone, self).__init__()

    self.super_block = OrderedDict()

    ############### INPUT LAYER ################
    self.super_block["input_layer"] = normed_Conv2d(
        in_channels = 3,
        out_channels = 32,
        kernel_size = 3
    )
    ############################################

    ######## FIRST CONV ONLY BOTTLENECK ########
    self.super_block["ConvBNSiLU_0"] = normed_Conv2d(
        in_channels = 32,
        out_channels = 32,
        kernel_size = 3
    )

    self.super_block["ConvBSSiLU_1"] = normed_Conv2d(
        in_channels = 32,
        out_channels = 64,
        kernel_size = 3,
        stride = 2,
        padding = 1
    )
    ############################################

    #SEQUENTIAL REPETITVE BOTTLENECK SEQUENCE ##

    self.super_block["bottleneck_0"] = bottleneck(
        in_channels = 64,
        num_1x1_filters_per_rep = 32,
        num_3x3_filters_per_rep = 64,
        num_reps = 2,
        block_number = 0,
        downsample = True,
    )

    self.super_block["ConvBSSiLU_2"] = normed_Conv2d(
        in_channels = 64,
        out_channels = 128,
        kernel_size = 3,
        stride = 2,
        padding = 1
    )

    self.super_block["bottleneck_1"] = bottleneck(
        in_channels = 128,
        num_1x1_filters_per_rep = 64,
        num_3x3_filters_per_rep = 128,
        num_reps = 6,
        block_number = 1,
        downsample = True
    )

    self.super_block["ConvBNSiLU_3"] = normed_Conv2d(
        in_channels = 128,
        out_channels = 256,
        kernel_size = 3,
        stride = 2,
        padding = 1
    )

    self.super_block["bottleneck_2"] = bottleneck(
        in_channels = 256,
        num_1x1_filters_per_rep = 128,
        num_3x3_filters_per_rep = 256,
        num_reps = 8,
        block_number = 2,
        downsample = True

    )

    self.super_block["ConvBNSiLU_4"] = normed_Conv2d(
        in_channels = 256,
        out_channels = 512,
        kernel_size = 3,
        stride = 2,
        padding = 1,
    )

    self.super_block["bottleneck_3"] = bottleneck(
        in_channels = 512,
        num_1x1_filters_per_rep = 256,
        num_3x3_filters_per_rep = 512,
        num_reps = 8,
        block_number = 3,
        downsample = True
    )
    ##########################################

    self.super_block = nn.Sequential(self.super_block)

  def forward(self, x):
    x = self.super_block(x)

    return x


class neck(nn.Module):
  def __init__(self):
    super(neck, self).__init__()

    self.layers = nn.ModuleDict()

    self.layers.add_module( name = "ConvBNSiLU_0",
                           module = normed_Conv2d(
                               in_channels = 512,
                               out_channels = 1024,
                               kernel_size = 3,
                               stride = 2,
                               padding = 1
                           ))

    self.layers.add_module( name = "bottleneck_0",
                           module = bottleneck(
                               in_channels = 1024,
                               num_1x1_filters_per_rep = 512,
                               num_3x3_filters_per_rep = 1024,
                               num_reps = 4,
                               block_number = 0
                           ))

    self.layers.add_module( name = "ConvBNSiLU_1",
                           module = normed_Conv2d(
                               in_channels = 1024,
                               out_channels = 512,
                               kernel_size = 3,
                               stride = 2,
                               padding = 1
                           ))

    self.layers.add_module( name = "bottleneck_1",
                           module = bottleneck(
                               in_channels = 512,
                               num_1x1_filters_per_rep = 256,
                               num_3x3_filters_per_rep = 512,
                               num_reps = 4,
                               block_number = 1
                           ))

  def forward(self, x):
    self.input = x
    self.outputs = OrderedDict()
    for name, layer in self.layers.items():
      x = layer(x)
      self.outputs[name] = x

    return self.outputs.copy()

class head_downsample(nn.Module):
  def __init__(self,
               in_channels):
    super(head_downsample, self).__init__()

    self.head = OrderedDict()

    self.in_channels = in_channels

    self.head["bottleneck_0"] = bottleneck(
        in_channels = self.in_channels,
        num_1x1_filters_per_rep = 256,
        num_3x3_filters_per_rep = 512,
        num_reps = 1
    )

    self.head["ConvBNSiLU_0"] = normed_Conv2d(
        do_norm = False,
        in_channels = 512,
        out_channels = 80,
        kernel_size = 3,
        stride = 2,
        padding = 1
    )
    
    self.head = nn.Sequential(self.head)

  def forward(self, x1, x2):
    x = torch.concat([x1, x2], axis = 1)

    x = self.head(x)

    return x

class head_upsample(nn.Module):
  def __init__(self,
               in_channels):
    super(head_upsample, self).__init__()

    self.upsampler = nn.Upsample(
        scale_factor = 2,
        mode = "nearest",
    )

    self.in_channels = in_channels 
    
    self.head = normed_Conv2d(
        do_norm = False,
        in_channels = self.in_channels,
        out_channels = 80,
        kernel_size = 3,
        stride = 2,
        padding = 1
    )

  def forward(self, x1, x2):
    x2  = self.upsampler(x2)

    x = torch.concat([x1, x2], axis = 1)

    x = self.head(x)

    return x


class head_3(nn.Module):
  def __init__(self):
    super(head_3, self).__init__()

    self.head = normed_Conv2d(
        do_norm = False,
        in_channels = 512,
        out_channels = 80,
        kernel_size = 4,
        # stride = 2,
        # padding = 1
    )

  def forward(self, x):
    x = self.head(x)

    return x


class my_yolo(nn.Module):
  def __init__(self):
    super(my_yolo, self).__init__()

    self.backbone = backbone()

    self.neck = neck()

    self.heads = nn.ModuleDict()

    self.heads.add_module(
        name = "head_1",
        module = head_downsample(
            in_channels = 2048
        )
    )

    self.heads.add_module(
        name = "head_2",
        module = head_downsample(
            in_channels = 1024
        )
    )

    self.heads.add_module(
        name = "head_3",
        module = head_3()
    )

    self.heads.add_module(
        name = "head_4",
        module = head_upsample(
            in_channels = 1536
        )
    )

    self.heads.add_module(
        name = "head_5",
        module = head_upsample(
            in_channels = 1536
        )
    )

  def forward(self, x):

    output_dict = OrderedDict()

    x = self.backbone(x)

    neck_outputs = self.neck(x)

    output_dict["head_1"] = self.heads["head_1"](
        neck_outputs["ConvBNSiLU_0"],
        neck_outputs["bottleneck_0"]
    )

    output_dict["head_2"] = self.heads["head_2"](
        neck_outputs["ConvBNSiLU_1"],
        neck_outputs["bottleneck_1"]
    )    

    output_dict["head_3"] = self.heads["head_3"](x)

    output_dict["head_4"] = self.heads["head_4"](
        neck_outputs["bottleneck_0"],
        neck_outputs["bottleneck_1"]
    )

    output_dict["head_5"] = self.heads["head_5"](
        x,
        neck_outputs["bottleneck_0"]
    )

    return output_dict


