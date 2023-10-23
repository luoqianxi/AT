# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable

from detectron2.layers import ShapeSpec,GradientScalarLayer

# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):    #dict
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,

        cam_heads: nn.Module,

        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        #(2)image-level I-Classifier
        self.cam_heads = cam_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        # self.D_img = None
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.bceLoss_func = nn.BCEWithLogitsLoss()
    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),

            "cam_heads": CAMHead(backbone.output_shape(), cfg.MODEL.ROI_HEADS.NUM_CLASSES),

            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # "dis_loss_ratio": cfg.xxx,
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """

        
        images = [x["image"].to(self.device) for x in batched_inputs]    #源域
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]  #目标域
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t
    
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]     #源域
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1

        if branch == "domain":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)

            #print("11111111111111111111image-level")
            #images = self.preprocess_image(batched_inputs)
            #if "instances" in batched_inputs[0]:
            #    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            #else:
            #    gt_instances = None

            #features = self.backbone(images.tensor)
            #(2) image-level class-wise alignment
            #ic_losses = self.cam_heads(features[self.roi_heads.in_features[0]], gt_instances)

            #D_img_out_s = self.D_img()
            #loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))
   
            images_s, images_t = self.preprocess_image_train(batched_inputs)
            features = self.backbone(images_s.tensor)  #dict

            # import pdb
            # pdb.set_trace()
           
            #对抗性训练的损失函数
            #GRL（源域）
            features_s = grad_reverse(features[self.dis_type])   #torch.cuda.FloatTensor
            #print(features_s.type())   #torch.cuda.FloatTensor
            #print(features_s.data.size())    #torch.Size([2, 1024, 38, 57]) 
            

            #I-Classifier（源域）
            images = self.preprocess_image(batched_inputs)   
        
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            features = self.backbone(images.tensor)
            #(2) image-level class-wise alignment
            ic_losses = self.cam_heads(features[self.roi_heads.in_features[0]], gt_instances)     #features_s  2维  torch.Size([2, 20])


            #鉴别器
            #self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) 
            D_img_out_s = self.D_img(features_s)        #需要一个4维的输入
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_t = self.backbone(images_t.tensor)
            features_t = grad_reverse(features_t[self.dis_type])
            #features_t = grad_reverse(features_t['p2'])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            # import pdb
            # pdb.set_trace()
  

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            losses["loss_D_img_t"] = loss_D_img_t
            losses["ic_losses"] = ic_losses
            #print(ic_losses)
            return losses, [], [], [], None

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None, None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

class CAMHead(nn.Module):
    """
    Image-level multi-label classifier for image-level class-wise alignment
    """

    def __init__(self, backbone_out_shape: Dict[str, ShapeSpec], num_classes: int):
        super(CAMHead, self).__init__()
        if 'res5' in backbone_out_shape.keys():
            in_channels = 2048
        elif 'res4' in backbone_out_shape.keys():
            in_channels = 1024
        elif 'plain5' in backbone_out_shape.keys():
            in_channels = 512
        else:
            raise KeyError("Unknown backbone output name: {}".format(backbone_out_shape.keys()))
        self.num_classes = num_classes  

        #print(in_channels)  #1024
        self.cam_conv = nn.Conv2d(in_channels, self.num_classes, kernel_size=1, bias=False)
        weight_init.c2_msra_fill(self.cam_conv) #None  权重初始化
        

    def forward(self, x, gt_instances):
        #print(x.shape)  #torch.Size([2, 1024, 38, 57])
        
        x = self.cam_conv(x)    
        #print(x.shape)       #torch.Size([2, 20, 38, 57])
        logits = F.avg_pool2d(x, (x.size(2), x.size(3)))
        #print(logits.shape)   #torch.Size([2, 20, 1, 1])     


        '''
        view()函数的功能根reshape类似，用来转换size大小。
        x = x.view(batchsize, -1)中batchsize指转换后有几行，
        而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        '''
        logits = logits.view(-1, self.num_classes)  #这句话一般出现在model类的forward函数中，具体位置一般都是在调用分类器之前。
        #print(logits.shape)  #torch.Size([2, 20])

        if gt_instances is None:
            return {'loss_cam': 0.0 * logits.sum()}

        gt_classes_img_oh = self.get_image_level_gt(gt_instances)

        losses = F.binary_cross_entropy_with_logits(
            logits, gt_classes_img_oh, reduction='mean'
        )

        return losses * 0.1

    @torch.no_grad()
    def get_image_level_gt(self, targets):
        """
        Convert instance-level annotations to image-level   把实例级标签转化为图像级
        """
        gt_classes_img = [torch.unique(t.gt_classes, sorted=True) for t in targets]
        gt_classes_img_int = [gt.to(torch.int64) for gt in gt_classes_img]
        gt_classes_img_oh = torch.cat(
            [
                torch.zeros(
                    (1, self.num_classes), dtype=torch.float, device=gt_classes_img[0].device
                ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
                for gt in gt_classes_img_int
            ],
            dim=0,
        )
        return gt_classes_img_oh

class GlobalDAHead(nn.Module):
    """
    Global domain classifier for image-level class-agnostic alignment
    """

    def __init__(self, backbone_out_shape: Dict[str, ShapeSpec]):
        super(GlobalDAHead, self).__init__()
        if 'res5' in backbone_out_shape.keys():
            in_channels = 2048
        elif 'res4' in backbone_out_shape.keys():
            in_channels = 1024
        elif 'plain5' in backbone_out_shape.keys():
            in_channels = 512
        else:
            raise KeyError("Unknown backbone output name: {}".format(backbone_out_shape.keys()))

        self.da_conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.da_conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.da_conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.da_bn1 = nn.BatchNorm2d(512)
        self.da_bn2 = nn.BatchNorm2d(128)
        self.da_bn3 = nn.BatchNorm2d(128)
        self.da_fc = nn.Linear(128, 1)

        self.gama = 5
        grl_weight = 1.0
        self.grl = GradientScalarLayer(-1.0 * grl_weight)

    def forward(self, x):
        x = self.grl(x)

        x = F.dropout(F.relu(self.da_bn1(self.da_conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.da_bn2(self.da_conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.da_bn3(self.da_conv3(x))), training=self.training)

        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        x = self.da_fc(x)

        da_targets = torch.zeros_like(x, requires_grad=False)
        num_source_input = x.shape[0] // 2
        da_targets[:num_source_input, ...] += 1
        losses = sigmoid_focal_loss_jit(x, da_targets, gamma=self.gama, reduction='mean')

        return {'loss_global_da': losses}