"""COCO-style evaluation metrics.
Implements the interface of COCO API and metric_fn in tf.TPUEstimator.
COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
from absl import flags
from absl import logging

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class EvaluationMetric(object):
    """COCO evaluation metric class."""

    def __init__(self, filename=None, output_of_NMS_pred):
        if filename:
        self.coco_gt = COCO(filename) # gt: ground truth
        self.filename = filename # JSON file name
        self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                            'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
        self.output_of_NMS_pred = []
        self.ann_json_dict = {
            'images': [],
            'type': 'instances',
            'annotations': [],
            'categories': []
        }
        self.image_id = 1
        self.annotation_id = 1
        self.category_ids = []
        
    def estimator_metric(self, output_of_NMS_pred, ground_truth_data):
        """
        detections: [image_id, x, y, width, height, score, class]
        groundtruth_data: representing [y1, x1, y2, x2, is_crowd, area, class]
        """
        