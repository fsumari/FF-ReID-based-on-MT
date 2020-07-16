import cv2
import time
import numpy as np
import sys
import tensorflow as tf
#from PIL import Image
import os
#********************

import matplotlib.pyplot as plt
import glob

# Con esto, retrocedi un carpeta .. para agregar , pYOLO
#basedir = os.path.dirname(__file__)
#print('base : ',basedir)
#sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
#print('base2 :',os.path.dirname(__file__))
##************************************************************************

from pYOLO.core import utils
#from pReID import cuhk03_dataset


#****1********PARAMETERS OD**********
#PARAMETRO
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "pYOLO/yolov3_coco.pb"

# PARA LA DETECCION DE OBJETOS
num_classes     = 80
input_size      = 416#608
graph           = tf.Graph()
#

class YOLO(object):
    def __init__(self):
        self.return_elements = return_elements
        self.pb_file = pb_file
        self.num_classes     = num_classes
        self.input_size      = input_size
        self.graph           = graph 
        self.return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)
        self.sess = tf.Session(graph=self.graph)
        self.info            = ''
    
    def pImageDetection(self, original_image):
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(np.copy(original_image), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        #return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
        #print('array IMAGE: ', image_data.shape)

        #with tf.Session(graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
                    feed_dict={ self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        #print('TF CONCATENACION SBBOX:', np.shape(pred_sbbox))
        #print('predd bbox: ',pred_bbox.shape)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        pbboxes = []
        for bbox in bboxes:
            if bbox[5] == 0:
                #pbboxes.append(bbox[:4])
                x = bbox[0]  
                y = bbox[1]  
                w = bbox[2]-bbox[0]
                h = bbox[3]-bbox[1]
                pbboxes.append([x,y,w,h])
        return pbboxes

    def pImageDetection_batch(self, frames, batch_size = 6):
                
        image_data = []
        original_frame_size = None
        #print('FRAMES : ', np.shape(frames))
        for original_frame in frames:
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            if original_frame_size == None:
                original_frame_size = original_frame.shape[:2]   
            ori_frame_data = utils.image_preporcess(np.copy(original_frame), [self.input_size, self.input_size])
            #ori_frame_data = ori_frame_data[np.newaxis, ...]
            image_data.append(ori_frame_data)

        image_data = np.asarray(image_data)
        print('array IMAGE: ', image_data.shape)
        ##********************************
        pred_sbbox, pred_mbbox, pred_lbbox  = self.sess.run( [self.return_tensors[1], \
                                                self.return_tensors[2], \
                                                self.return_tensors[3]],\
                                                feed_dict = {self.return_tensors[0]: image_data})
        print('TF CONCATENACION SBBOX:', np.shape(pred_sbbox))
        pred_batch_bbox = []
        pbbx_batch = []
        for p_sbb, p_mbb, p_lbb in zip(pred_sbbox, pred_mbbox, pred_lbbox):
            pred_bbox = np.concatenate([np.reshape(p_sbb, (-1, 5 + self.num_classes)),
                                    np.reshape(p_mbb, (-1, 5 + self.num_classes)),
                                    np.reshape(p_lbb, (-1, 5 + self.num_classes))], axis=0)
            #pred_batch_bbox.append(pred_bbox)
            bboxes = utils.postprocess_boxes(pred_bbox, original_frame_size, self.input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            pbboxes = []
            for bbox in bboxes:
                if bbox[5] == 0:
                    #pbboxes.append(bbox[:4])
                    
                    x = bbox[0]  
                    y = bbox[1]  
                    w = bbox[2]-bbox[0]
                    h = bbox[3]-bbox[1]
                    '''
                    x = bbox[0]  
                    y = bbox[1]  
                    w = bbox[2]
                    h = bbox[3]
                    '''
                    pbboxes.append([x,y,w,h])
            pbbx_batch.append(pbboxes)
        
        return pbbx_batch
