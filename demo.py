import tensorflow as tf
import numpy as np
import cv2
import cuhk03_dataset
import time
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import copy
import sys
from pYOLO import yolo_cropper
from pReID import personReID
from matplotlib.widgets import TextBox
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from mtCropping import MTcropper

FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')
#tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
#tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
#tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'demo', 'Mode graph, val, demo, test')
#tf.flags.DEFINE_string('image1', '', 'First image path to compare')
#tf.flags.DEFINE_string('image2', '', 'Second image path to compare')
#tf.flags.DEFINE_string('path_test', '', 'Images path to compare')
#
tf.flags.DEFINE_string('query_path', '', 'First image path to compare')
tf.flags.DEFINE_string('cropps_path', '', 'gallery')
#
tf.flags.DEFINE_string('queries_path','','carpet of queries path to compare')
tf.flags.DEFINE_string('video_path','','video path to cropping')
tf.flags.DEFINE_string('data_dir', '../data/dataReal_pt1', 'path to dataset')#DATASET
tf.flags.DEFINE_string('p_name', 'predictV2', 'name path of predict file')
#
#tf.flags.DEFINE_string('t','' ,'t is number of frames per sequence')
tf.flags.DEFINE_string('t_skip','' ,'t is number of frames per sequence')
tf.flags.DEFINE_string('beta','' ,'beta is the threshold')
tf.flags.DEFINE_string('eta','' ,'eta is the TOP')
#
tf.flags.DEFINE_string('threshold', '0.9' ,'threshold for True reid')
tf.flags.DEFINE_integer('top', '20' ,'Top for reid')
#
tf.flags.DEFINE_string('graph', '' ,'Top for reid')
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160

def main(argv=None):
    
    #### INICIALIZO RED NEURAL

    if FLAGS.mode == 'classic_test':
        fpath, fname = os.path.split(FLAGS.cropps_path)
        fpath_reid = fpath + '/out_reid'
        if not os.path.isfile(fpath_reid):
            os.system('mkdir '+ fpath_reid)
        reidentifier = personReID.personReIdentifier()
        topN = FLAGS.top
        print('*******CLASSIC RE-ID TEST*******')
        reidentifier.PersonReIdentification(FLAGS.query_path, FLAGS.cropps_path, fpath_reid, topN,show_query = True)

    if FLAGS.mode == 'rw_test':
        cropper = yolo_cropper.YOLOcropper()#objeto que recorta
        reidentifier = personReID.personReIdentifier()#objeto que hace reid
                        
        print('*******REAL WORLD RE-ID SIMPLE TEST*******')
        ############# FLAGS
        fpath, fname = os.path.split(FLAGS.video_path )
        fpath_cropps = fpath + '/cropps'
        if not os.path.isfile(fpath_cropps):
            os.system('mkdir '+ fpath_cropps)
        
        tf.flags.DEFINE_string('out_cropps_path', fpath_cropps ,'out cropps path to gallery')
        
        #cropper.personCropping(FLAGS.video_path, FLAGS.out_cropps_path)
        topN = FLAGS.top
        fpath_reid = fpath + '/out_reid'
        if not os.path.isfile(fpath_reid):
            os.system('mkdir '+ fpath_reid)
        ############### FIN FLAGS
        reidentifier.PersonReIdentification(FLAGS.query_path, FLAGS.out_cropps_path, fpath_reid, topN, show_query = True)
        #personReidentification(sess, FLAGS.query_path, FLAGS.out_cropps_path, fpath_reid, images, is_train, inference)
    if FLAGS.mode == 'mt_test':
        
        reidentifier = personReID.personReIdentifier()#objeto que hace reid
        cropper = MTcropper(1) #recortar tracks
        print('*******REAL WORLD MT+RE-ID SIMPLE TEST*******')
        ############# FLAGS
        fpath, fname = os.path.split(FLAGS.video_path )
        fpath_cropps = fpath + '/mtcropps'
        if not os.path.isfile(fpath_cropps):
            os.system('mkdir '+ fpath_cropps)
        
        tf.flags.DEFINE_string('out_cropps_path', fpath_cropps ,'out cropps path to gallery')
        
        cropper.personCropping(FLAGS.video_path, FLAGS.out_cropps_path)
        topN = FLAGS.top
        fpath_reid = fpath + '/out_mt+reid'
        if not os.path.isfile(fpath_reid):
            os.system('mkdir '+ fpath_reid)
        ############### FIN FLAGS
        reidentifier.PersonReIdentification_MT(FLAGS.query_path, FLAGS.out_cropps_path, fpath_reid, topN, show_query = True)
        
    #*************************************************************

        
if __name__ == '__main__':
    tf.compat.v1.app.run()
