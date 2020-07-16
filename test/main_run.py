import cv2
import time
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import copy

# Con esto, retrocedi un carpeta .. para agregar , pYOLO
basedir = os.path.dirname(__file__)
print('base : ',basedir)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
print('base2 :',os.path.dirname(__file__))

##************************************************************************
from pYOLO import yolo_cropper
from pReID import cuhk03_dataset
from pReID import personReID
#****1********PARAMETERS REID**********
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160
def sortSecond(val):
    return val[1]
def sortFirst(val):
    return val[0]
#****2********PARAMETERS REID**********

#****1*********FLAGS OLD REID***********
FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')#TRAIN REID
#tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')#TRAIN REID
#tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')#MODELO REID

#tf.flags.DEFINE_float('learning_rate', '0.01', '')#TRAIN REID
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test, data')#MODO PARA REID
#####
#tf.flags.DEFINE_string('image1', '', 'First image path to compare')
#tf.flags.DEFINE_string('image2', '', 'Second image path to compare')
#####

#***2**********FLAGS OLD REID***********

#**1***********FLAGS NEW REID***********
tf.flags.DEFINE_string('query_path','','query path to compare')
tf.flags.DEFINE_string('queries_path','','carpet of queries path to compare')
tf.flags.DEFINE_string('video_path','','video path to cropping')
tf.flags.DEFINE_string('data_dir', '../data/dataSet', 'path to dataset')#DATASET


#**2***********FLAGS NEW REID***********

reidentifier = personReID.personReIdentifier()#objeto que hace el reid
#FLAGS.batch_size = 1

def getFrameNumber(count):
    count = count + 1
    return count
def sortSecond(val):
    return val[1]

def sortFirst(val):
    return val[0]

def personReID_RealTest(queries_path, video_path):
    queries = sorted(glob.glob(queries_path+'/*.png'))
    print(len(queries))

    # hacer las sequencias**************
    fpath, fname = os.path.split(video_path )
    seq_path = fpath+'/seq_videos'
    if not os.path.isfile(seq_path):
        os.system('mkdir '+ seq_path)

    carpet = 0
    cont_seq=0
    n_carpet=''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tam_seq = 30 * fps # sequencias de 30 segundos
    
    while(True):
        ret, frame = cap.read()
        if ret == True:
            height, width, layers = frame.shape
            size = (width,height)
            #print('cont seq: ',cont_seq)
            #print('tam seq: ',tam_seq)
            #print('carpet: ',carpet)
            if(cont_seq <= tam_seq ):
                if(cont_seq==tam_seq):
                    cont_seq = 0
                    carpet = carpet+1
                
                if(cont_seq==0):
                    ncarpet = '{0:06}'.format(getFrameNumber(carpet))                    
                    os.system('mkdir '+ seq_path+'/'+ncarpet)
                    pathOut = seq_path+'/'+ncarpet+'/sub_video.avi'
                    out = cv2.VideoWriter(pathOut , cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                out.write(frame)
                cont_seq = cont_seq + 1
        else:
            break
    #FIN DE HACER SEQUENCIAS**********************************************

    seq_videos = sorted(glob.glob(seq_path+'/*'))
    
    print(queries)
    print(seq_videos)
    
    for v in seq_videos:
        ############# FLAGS
        #fpath, fname = os.path.split(v)
        fpath_cropps = v + '/cropps'
        if not os.path.isfile(fpath_cropps):
            os.system('mkdir '+ fpath_cropps)
        #tf.flags.DEFINE_string('out_cropps_path', fpath_cropps ,'out cropps path to gallery')

        topN = 10

        
        #tf.flags.DEFINE_string('out_reid_path', fpath_reid ,'out timestamps and topN of ReID')
        ############### FIN FLAGS
        cropper.personCropping( v+'/sub_video.avi', fpath_cropps)

        for q in queries:
            qpath, qname = os.path.split(q)
            temp_qname = qname
            qname = qname.split('.')
            
            fpath_reid = v + '/out_reid_'+ qname[0]
            if not os.path.isfile(fpath_reid):
                os.system('mkdir '+ fpath_reid)
            ## guardo la query
            temp_query = cv2.imread(q)

            cv2.imwrite(fpath_reid+'/'+temp_qname ,temp_query)
            ##
            fpath_reid_out = fpath_reid +'/top'+str(topN)
            if not os.path.isfile(fpath_reid_out):
                os.system('mkdir '+ fpath_reid_out)
            reidentifier.PersonReIdentification(q ,fpath_cropps, fpath_reid_out, topN)

#def generate_dict_truth_predict(queries_ID, sequences_ID):


def calculate_relations(dict_values, m_truth, m_predict, size_seq = 5, ID_limit = 100):
    tp, tn, fp, fn = 0,0,0,0
    dict_truth = copy.deepcopy(dict_values)
    dict_predict = copy.deepcopy(dict_values)

    for list_truth in m_truth:
        list_truth.sort(key = sortSecond , reverse = False)
        for tup_t in list_truth:
            try:
                dict_truth[int(tup_t[1])][int(tup_t[9])] = 1
            except:
                continue
    print('dictionary of values truth: \n', dict_truth)
    ####
    
    for list_predict in m_predict:
        list_predict.sort(key = sortSecond , reverse = False)
        for tup_p in list_predict:
            try:
                dict_predict[int(tup_p[0])][int(tup_p[1])] = 1 # en el id, con seq
            except:
                continue
    
    print('dictionary of values predict: \n', dict_predict)
    ###### CALCULANDO ... TP, TN, FP, FN
    
    for key_query_ID in dict_values:        
        #print('key: ', key_query_ID)
        for key_seq in dict_values[key_query_ID]:
            if(dict_predict[int(key_query_ID)][key_seq] == 1 and dict_truth[int(key_query_ID)][key_seq] == 1):
                tp = tp+1
            elif(dict_predict[int(key_query_ID)][key_seq] == 0 and dict_truth[int(key_query_ID)][key_seq] == 0):
                tn = tn+1
            elif(dict_predict[int(key_query_ID)][key_seq] == 1 and dict_truth[int(key_query_ID)][key_seq] == 0):
                fp = fp+1
            elif(dict_predict[int(key_query_ID)][key_seq] == 0 and dict_truth[int(key_query_ID)][key_seq] == 1):
                fn = fn+1
    
    return tp, tn, fp, fn

def main(argv=None):
    #if FLAGS.mode != 'metrics':
        #global cropper
        #cropper = yolo_cropper.YOLOcropper()#objeto que recorta
        #reidentifier 
        

    if FLAGS.mode == 'simple_test':#una query, un video sin sacar sequencias
        print('*******SIMPLE TEST*******')
        ############# FLAGS
        fpath, fname = os.path.split(FLAGS.video_path )
        fpath_cropps = fpath + '/cropps'
        if not os.path.isfile(fpath_cropps):
            os.system('mkdir '+ fpath_cropps)
        tf.flags.DEFINE_string('out_cropps_path', fpath_cropps ,'out cropps path to gallery')

        topN = 10

        fpath_reid = fpath + '/out_reid_'+str(topN)
        if not os.path.isfile(fpath_reid):
            os.system('mkdir '+ fpath_reid)
        tf.flags.DEFINE_string('out_reid_path', fpath_reid ,'out timestamps and topN of ReID')
        ############### FIN FLAGS
        cropper.personCropping(FLAGS.video_path, FLAGS.out_cropps_path)
        reidentifier.PersonReIdentification(FLAGS.query_path, FLAGS.out_cropps_path, FLAGS.out_reid_path, topN)
        
    if FLAGS.mode == 'real_test':#varias querys, un video > 30 seg, paras sacar sequencias
        topN = 10
        queries = sorted(glob.glob(FLAGS.queries_path+'/*.png'))
        print(len(queries))

        # hacer las sequencias**************
        fpath, fname = os.path.split(FLAGS.video_path )
        seq_path = fpath+'/seq_videos'
        if not os.path.isfile(seq_path):
            os.system('mkdir '+ seq_path)

        carpet = 0
        cont_seq=0
        n_carpet=''
        cap = cv2.VideoCapture(FLAGS.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        tam_seq = 30 * fps # sequencias de 30 segundos
        
        while(True):
            ret, frame = cap.read()
            if ret == True:
                height, width, layers = frame.shape
                size = (width,height)
                #print('cont seq: ',cont_seq)
                #print('tam seq: ',tam_seq)
                #print('carpet: ',carpet)
                if(cont_seq <= tam_seq ):
                    if(cont_seq==tam_seq):
                        cont_seq = 0
                        carpet = carpet+1
                    
                    if(cont_seq==0):
                        ncarpet = '{0:06}'.format(getFrameNumber(carpet))                    
                        os.system('mkdir '+ seq_path+'/'+ncarpet)
                        pathOut = seq_path+'/'+ncarpet+'/sub_video.avi'
                        out = cv2.VideoWriter(pathOut , cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                    out.write(frame)
                    cont_seq = cont_seq + 1
            else:
                break
        #FIN DE HACER SEQUENCIAS**********************************************

        seq_videos = sorted(glob.glob(seq_path+'/*'))
        
        print(queries)
        print(seq_videos)
        
        for v in seq_videos:
            ############# FLAGS
            #fpath, fname = os.path.split(v)
            fpath_cropps = v + '/cropps'
            if not os.path.isfile(fpath_cropps):
                os.system('mkdir '+ fpath_cropps)
            #tf.flags.DEFINE_string('out_cropps_path', fpath_cropps ,'out cropps path to gallery')

            topN = 10

            
            #tf.flags.DEFINE_string('out_reid_path', fpath_reid ,'out timestamps and topN of ReID')
            ############### FIN FLAGS
            cropper.personCropping( v+'/sub_video.avi', fpath_cropps)

            for q in queries:
                qpath, qname = os.path.split(q)
                temp_qname = qname
                qname = qname.split('.')
                
                fpath_reid = v + '/out_reid_'+ qname[0]
                if not os.path.isfile(fpath_reid):
                    os.system('mkdir '+ fpath_reid)
                ## guardo la query
                temp_query = cv2.imread(q)

                cv2.imwrite(fpath_reid+'/'+temp_qname ,temp_query)
                ##
                fpath_reid_out = fpath_reid +'/top'+str(topN)
                if not os.path.isfile(fpath_reid_out):
                    os.system('mkdir '+ fpath_reid_out)
                reidentifier.PersonReIdentification(q ,fpath_cropps, fpath_reid_out, topN)
                
    if FLAGS.mode == 'val':# data_dir, direccion de Datset Y PROCESAR
    #varias querys, varios videos > 30 seg, paras sacar sequencias
        print('val')
        
        testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
        testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))
        #in carpet test, we found , video.avi and queries /*.png
        
        print(testAB)
        print(testBA)
        for carpet_test_path in testAB:
            personReID_RealTest(carpet_test_path , carpet_test_path + '/video_in.avi')
        
        for carpet_test_path in testBA:
            personReID_RealTest(carpet_test_path , carpet_test_path + '/video_in.avi')
    if FLAGS.mode == 'metrics':

        testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
        testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))
        #GENERANDO LISTA DE QUERYS 
        
        
        dict_values_AB = {}# aqui estaran dos los valores y resultados
        dict_values_BA = {}
        
        #
        matrix_truth_AB = []
        matrix_predict_AB = []
        #A->B
        for carpet_seq in testAB:
            #querys names
            for query_path in sorted(glob.glob(carpet_seq + '/*.png')):
                fpath, fname = os.path.split(query_path)
                fname = fname.split('_')[1]
                fname = fname.split('.')[0]
                
                #print(fpath)
                dict_temp = {}
                for seq_path in sorted(glob.glob(fpath + '/seq_videos/*')):
                    seq_path_temp, seq_name = os.path.split(seq_path)
                    #print('seq: ', seq_name)
                    dict_temp[int(seq_name)] = 0

                dict_values_AB[int(fname)] = dict_temp

            ##
            try:
                ground_truth = pd.read_csv(carpet_seq+'/ground_truth.csv', header=None)
            except pd.errors.EmptyDataError:
                print('el archivo CSV GROUND TRUTH esta vacio\n')
            
            try:
                predict = pd.read_csv(carpet_seq+'/prediction.csv', header=None)
            except pd.errors.EmptyDataError:
                print('el archivo CSV PREDICT esta vacio\n')

            #print(carpet_seq+'\n ,prediction: \n',  predict.to_numpy())
            matrix_truth_AB.append( [ tuple(e) for e in ground_truth.to_numpy() ])# convierto a una matrix de lista de tuplas
            #print(carpet_seq+'\n ,truth: \n', ground_truth.to_numpy())
            matrix_predict_AB.append( [ tuple(e) for e in predict.to_numpy() ])
        
        #B->A
        matrix_truth_BA = []
        matrix_predict_BA = []

        for carpet_seq in testBA:
            #querys names and Dictionary
            for query_path in sorted(glob.glob(carpet_seq + '/*.png')):
                fpath, fname = os.path.split(query_path)
                fname = fname.split('_')[1]
                fname = fname.split('.')[0]
                
                #print(fpath)
                dict_temp = {}
                for seq_path in sorted(glob.glob(fpath + '/seq_videos/*')):
                    seq_path_temp, seq_name = os.path.split(seq_path)
                    #print('seq: ', seq_name)
                    dict_temp[int(seq_name)] = 0

                dict_values_BA[int(fname)] = dict_temp

            ##
            try:
                ground_truth = pd.read_csv(carpet_seq+'/ground_truth.csv', header=None)
            except pd.errors.EmptyDataError:
                print('el archivo CSV GROUND TRUTH esta vacio\n')
            
            try:
                predict = pd.read_csv(carpet_seq+'/prediction.csv', header=None)
            except pd.errors.EmptyDataError:
                print('el archivo CSV PREDICT esta vacio\n')
            
            #print(carpet_seq+'\n ,prediction: \n',  predict.to_numpy())
            matrix_truth_BA.append( [ tuple(e) for e in ground_truth.to_numpy() ])# convierto a una matrix de lista de tuplas
            #print(carpet_seq+'\n ,truth: \n', ground_truth.to_numpy())
            matrix_predict_BA.append( [ tuple(e) for e in predict.to_numpy() ])
        
        print('dictionary of values ab: \n', dict_values_AB)
        print('dictionary of values ba: \n', dict_values_BA)
        
        tp1, tn1, fp1, fn1 = calculate_relations(dict_values_AB, matrix_truth_AB, matrix_predict_AB)
        
        tp2, tn2, fp2, fn2 = calculate_relations(dict_values_BA , matrix_truth_BA, matrix_predict_BA)
        
        tpT = tp1 + tp2 
        tnT = tn1 + tn2
        fpT = fp1 + fp2
        fnT = fn1+ fn2
        print('metrics: \n' ,'TP: ' ,tpT,'\n TN :', tnT,'\n FP: ', fpT,'\n FN:' ,fnT)
        acc = (tpT + tnT) / (tpT + tnT + fpT + fnT)
        precision = tpT /(tpT + fpT)
        recall = tpT / (tpT + fnT)
        f1 = (precision + recall) / 2

        print('acc: ', acc)
        print('precision: ', precision)
        print('recall: ', recall)
        print('f1: ', f1) 
        
        #print('queries all: ', )
        #print('queries all: ', )

    if FLAGS.mode == 'demo':#un video(tomar querys), un video > 30 seg, paras sacar sequencias
        print('demo')


if __name__ == '__main__':
    tf.app.run()