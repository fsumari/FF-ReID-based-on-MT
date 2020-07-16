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
#from pYOLO import yolo_cropper
#from pReID import cuhk03_dataset
#from pReID import personReID
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
tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')#TRAIN REID
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')#TRAIN REID
tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')#MODELO REID

tf.flags.DEFINE_float('learning_rate', '0.01', '')#TRAIN REID
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
tf.flags.DEFINE_string('data_dir', '../data/dataReal_pt1', 'path to dataset')#DATASET


#**2***********FLAGS NEW REID***********

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

def preprocess(images, is_train):
    def train():
        split = tf.split(images, [1, 1])
        shape = [1 for _ in range(split[0].get_shape()[1])]
        for i in range(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
            split[i] = tf.split(split[i], shape)
            for j in range(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3, 3])
                split[i][j] = tf.random_crop(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.random_flip_left_right(split[i][j])
                split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    def val():
        split = tf.split(images, [1, 1])
        shape = [1 for _ in range(split[0].get_shape()[1])]
        for i in range(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT, IMAGE_WIDTH])
            split[i] = tf.split(split[i], shape)
            for j in range(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    return tf.cond(is_train, train, val)

def network(images1, images2, weight_decay):
    with tf.variable_scope('network'):
        # Tied Convolution
        conv1_1 = tf.layers.conv2d(images1, 20, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_1')
        pool1_1 = tf.layers.max_pooling2d(conv1_1, [2, 2], [2, 2], name='pool1_1')
        conv1_2 = tf.layers.conv2d(pool1_1, 25, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_2')
        pool1_2 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name='pool1_2')
        conv2_1 = tf.layers.conv2d(images2, 20, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_1')
        pool2_1 = tf.layers.max_pooling2d(conv2_1, [2, 2], [2, 2], name='pool2_1')
        conv2_2 = tf.layers.conv2d(pool2_1, 25, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_2')
        pool2_2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name='pool2_2')

        # Cross-Input Neighborhood Differences
        trans = tf.transpose(pool1_2, [0, 3, 1, 2])
        shape = trans.get_shape().as_list()
        m1s = tf.ones([shape[0], shape[1], shape[2], shape[3], 5, 5])
        reshape = tf.reshape(trans, [shape[0], shape[1], shape[2], shape[3], 1, 1])
        f = tf.multiply(reshape, m1s)

        trans = tf.transpose(pool2_2, [0, 3, 1, 2])
        reshape = tf.reshape(trans, [1, shape[0], shape[1], shape[2], shape[3]])
        g = []
        pad = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
        for i in range(shape[2]):
            for j in range(shape[3]):
                g.append(pad[:,:,:,i:i+5,j:j+5])

        concat = tf.concat(g, axis=0)
        reshape = tf.reshape(concat, [shape[2], shape[3], shape[0], shape[1], 5, 5])
        g = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
        reshape1 = tf.reshape(tf.subtract(f, g), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
        reshape2 = tf.reshape(tf.subtract(g, f), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
        k1 = tf.nn.relu(tf.transpose(reshape1, [0, 2, 3, 1]), name='k1')
        k2 = tf.nn.relu(tf.transpose(reshape2, [0, 2, 3, 1]), name='k2')

        # Patch Summary Features
        l1 = tf.layers.conv2d(k1, 25, [5, 5], (5, 5), activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l1')
        l2 = tf.layers.conv2d(k2, 25, [5, 5], (5, 5), activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l2')

        # Across-Patch Features
        m1 = tf.layers.conv2d(l1, 25, [3, 3], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m1')
        pool_m1 = tf.layers.max_pooling2d(m1, [2, 2], [2, 2], padding='same', name='pool_m1')
        m2 = tf.layers.conv2d(l2, 25, [3, 3], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m2')
        pool_m2 = tf.layers.max_pooling2d(m2, [2, 2], [2, 2], padding='same', name='pool_m2')

        # Higher-Order Relationships
        concat = tf.concat([pool_m1, pool_m2], axis=3)
        reshape = tf.reshape(concat, [FLAGS.batch_size, -1])
        fc1 = tf.layers.dense(reshape, 500, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 2, name='fc2')

        return fc2

def personReIdentification(sess, image1, cropps_path, out_reid_path):
    files = sorted(glob.glob(cropps_path + '/*.png'))
    print('cropps files: ',len(files))

    image1 = cv2.imread(image1)
    image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image1)
    plt.show()

    start = time.time()
    image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
    
    list_all = []
    for x in files:
        image2 = cv2.imread(x)
        image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        test_images = np.array([image1, image2])
        feed_dict = {images: test_images, is_train: False}
        prediction = sess.run(inference, feed_dict=feed_dict)
        
        if bool(not np.argmax(prediction[0])):
            tupl = (x, prediction[0][0], prediction[0][1])
            list_all.append(tupl)
    
    list_all.sort(key = sortSecond , reverse = True)
    
    end = time.time()
    print("Time in seconds: ")
    print(end - start)

    #print (list_all)
    print ("size list predict: ", len(list_all))
    
    i = 0# para ordenar, de acuerdo al acierto

    list_reid_coords = []
    list_score = []

    #out_reid_path = "/home/oliver/Documentos/RealSystemReID/RealPersonReID/data/seq1/rpta/"
    for e in list_all:
        temp_img = cv2.imread(e[0])
        # estoy leyendo el crop de disco ... para luego escribir en otro lado
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        
        fpath, fname = os.path.split(e[0])
        if (i > 10 ):
            break
        #plt.imshow(temp_img)
        #plt.show()
        #cv2.namedWindow('Person-ReID', cv2.WINDOW_NORMAL)                
        #cv2.imshow('Person-ReID', temp_img)

        #estoy escribiendo ..
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)                
        cv2.imwrite(out_reid_path+str(i+1)+'_'+fname, temp_img)
        #cv2.waitKey(1)
        path_f, name_f = os.path.split(e[0])
        splits_coords = name_f.rsplit('_')
        #print("coord: ",splits_coords)
        last_coord = splits_coords[5].rsplit('.')
        
        i = i +1

        list_reid_coords.append(( int(splits_coords[1]), splits_coords[2], splits_coords[3], splits_coords[4], last_coord[0]))
        #pathi, nameimage = e[0]
        list_score.append((name_f, e[1], e[2]))
        print (i, e[0]," - ", e[1], " - ", e[2])
    #list_reid_coords.sort(key = sortFirst)
    ## sort the coords for num of frame
    print (list_reid_coords)
    ## escribo un csv
    
    df = pd.DataFrame(np.array(list_reid_coords))
    df.to_csv(out_reid_path+"coords.csv", header = False)
    df = pd.DataFrame(np.array(list_score))
    df.to_csv(out_reid_path+"scores.csv", header = False)

def main(argv=None):
    FLAGS.batch_size = 1

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    global images 
    images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 2], name='labels')
    global is_train 
    is_train = tf.placeholder(tf.bool, name='is_train')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weight_decay = 0.0005
    tarin_num_id = 0
    val_num_id = 0

    #if FLAGS.mode == 'train':
    #    tarin_num_id = cuhk03_dataset.get_num_id(FLAGS.data_dir, 'train')
    #elif FLAGS.mode == 'val':
    #    val_num_id = cuhk03_dataset.get_num_id(FLAGS.data_dir, 'val')
    
    images1, images2 = preprocess(images, is_train)

    print('=======================Build Network=======================')
    logits = network(images1, images2, weight_decay)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    global inference
    inference = tf.nn.softmax(logits)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    train = optimizer.minimize(loss, global_step=global_step)
    lr = FLAGS.learning_rate

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('==================================Restore model==================================')
            saver.restore(sess, ckpt.model_checkpoint_path)


        #if FLAGS.mode != 'metrics':
            #global cropper
            #cropper = yolo_cropper.YOLOcropper()#objeto que recorta
            #global reidentifier 
            #reidentifier = personReID.personReIdentifier()#objeto que hace el reid
            

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
            #cropper.personCropping(FLAGS.video_path, FLAGS.out_cropps_path)
            personReIdentification(sess, FLAGS.query_path, FLAGS.out_cropps_path, FLAGS.out_reid_path)
            #reidentifier.PersonReIdentification(FLAGS.query_path, FLAGS.out_cropps_path, FLAGS.out_reid_path, topN)
            
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
            #DEBERIA DE HABER UNA PARTE DONDE GENERE PREDICTS.CSV
        if FLAGS.mode == 'new_predict':#genero los predict a partir de los resultados
            testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
            testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))

        if FLAGS.mode == 'metrics':#calculo los resultados, acc, precision ...etc

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
                ground_truth = pd.DataFrame()
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