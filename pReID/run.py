import tensorflow as tf
import numpy as np
import cv2
import cuhk03_dataset
import time
import matplotlib.pyplot as plt
import glob
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test, data')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')
tf.flags.DEFINE_string('path_test', '', 'Images path to compare')

IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160

def sortSecond(val):
    return val[1]

def sortFirst(val):
    return val[0]

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

def main(argv=None):
   

    if FLAGS.mode == 'test':
        FLAGS.batch_size = 1

    if FLAGS.mode == 'data':
        FLAGS.batch_size = 1

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 2], name='labels')
    is_train = tf.placeholder(tf.bool, name='is_train')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weight_decay = 0.0005
    tarin_num_id = 0
    val_num_id = 0

    if FLAGS.mode == 'train':
        tarin_num_id = cuhk03_dataset.get_num_id(FLAGS.data_dir, 'train')
    elif FLAGS.mode == 'val':
        val_num_id = cuhk03_dataset.get_num_id(FLAGS.data_dir, 'val')
    
    images1, images2 = preprocess(images, is_train)

    print('=======================Build Network=======================')
    logits = network(images1, images2, weight_decay)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
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

        if FLAGS.mode == 'train':
            step = sess.run(global_step)
            for i in range(step, FLAGS.max_steps + 1):
                batch_images, batch_labels = cuhk03_dataset.read_data(FLAGS.data_dir, 'train', tarin_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
                feed_dict = {learning_rate: lr, images: batch_images,
                    labels: batch_labels, is_train: True}
                sess.run(train, feed_dict=feed_dict)
                train_loss = sess.run(loss, feed_dict=feed_dict)
                print('Step: %d, Learning rate: %f, Train loss: %f' % (i, lr, train_loss))

                lr = FLAGS.learning_rate * ((0.0001 * i + 1) ** -0.75)
                if i % 1000 == 0:
                    saver.save(sess, FLAGS.logs_dir + 'model.ckpt', i)
        elif FLAGS.mode == 'val':
            total = 0.
            for _ in range(10):
                batch_images, batch_labels = cuhk03_dataset.read_data(FLAGS.data_dir, 'val', val_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
                feed_dict = {images: batch_images, labels: batch_labels, is_train: False}
                prediction = sess.run(inference, feed_dict=feed_dict)
                prediction = np.argmax(prediction, axis=1)
                label = np.argmax(batch_labels, axis=1)

                for i in range(len(prediction)):
                    if prediction[i] == label[i]:
                        total += 1
            print('Accuracy: %f' % (total / (FLAGS.batch_size * 10)))

            '''
            for i in range(len(prediction)):
                print('Prediction: %s, Label: %s' % (prediction[i] == 0, labels[i] == 0))
                image1 = cv2.cvtColor(batch_images[0][i], cv2.COLOR_RGB2BGR)
                image2 = cv2.cvtColor(batch_images[1][i], cv2.COLOR_RGB2BGR)
                image = np.concatenate((image1, image2), axis=1)
                cv2.imshow('image', image)
                key = cv2.waitKey(0)
                if key == 1048603:  # ESC key
                    break
            '''
        elif FLAGS.mode == 'test':
            image1 = cv2.imread(FLAGS.image1)
            image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            
            
            image2 = cv2.imread(FLAGS.image2)
            image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            
            f = plt.figure()
            f.add_subplot(1,2, 1)
            plt.imshow(image1)
            f.add_subplot(1,2, 2)
            plt.imshow(image2)
            plt.show()
            print("===============================Show Images==================================================")
            
            start = time.time()

            image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            
            
            
            test_images = np.array([image1, image2])
            test_images2 = np.array([image2, image1])
            
            feed_dict = {images: test_images, is_train: False}
            feed_dict2 = {images: test_images2, is_train: False}
            
            #print(feed_dict)
            
            prediction = sess.run(inference, feed_dict=feed_dict)
            prediction2 = sess.run(inference, feed_dict=feed_dict2)
            
            print("=======================Prediction1=======================")
            print(prediction)
            print(bool(not np.argmax(prediction[0])))
            #print(prediction[0])
            print("=======================Prediction2=======================")
            print(prediction2)
            print(bool(not np.argmax(prediction2[0])))
            
            end = time.time()
            print("Time in seconds: ")
            print(end - start)
        
        elif FLAGS.mode == 'data':
            print("path_test:",FLAGS.path_test)

            files = sorted(glob.glob('/home/oliver/Documentos/person-reid/video3_4/*.png'))
            print(len(files))

            image1 = cv2.imread(FLAGS.image1)
            image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            
            plt.imshow(image1)
            plt.show()
            
            '''
            image2 = cv2.imread(FLAGS.image2)
            image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            f = plt.figure()
            f.add_subplot(1,2, 1)
            plt.imshow(image1)
            f.add_subplot(1,2, 2)
            plt.imshow(image2)
            plt.show()
            
            print("===============================Show Images==================================================")
            '''
            start = time.time()
            image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            #image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            #list_pred=[]
            #list_bool=[]
            list_all = []
            for x in files:
                image2 = cv2.imread(x)
                image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
                test_images = np.array([image1, image2])
                feed_dict = {images: test_images, is_train: False}
                prediction = sess.run(inference, feed_dict=feed_dict)
                #print(bool(not np.argmax(prediction[0])))
                #list_bool.append(bool(not np.argmax(prediction[0])))
                #list_pred.append(prediction[0])
                if bool(not np.argmax(prediction[0])):
                    tupl = (x, prediction[0][0], prediction[0][1])
                    list_all.append(tupl)
            list_all.sort(key = sortSecond , reverse = True)
            
            end = time.time()
            print("Time in seconds: ")
            print(end - start)

            #print (list_all)
            print ("size list: ", len(list_all))
            ####3
            #cv2.namedWindow('Person-ReID', cv2.WINDOW_FULLSCREEN)
            #cv2.resizeWindow('Person-ReID', 480, 320)
            ####
            i = 0
            list_reid = []
            for e in list_all:
                temp_img = cv2.imread(e[0])
                temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                fpath, fname = os.path.split(e[0])
                if (i > 15 ):
                    break
                #plt.imshow(temp_img)
                #plt.show()
                #cv2.namedWindow('Person-ReID', cv2.WINDOW_NORMAL)                
                #cv2.imshow('Person-ReID', temp_img)                
                cv2.imwrite("output_query/"+fname, temp_img)
                #cv2.waitKey(1)
                path_f, name_f = os.path.split(e[0])
                splits_coords = name_f.rsplit('_')
                #print("coord: ",splits_coords)
                list_reid.append(( int(splits_coords[1]), splits_coords[2], splits_coords[3], splits_coords[4], splits_coords[5]))
                i = i +1
                print (i, e[0]," - ", e[1], " - ", e[2])
            list_reid.sort(key = sortFirst)
            ## sort the coords for num of frame
            print (list_reid)

            f_frames = sorted(glob.glob('/home/oliver/Documentos/person-reid/frames/video3/*.png'))
            j = 0
            cv2.namedWindow('Person-ReID', cv2.WINDOW_NORMAL)                
            cv2.resizeWindow('Person-ReID', 640, 480)
            flag_draw = False
            k = 0
            ###PINTO EN LOS FRAMES
            for frame in f_frames:
                imgFrame = cv2.imread(frame , cv2.IMREAD_UNCHANGED)
                frame_p, frame_n = os.path.split(frame)
                temp_f = frame_n.rsplit('.')
                #cv2.imshow('Person-ReID', imgFrame)
                #cv2.waitKey(1)
                #print(int(temp_f[0]))
                if(j < len(list_reid)):
                    if (int(temp_f[0]) == list_reid[j][0]):
                        #pintar como TRUE
                        print (int(temp_f[0]) ,"--entro--",j, " ", list_reid[j])
                        #cv2.polylines(imgFrame , [np.int0([list_reid[j][1], list_reid[j][2], list_reid[j][3], list_reid[j][4]]).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                        #cv2.rectangle(imgFrame,(int(list_reid[j][4]), int(list_reid[j][3])) , (int(list_reid[j][2]),int(list_reid[j][1])), (0, 255, 0), 3)
                        #cv2.rectangle(imgFrame,(int(list_reid[j][3]), int(list_reid[j][4])),(int(list_reid[j][1]), int(list_reid[j][2])), (0, 255, 0), 3)
                        #color = cv2.cvtColor(np.uint8([[[num_random, 128, 200]]]),cv2.COLOR_HSV2RGB).squeeze().tolist()
                        #####################
                        #color = cv2.cvtColor(np.uint8([[[0, 128, 200]]]),cv2.COLOR_HSV2RGB).squeeze().tolist()
                        cv2.rectangle(imgFrame, (int(list_reid[j][3]), int(list_reid[j][1])) , (int(list_reid[j][4]),int(list_reid[j][2])) , (0,255,0), 10)
                        #cv2.imwrite('outReid/'+temp_f[0]+'.png',imgFrame)
                        flag_draw = True
                        k = 0
                        j=j+1
                    #else:
                        #cv2.imwrite('outReid/'+temp_f[0]+'.png',imgFrame)
                    #    cv2.imshow('Person-ReID', imgFrame)
                    #    cv2.waitKey(1)
                #else:
                    #cv2.imshow('Person-ReID', imgFrame)
                    #cv2.waitKey(1)
                    #cv2.imwrite('outReid/'+temp_f[0]+'.png',imgFrame)
                if (flag_draw == True):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(imgFrame,'True',(200,200), font, 4,(0,255,0),4,cv2.LINE_AA)
                    k = k + 1
                    if (k > 15):
                        flag_draw = False
                        k = 0
                
                #cv2.imwrite('outReid/'+temp_f[0]+'.png',imgFrame)    
                cv2.imshow('Person-ReID', imgFrame)
                cv2.waitKey(1)  

            #print(e[0]," , ", e[1], "\n")
            
            #i=0
            #for x in list_bool:
            #    if x==True:
            #        print(files[i],list_pred[i],list_bool[i])
            #    i=i+1

            #test_images = np.array([image1, image2])
            #test_images2 = np.array([image2, image1])
            
            #feed_dict = {images: test_images, is_train: False}
            #feed_dict2 = {images: test_images2, is_train: False}
            
            #print(feed_dict)
            
            #prediction = sess.run(inference, feed_dict=feed_dict)
            #prediction2 = sess.run(inference, feed_dict=feed_dict2)
            
            print("=======================Prediction List=======================")
            #print("Tamaño preds:",len(list_pred))
            #print("Tamaño bools:",len(list_bool))
            #print(list_pred)
            #print(bool(not np.argmax(prediction[0])))
            #print(prediction[0])
            #print("=======================Prediction=======================")
            #print(prediction2)
            #print(bool(not np.argmax(prediction2[0])))
            
            
                
if __name__ == '__main__':
    tf.app.run()
