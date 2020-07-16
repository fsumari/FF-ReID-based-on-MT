import cv2
import time
import numpy as np
#import core.utils as utils
import tensorflow as tf
from PIL import Image
from pYOLO.yolo import YOLO
#from pTrack.kdtree_tracker import KDTreeTracker
from pTrack.knn_tracker import KNNTracker

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from scipy.spatial import distance
import os

class MTcropper(object):
    def __init__(self, num_frame):
        #self.model = VGG16(weights='imagenet', include_top=False, pooling='max')
        #self.model.summary()
        self.detector = YOLO()
        self.tracker = KNNTracker()
        self.init_frame = num_frame
        self.model = VGG16(weights='imagenet', include_top=False, pooling='max')
        self.model.summary()
    def personCropping(self, video_path, cropps_out_path, write_video = False):
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        pathOut, Outname = os.path.split(video_path)
        curr_time = time.time()
        cont_frame = self.init_frame
        while True:
            return_value, frame = vid.read()
            
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                copy_frame=frame.copy()
                image = Image.fromarray(frame)
                if(cont_frame == self.init_frame and write_video == True):
                    height, width, layers = frame.shape
                    size = (width,height)
                    out = cv2.VideoWriter(pathOut+'/video_out.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
            else:
                #raise ValueError("No image!")
                self.tracker.delete_all(cropps_out_path)
                print('No image!')
                if(write_video):
                    out.release()
                break

            ## ************* PERSON DETECTION                
            prev_time = time.time()
            person_bboxes =  self.detector.pImageDetection(frame)
            features = np.ndarray((len(person_bboxes), 512))# 512 features per DETECTION
            ###
            bbox_color = (244, 52, 92)
            bbox_thick = 2
            
            # *********** GET FEATURES AND/OR DRAW THE DETECTIONS
            for i, bbox in enumerate(person_bboxes):
                #coor = np.array(bbox[:4], dtype=np.int32)
                x_ul, y_ul = int(bbox[0]), int(bbox[1])
                x_br, y_br = int(bbox[0] + bbox[2]) , int(bbox[1] +bbox[3])
                # DRAW detection
                #cv2.rectangle(frame , (x_ul, y_ul), (x_br, y_br), (0,255,0), bbox_thick)
                #print(bbox)
                # GET FEATURES
                cropp_img = copy_frame[y_ul:y_br ,x_ul:x_br]
                cropp_img = np.expand_dims(cropp_img, axis=0)
                cropp_img = preprocess_input(cropp_img)
                print('crop img shape: ', cropp_img.shape)
                pred = self.model.predict(cropp_img)
                features[i] = pred.flatten()
                #pbboxes.append((np.array(bbox).astype('int')))
            
            person_bboxes = np.array(person_bboxes)
            features = np.array(features)
            print('antes try pbox: ', np.shape(person_bboxes))
            print('antes try pbox: ', np.shape(features))
            try:
                tracks, pbboxs = self.tracker.update_matchs(person_bboxes, features, cropps_out_path)
                
                    ## *************** PERSON MATCHING TRACKING
                for (ID, value) in tracks.items():
                    #
                    print('try pbox: ', np.shape(pbboxs[ID]))
                    print('try tracks: ', np.shape(tracks[ID]))
                
                    coords_ul = (int(pbboxs[ID][0]), int(pbboxs[ID][1]))
                    coords_br = (int(pbboxs[ID][0] + pbboxs[ID][2]), int(pbboxs[ID][1] +pbboxs[ID][3]))
                    cv2.rectangle(frame , coords_ul, coords_br, (0,255,0), bbox_thick)
                    #center = value # center
                    #cv2.circle(frame , (center[0], center[1]) , 2 , bbox_color, 4)
                    #*** DRAWING TRACKs
                    id_str = "{}".format(ID)
                                
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    fontScale = 1
                    #org = (center[0] , pbboxs[ID][1])
                    org = (int(pbboxs[ID][0]), int(pbboxs[ID][1]))
                    thickness = 1
                    color = (0, 255, 0)
                    print('put text')
                    frame = cv2.putText(frame, id_str\
                            , org, font, fontScale, color, thickness, cv2.LINE_AA)
                    #*** CROPPING
                    print('cropping')
                    crop_img = copy_frame[ coords_ul[1]: coords_br[1], \
                                            coords_ul[0]: coords_br[0]]
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                    ncarpet = '{0:06}'.format(ID)
                    nframe = '{0:06}'.format(cont_frame)

                    os.system('mkdir '+ cropps_out_path+'/'+ncarpet)
                    name_crop = cropps_out_path + '/'+ ncarpet+'/'+ nframe +'_'+ id_str+'.png'
                    cv2.imwrite(name_crop, crop_img)
            #***********************************
            except:
                print("Não tem detecçoes novas ..")
            cont_frame = cont_frame+1
            #WRITE VIDEO
            frame_v = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if(write_video):
                out.write(frame_v)
            #exec_time = curr_time - prev_time
            result = np.asarray(frame )
            #info = "time: %.2f ms" %(1000*exec_time)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
            
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                vid.release()
                if(write_video):
                    out.release()
                break
        
if __name__ == '__main__':
    
    video_in = 'data/seq1/video_in.avi'
    p_out, out_name = os.path.split(video_in)

    fpath_cropps = p_out+'/cropp_ids'
    if not os.path.isfile(fpath_cropps):
        os.system('mkdir '+ fpath_cropps)
    cropper = MTcropper(1)
    cropper.personCropping(video_in, fpath_cropps)
                    

