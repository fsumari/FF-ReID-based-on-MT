import cv2
import time
import numpy as np
#import core.utils as utils
import tensorflow as tf
from PIL import Image
from pYOLO.yolo import YOLO
from pTrack.kdtree_tracker import KDTreeTracker
from scipy.spatial import distance
import os

global ID
#ID = 1
cont_frame = 0

### creating tracking
#tracker = DistanceTracker()
tracker = KDTreeTracker()
#tracker = cv2.TrackerKCF_create()

fps_temp = None
#track_t = cv2.TrackerMOSSE_create()
trackers = []
#trackers.append(track_t)
#out = None
video_path = "./data/seq1/video_in.avi"
detector = YOLO()
#with tf.Session(graph=graph) as sess:
vid = cv2.VideoCapture(video_path)
fps = vid.get(cv2.CAP_PROP_FPS)
pathOut, Outname = os.path.split(video_path)
#skip_frame = 5 # OD cada 30 frames
#print('skip frames: ', skip_frame)
#id_t = 0
curr_time = time.time()

while True:
    return_value, frame = vid.read()
    
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copy_frame=frame.copy()
        image = Image.fromarray(frame)
        if(cont_frame == 0):
            height, width, layers = frame.shape
            size = (width,height)
            out = cv2.VideoWriter(pathOut+'/OD+MT_out.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    else:
        #raise ValueError("No image!")
        print('No image!')
        out.release()
        break

    ## ************* PERSON DETECTION
    #if (cont_frame % skip_frame == 0):
        
    prev_time = time.time()
    person_bboxes =  detector.pImageDetection(frame)
    
    ###
    ## BEGIN COUNT AND DRAW
    #image = frame#### 
    bbox_color = (244, 52, 92)
    bbox_thick = 2
    
    #pbboxes = []
    
    # *********** DRAW THE DETECTIONS
    '''
    for i, bbox in enumerate(person_bboxes):
        #coor = np.array(bbox[:4], dtype=np.int32)
        coords_ul = (int(bbox[0]), int(bbox[1]))
        coords_br = (int(bbox[0] + bbox[2]), int(bbox[1] +bbox[3]))
        #draw detection
        cv2.rectangle(frame , coords_ul, coords_br, (0,0,255), bbox_thick)
        #print(bbox)
        pbboxes.append((np.array(bbox).astype('int')))
    '''
    person_bboxes = np.array(person_bboxes)
    try:
        tracks, pbboxs = tracker.update_matchs(person_bboxes)
        print('pbox: ',pbboxs)
        print('tracks: ',tracks)

            ## *************** PERSON MATCHING TRACKING
        for (ID, value) in tracks.items():
            #
            coords_ul = (int(pbboxs[ID][0]), int(pbboxs[ID][1]))
            coords_br = (int(pbboxs[ID][0] + pbboxs[ID][2]), int(pbboxs[ID][1] +pbboxs[ID][3]))
            cv2.rectangle(frame , coords_ul, coords_br, (0,255,0), bbox_thick)
            center = value # center
            cv2.circle(frame , (center[0], center[1]) , 2 , bbox_color, 4)
            #*** DRAWING TRACKs
            id_str = "{}".format(ID)
                        
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 1
            org = (center[0] , pbboxs[ID][1])
            thickness = 1
            color = (0, 255, 0)
            frame = cv2.putText(frame, id_str\
                    , org, font, fontScale, color, thickness, cv2.LINE_AA)
            #*** CROPPING
            crop_img = copy_frame[ coords_ul[1]: coords_br[1], \
                                    coords_ul[0]: coords_br[0]]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            ncarpet = '{0:06}'.format(ID)
            fpath_cropps = pathOut+'/cropp_ids'
            if not os.path.isfile(fpath_cropps):
                os.system('mkdir '+ fpath_cropps)
            os.system('mkdir '+ fpath_cropps+'/'+ncarpet)
            name_crop = fpath_cropps + '/'+ ncarpet+'/'+ str(cont_frame) +'_'+ id_str+'.png'
            cv2.imwrite(name_crop, crop_img)
            
            #***
    #***********************************
    except:
        print("Não tem detecçoes novas ..")
    

    
    cont_frame = cont_frame+1
    #WRITE VIDEO
    frame_v = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_v)

    
    #exec_time = curr_time - prev_time
    result = np.asarray(frame )
    #info = "time: %.2f ms" %(1000*exec_time)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
    
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        vid.release()
        out.release()
        break


