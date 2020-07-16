from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import os, glob
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

model = VGG16(weights='imagenet', include_top=False,pooling="max")
model.summary()


def generate_features_to_disk(folder_path):
    #folder_path = '/home/josemiki/vision/cropssmt/crops/'
    matrix_features_images=[]
    for infolder in sorted(glob.glob( os.path.join(folder_path, '*'))):
        list_features_per_image=[]
        #list_name_image=[]
        for infile in sorted(glob.glob( os.path.join(infolder, '*.png'))):
        #    list_name_image.append(infile)
            img = image.load_img(infile, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            vgg16_feature=model.predict(img_data)
            vgg16_feature_np = np.array(vgg16_feature)
            list_features_per_image.append(vgg16_feature_np.flatten())
            
        vgg16_feature_list_np = np.array(list_features_per_image)
        with open(infolder+"/features.txt", "wb") as fp:
            pickle.dump(list_features_per_image, fp)

def load_features_from_disk(folder_path):
    for infolder in sorted(glob.glob( os.path.join(folder_path, '*'))):
        list_name_image=[]
        for infile in sorted(glob.glob( os.path.join(infolder, '*.png'))):
            list_name_image.append(infile)
        for infile in sorted(glob.glob( os.path.join(infolder, '*.txt'))):
            with open(infile, "rb") as fp: 
                vgg16_feature_list_np = pickle.load(fp)
            nn=3
            kmeans = KMeans(n_clusters=1, random_state=0).fit(vgg16_feature_list_np)
            neigh = NearestNeighbors(n_neighbors=nn)
            neigh.fit(vgg16_feature_list_np)
            neight_dst,neight_in=neigh.kneighbors(kmeans.cluster_centers_)
            neight_in=neight_in.flatten()
            fig=plt.figure(figsize=(8, 8))
            for x in range(nn):
                print("pos: ",neight_in[x],"image: ",list_name_image[neight_in[x]])
                img = image.load_img(list_name_image[neight_in[x]])
                fig.add_subplot(1, 3, x+1)
                plt.imshow(img)
            plt.show()