
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
#from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import pickle

class KNNTracker():
	def __init__(self, max_miss=20):
		# sigente ID, o primero já pe inicializado
		self.sgte_ID = 1
		self.tracks = OrderedDict() # dict de tracks(features) atuales
		self.all_features = OrderedDict() # dict listas de features -> para salvarlos num arquivo 
		self.pbboxs = OrderedDict() # dict de boxes atuales
		self.tracks_missed = OrderedDict() #tracks que foram perdidos
		self.max_miss = max_miss # cantidade de frame que um track pode tar perdido

	def add_track(self, feature, pbbox):
		#para adicionar nos dicts
		self.tracks[self.sgte_ID] = feature
		#try:
		('ADD TRACK --')
		#	self.all_features[self.sgte_ID].append(feature) # no caso que o ID exista
		#except:
		self.all_features[self.sgte_ID] = [feature] # no caso, q é a primera vez
		#self.all_features[self.sgte_ID].append(feature)
		print('DESPUES ADD TRACK --')
		#print('shape all track ', np.shape(self.all_features[self.sgte_ID]))
		
		self.pbboxs[self.sgte_ID] = pbbox
		self.tracks_missed[self.sgte_ID] = 0
		self.sgte_ID += 1

	def del_track(self, track_ID, cropps_path):
		ncarpet = '{0:06}'.format(track_ID)
		print('all fetures', np.shape(self.all_features[track_ID]))
		with open(cropps_path+'/'+ncarpet+'/features.txt', "wb") as fp:
			pickle.dump(self.all_features[track_ID], fp)
		del self.tracks[track_ID]
		del self.pbboxs[track_ID]
		del self.tracks_missed[track_ID]
		del self.all_features[track_ID]
	
	def delete_all(self, cropps_path):
		copy_tracks = self.tracks.copy() 
		for ID in copy_tracks.keys():
			self.del_track(ID, cropps_path)

	def update_matchs(self, detects_bboxs, detects_features, cropps_path):
	# os bboxs, sao (x,y, w,h)
		print('***************')
		#caso base, cuando no tengo detecciones
		if (detects_bboxs.shape[0] == 0):
			print('entro cuando no tengo detecciones')
			for track_ID in list(self.tracks_missed.keys()):
				self.tracks_missed[track_ID] += 1
				if self.tracks_missed[track_ID] > self.max_miss:
					self.del_track(track_ID,cropps_path)

			return self.tracks
		'''
		detects_centers = np.zeros((bboxs.shape[0], 2), dtype="int")
		detects_bboxs = np.zeros(bboxs.shape, dtype="int")

		## novos features como input
		for (i, ( x ,y ,w, h)) in enumerate(bboxs):
			center_x = int(x) + int(w/2.0)
			center_y = int(y) + int(h/2.0)
			detects_centers[i] = (center_x, center_y)
			detects_bboxs[i] = [x,y,w,h]
		'''		
		if len(self.tracks) == 0:
			print('debe ser la primera vez')
			for i in range(0, len(detects_bboxs)):
				self.add_track(detects_features[i], detects_bboxs[i])
		else:
			
			track_IDs = list(self.tracks.keys())
			track_features = list(self.tracks.values())
			#tracks_bboxs = list(self.tracks.values())
			print('tracks features: \n',np.shape(track_features))
			print('detects features: \n',np.shape(detects_features))
			#print(track_features)
			# aqui tou fazendo o KNN
			'''
			vecs = []
			for(i, img in enumerate(tracks_bboxs)):
				pred = model.predict(img_data)
			'''
			knn = NearestNeighbors(metric='cosine', algorithm='brute')
			knn.fit(track_features)
			#tree = KDTree(np.array(track_features)) #
			print('desp de fot de knn')
			#match_tracks = set()
			#match_detects = set()
			match_tracks = []
			match_detects = []
			# **** knn with kdtree for centers .. 
			#depois gerar para descriptors
			for (pos ,feature) in enumerate(detects_features):
				print('antes knn')
				#dist, indx = tree.query([center], k=1)
				dist, indx = knn.kneighbors(feature.reshape(1,-1), n_neighbors=1)
				dist = dist.flatten()
				#indx = indx.flatten() 
				print('dsps knn')
				print('ind: ', indx)
				#print('feature: ', feature)
				print('match tracks: \n ', match_tracks)
				print('match detetecs: \n ', match_detects)
				if ((pos in match_detects) or (indx[0][0] in match_tracks)):
					print('entro IF')
					continue
				#match_tracks.add(indx[0][0])
				#match_detects.add(pos)
				track_ID = track_IDs[indx[0][0]]
				self.tracks[track_ID] = detects_features[pos]
				self.all_features[track_ID].append(detects_features[pos])
				self.pbboxs[track_ID] = detects_bboxs[pos]
				self.tracks_missed[track_ID] = 0
				#para saber
				match_tracks.append(indx[0][0])
				match_detects.append(pos)
			conj_track = np.arange(0, len(track_features))
			conj_detect = np.arange(0, len(detects_features))
			print('conjuntos: ',conj_detect, conj_track)
			unmatch_tracks = np.setdiff1d(conj_track, match_tracks)
			unmatch_detects = np.setdiff1d(conj_detect, match_detects)
			print('unmatch tracks: \n ', unmatch_tracks)
			print('unmatch detects: \n ', unmatch_detects)
			
			#precorrer os unmatchs
			if (len(track_features) >= len(detects_features)):
				
				for idx in unmatch_tracks:
					print('track ids: ', track_IDs[idx])
					track_ID = track_IDs[idx]
					self.tracks_missed[track_ID] += 1
					if (self.tracks_missed[track_ID] > self.max_miss):
						self.del_track(track_ID, cropps_path)
			else:
				for idx in unmatch_detects:
					self.add_track(detects_features[idx], detects_bboxs[idx])
		print('pbox dentro: ',self.pbboxs)
		return self.tracks, self.pbboxs
