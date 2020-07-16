
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from sklearn.neighbors import KDTree

class KDTreeTracker():
	def __init__(self, max_miss=20):
		# sigente ID, o primero já pe inicializado
		self.sgte_ID = 1
		self.tracks = OrderedDict() # dict de tracks
		self.pbboxs = OrderedDict()
		self.tracks_missed = OrderedDict() #tracks que foram perdidos
		self.max_miss = max_miss # cantidade de frame que um track pode tar perdido

	def add_track(self, center, pbbox):
		#para adicionar nos dicts
		self.tracks[self.sgte_ID] = center
		self.pbboxs[self.sgte_ID] = pbbox
		self.tracks_missed[self.sgte_ID] = 0
		self.sgte_ID += 1

	def del_track(self, track_ID):
		del self.tracks[track_ID]
		del self.pbboxs[track_ID]
		del self.tracks_missed[track_ID]
	
	def update_matchs(self, bboxs):
	# os bboxs, sao (x,y, w,h)
		print('***************')
		#caso base, cuando no tengo detecciones
		if (bboxs.shape[0] == 0):
			for track_ID in list(self.tracks_missed.keys()):
				self.tracks_missed[track_ID] += 1
				if self.tracks_missed[track_ID] > self.max_miss:
					self.del_track(track_ID)

			return self.tracks

		detects_centers = np.zeros((bboxs.shape[0], 2), dtype="int")
		detects_bboxs = np.zeros(bboxs.shape, dtype="int")

		for (i, ( x ,y ,w, h)) in enumerate(bboxs):
			center_x = int(x) + int(w/2.0)
			center_y = int(y) + int(h/2.0)
			detects_centers[i] = (center_x, center_y)
			detects_bboxs[i] = [x,y,w,h]
		if len(self.tracks) == 0:
			for i in range(0, len(detects_centers)):
				self.add_track(detects_centers[i], detects_bboxs[i])
		else:
			
			track_IDs = list(self.tracks.keys())
			track_centers = list(self.tracks.values())
			print('tracks center: \n',track_centers)
			print('detects center: \n',detects_centers)
			
			tree = KDTree(np.array(track_centers)) #
			
			#match_tracks = set()
			#match_detects = set()
			match_tracks = []
			match_detects = []
			# **** knn with kdtree for centers .. 
			#depois gerar para descriptors
			for (pos ,center) in enumerate(detects_centers):
				dist, indx = tree.query([center], k=1)
				print('ind: ', indx)
				print('center: ', center)
				print('match tracks: \n ', match_tracks)
				print('match detetecs: \n ', match_detects)
				if ((pos in match_detects) or (indx[0][0] in match_tracks)):
					print('entro IF')
					continue
				#match_tracks.add(indx[0][0])
				#match_detects.add(pos)
				track_ID = track_IDs[indx[0][0]]
				self.tracks[track_ID] = detects_centers[pos]
				self.pbboxs[track_ID] = detects_bboxs[pos]
				self.tracks_missed[track_ID] = 0
				#para saber
				match_tracks.append(indx[0][0])
				match_detects.append(pos)
			conj_track = np.arange(0, len(track_centers))
			conj_detect = np.arange(0, len(detects_centers))
			print('conjuntos: ',conj_detect, conj_track)
			unmatch_tracks = np.setdiff1d(conj_track, match_tracks)
			unmatch_detects = np.setdiff1d(conj_detect, match_detects)
			print('unmatch tracks: \n ', unmatch_tracks)
			print('unmatch detects: \n ', unmatch_detects)
			
			#precorrer os unmatchs
			if (len(track_centers) >= len(detects_centers)):
				
				for idx in unmatch_tracks:
					print('track ids: ', track_IDs[idx])
					track_ID = track_IDs[idx]
					self.tracks_missed[track_ID] += 1
					if (self.tracks_missed[track_ID] > self.max_miss):
						self.del_track(track_ID)
			else:
				for idx in unmatch_detects:
					self.add_track(detects_centers[idx], detects_bboxs[idx])
		print('pbox dentro: ',self.pbboxs)
		return self.tracks, self.pbboxs
