from __future__ import print_function, division
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import h5py
from config import args
import time


class NgsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.T_Gnd = scp.loadmat(mat_file)['tracksGndT'] # Added to enable deriving the 'fut' based on the ground truth future frames
        self.t_h = t_h  # 
        self.t_f = t_f  # 
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid
        self.alltime = 0
        self.count = 0

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)  # dataset id
        vehId = self.D[idx, 1].astype(int)  # agent id
        RdAng = self.D[idx, 11]  # Added
        t = self.D[idx, 2]  # frame
        grid = self.D[idx, 14:]  # grid id
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId, RdAng)  # Added RdAng
        refdistance = np.zeros_like(hist[:, 0])
        refdistance = refdistance.reshape(len(refdistance), 1)
        fut = self.getFuture(vehId, t, dsId, RdAng)  # Added RdAng
        va = self.getVA(vehId, t, vehId, dsId, RdAng) # Added RdAng
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId, RdAng)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis)  
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId, RdAng)) # Added RdAng
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([args['lon_length']])
        lon_enc[int(self.D[idx, 13] - 1)] = 1
        lat_enc = np.zeros([args['lat_length']])
        lat_enc[int(self.D[idx, 12] - 1)] = 1

        # hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        return hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 7]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 7]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose() 
            vehTrack = self.T[dsId - 1][vehId - 1].transpose() 
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 8]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 8]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId, RdAng):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:7]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                velX = vehTrack[stpt:enpt:self.d_s, 3]
                velY = vehTrack[stpt:enpt:self.d_s, 4]
                acclX = vehTrack[stpt:enpt:self.d_s, 5]
                acclY = vehTrack[stpt:enpt:self.d_s, 6]

                velCon = self.vecConv(velX, velY, RdAng)
                velCon = np.asarray(velCon).transpose()
                acclCon = self.vecConv(acclX, acclY, RdAng)
                acclCon = np.asarray(acclCon).transpose()

                a = velCon[:, 0]
                b = acclCon[:, 0]

                va = np.column_stack((a,b))

            if len(velX) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return va

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId, RdAng):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose() 
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]
            refPosCon = self.cordConv(refPos[1], refPos[0], RdAng)  # added to convert coordinates to the target vehicle frame

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h) 
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                histOri = vehTrack[stpt:enpt:self.d_s, 1:3]  # -(refPos) #Introduced abs()

                histCon = self.cordConv(histOri[:, 1], histOri[:, 0], RdAng)  # added to convert coordinates to the target vehicle frame
                histCon = np.asarray(histCon).transpose()
                refPosCon = np.asarray(refPosCon)
                hist = np.asarray(histCon)
                for i in range(len(histCon)):
                    if (histCon[i, 1] == 0) and (histCon[i, 0] == 0):
                        hist[i, :] = 0
                    else:
                        hist[i, :] = histCon[i, :] - refPosCon
                # hist = histCon[:,:] - refPosCon
                hist[:, [0, 1]] = hist[:, [1, 0]]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist
            #     hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            # if len(hist) < self.t_h // self.d_s + 1:
            #     return np.empty([0, 2])
            # return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId, RdAng):
        # vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        # refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        # stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        # enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        # fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        vehTrack = self.T_Gnd[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        refPosCon = self.cordConv(refPos[1], refPos[0], RdAng)  # added
        # refPosCon = np.asarray(refPosCon)
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        futOri = vehTrack[stpt:enpt:self.d_s, 1:3]  # -(refPos) #Introduced abs()
        futCon = self.cordConv(futOri[:, 1], futOri[:, 0],
                               RdAng)  # added to convert coordinates to the target vehicle frame
        futCon = np.asarray(futCon).transpose()

        refPosCon = np.asarray(refPosCon)
        fut = np.asarray(futCon)
        for i in range(len(futCon)):
            if (futCon[i, 1] == 0) and (futCon[i, 0] == 0):
                fut[i, :] = 0
            else:
                fut[i, :] = futCon[i, :] - refPosCon
        # fut = futCon[:, :] - (refPosCon)
        fut[:, [0, 1]] = fut[:, [1, 0]]

        return fut

    ## Added Helper function to convert the coordinates
    def cordConv(self, X, Y, ang):
        deltaY = Y - X * np.tan(ang)
        deltaX = deltaY * np.sin(ang)
        xCon = X / np.cos(ang) + deltaX
        yCon = np.cos(ang) * deltaY
        return xCon, yCon

    ## Added Helper function to convert the velocity and acceleration vectors to the target car's origin-based reference frame
    def vecConv(self, vecX, vecY, ang):
        ang = abs(ang)
        vecXCon = vecX * np.cos(ang) - vecY * np.sin(ang)
        vecYCon = vecX * np.sin(ang) + vecY * np.cos(ang)
        return vecXCon, vecYCon

    ## Collate function for dataloader
    def collate_fn(self, samples):
        ttt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _ in samples:
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)  
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), args['lat_length'])  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), args['lon_length'])  # (batch,2)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                       neighborsdistance, cclass, neighborsclass) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1

            # # Updated the 'op_mask_batch' to reflect only the sensor detected ground truth 'fut' coordinates.
            # for i in range(len(fut_batch)):
            #     if (fut_batch[i, sampleId, 1] == 0) and (fut_batch[i, sampleId, 0] == 0):
            #         op_mask_batch[i, sampleId, :] = 0

            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size)#.byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
                    count4 += 1
        #  mask_batch 
        self.alltime += (time.time() - ttt)
        self.count += args['num_worker']
        #if (self.count > args['time']):
        #    print(self.alltime / self.count, "data load time")
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position


