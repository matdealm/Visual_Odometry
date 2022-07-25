from glob import glob
import cv2, skimage, os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

class Cells:
    def __init__(self):
        self.pts = list()
        self.pairs = dict()

    def rand_pt(self):
        return random.choice(self.pts)
class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            self.P = np.array([[self.focal_length,0,self.pp[0],0],[0,self.focal_length,self.pp[1],0],[0,0,1,0]],dtype=np.float64)
            self.K = self.P[:,:3]
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
        self.pts = list()
        self.pairs = dict()
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)
     
    def rand_pt(self):
        return random.choice(self.pts)   
    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        
        return np.array([[x], [y], [z]]) 

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)
    
    def _form_tranform_matrix(self, R, t):
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def get_matches(self, frame_id):
        
        orb = cv2.ORB_create(3000)    
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=12, multi_probe_level=2)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        kp1, des1 = orb.detectAndCompute(self.imread(self.frames[frame_id - 1]), None)
        kp2, des2 = orb.detectAndCompute(self.imread(self.frames[frame_id]), None)
        
        # Find matches
        matches = flann.knnMatch(des1, des2, k=2)

        # Find the matches that are not too far apart
        best_matches = []
        cf = self.imread(self.frames[frame_id])
        img_dim = cf.shape
        
        y_bar, x_bar = np.array(img_dim[:])/8
        
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                best_matches.append(m)
        q1 = np.zeros((len(best_matches),2))
        q2 = np.zeros((len(best_matches),2))
        grid = np.empty((8,8), dtype=object)
        grid[:,:] = Cells()
        
        
            # Generatin Grid & extracting points from matches #
        for i, match in enumerate(best_matches):
            j = int(kp1[match.queryIdx].pt[0]/x_bar)
            k = int(kp1[match.queryIdx].pt[1]/y_bar)
            grid[j,k].pts.append(kp1[match.queryIdx].pt)
            grid[j,k].pairs[kp1[match.queryIdx].pt] = kp2[match.trainIdx].pt

            q1[i] = kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1]
            q2[i] = kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]    
        
        # Get the image points from the good matches
        #q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        #q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        
        return q1, q2, grid            
    
    def find_pose(self, q1, q2, grid, frame_id):
        
        # find fundamental matrix
          
        F = self.FM_RANSAC(q1, q2, grid, 1)
        
        # Essential matrix
        #E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)
        E = self.EstimateEssentialMatrix(F,self.K)
        
        # Decompose the Essential matrix into R and t
        #_, R, t, _ = cv2.recoverPose(E, q1, q2)
        R, t = self.decomp_essential_mat(E, q1, q2)
        t = t * self.get_scale(frame_id)
    
        # Get transformation matrix
        transform_matrix = self._form_tranform_matrix(R, np.squeeze(t))
        
        return transform_matrix
    
    # Get Random 8 points from different regions in a Image 
    def get_rand8(self, grid):            
        cells = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        rand_grid_index = random.choices(cells, k = 8)   
        rand8 = list() 
        rand8_ = list()        
        for index in rand_grid_index:
            if grid[index].pts: 
                pt = grid[index].rand_pt()
                rand8.append(pt)
            else:
                index = random.choice(cells)
                while not grid[index].pts or index in rand_grid_index:
                    index = random.choice(cells) 
                pt = grid[index].rand_pt()
                rand8.append(pt)

            # find the correspondence given point
            rand8_.append(grid[index].pairs[pt])
        return rand8, rand8_
    
    #calculate fundamental matrix
    def fundamental_matrix(self, q1, q2):
        #F_CV,_ = cv.findFundamentalMat(pts_cf,pts_nf,cv.FM_8POINT)
        mat = []
        origin = [0.,0.]
        origin_ = [0.,0.]	
        origin = np.mean(q1, axis = 0)
        origin_ = np.mean(q2, axis = 0)	
        k = np.mean(np.sum((q1 - origin)**2 , axis=1, keepdims=True)**.5)
        k_ = np.mean(np.sum((q2 - origin_)**2 , axis=1, keepdims=True)**.5)
        k = np.sqrt(2.)/k
        k_ = np.sqrt(2.)/k_
        x = ( q1[:, 0].reshape((-1,1)) - origin[0])*k
        y = ( q1[:, 1].reshape((-1,1)) - origin[1])*k
        x_ = ( q2[:, 0].reshape((-1,1)) - origin_[0])*k_
        y_ = ( q2[:, 1].reshape((-1,1)) - origin_[1])*k_
        A = np.hstack((x_*x, x_*y, x_, y_ * x, y_ * y, y_, x,  y, np.ones((len(x),1))))	
        U,S,V = np.linalg.svd(A)
        F = V[-1]
        F = np.reshape(F,(3,3))
        U,S,V = np.linalg.svd(F)
        S[2] = 0
        F = U@np.diag(S)@V	
        T1 = np.array([[k, 0,-k*origin[0]], [0, k, -k*origin[1]], [0, 0, 1]])
        T2 = np.array([[k_, 0,-k_*origin_[0]], [0, k_, -k_*origin_[1]], [0, 0, 1]])
        F = T2.T @ F @ T1
        F = F / F[-1,-1]
        
        return F   
    
    
    # Estimate Fundamental Matrix from the given correspondences using RANSAC
    def FM_RANSAC(self, q1, q2, grid, epsilon = 1):
        max_inliers= 0
        F_best = []
        S_in = []
        confidence = 0.99
        N = sys.maxsize
        count = 0
        while N > count:
            S = []
            counter = 0
            x_1,x_2 = self.get_rand8(grid)
            F = self.fundamental_matrix(np.array(x_1), np.array(x_2))
            ones = np.ones((len(q1),1))
            x = np.hstack((q1,ones))
            x_ = np.hstack((q2,ones))
            e, e_ = x @ F.T, x_ @ F
            error = np.sum(e_* x, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e[:, :-1],e_[:,:-1]))**2, axis = 1, keepdims=True)
            inliers = error<=epsilon
            counter = np.sum(inliers)
            if max_inliers <  counter:
                max_inliers = counter
                F_best = F 
            I_O_ratio = counter/len(q1)
            if np.log(1-(I_O_ratio**8)) == 0: continue
            N = np.log(1-confidence)/np.log(1-(I_O_ratio**8))
            count += 1
        return F_best     
    
    # Estimate Essential Matrix 
    def EstimateEssentialMatrix(self,F,K):
        E = K.T @ F @ K
        U, S, V = np.linalg.svd(E)
        S = [[1,0,0],[0,1,0],[0,0,0]]   
        E = U @ S @ V
        #E = E/np.linalg.norm(E)
        return E

    # Decompose Essential Matrix
    def decomp_essential_mat(self, E, q1, q2):
    
        def sum_z_cal_scale(R, t):
            
            # Get the transformation matrix
            T = self._form_tranform_matrix(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            
            # Also seen from cam
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        
        #decompose the essential without opencv
        #U, _, V = np.linalg.svd(E)
        #W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    
        #R1 = np.dot(np.dot(U,W),V)
        #R2 = np.dot(np.dot(U,W.T),V)
            
        #t = U[:,2]
        #t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []

        for R, t in pairs:
            z_sum = sum_z_cal_scale(R, t)
            z_sums.append(z_sum)


        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]

        R1, t = right_pair

        return [R1, t]
    
    def gt_mat(self, frame_id):
        pose = np.array(self.pose[frame_id], dtype=np.float64)
        pose = pose.reshape(3, 4)
        pose = np.vstack((pose, [0, 0, 0, 1]))
        return pose
    
    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The returned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        
        gt_path = np.zeros((len(self.frames),3))
        estimated_path = np.zeros((len(self.frames),3))
        
        for i in range(len(self.frames)):
                                              
            gt_pose = self.gt_mat(i)                          
                                           
            if i == 0:
                cur_pose = gt_pose
            else: 
                q1, q2, grid = self.get_matches(i)
                transf = self.find_pose(q1,q2,grid, i)
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            gt_path[i] = [gt_pose[0,3], gt_pose[1,3], gt_pose[2,3]]
            estimated_path[i] = [cur_pose[0,3], cur_pose[1,3], cur_pose[2,3]]
        
            #print(gt_path[i])
            #print(estimated_path[i])
            
        return estimated_path
    
    
    
if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)