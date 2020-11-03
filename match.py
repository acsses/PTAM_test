import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

class mapping:


    img_path1="test1.jpg"
    img_path2="test2.jpg"
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    r_roll =math.radians(0)
    r_yaw =math.radians(0)
    r_pitch =math.radians(0)
    view_w=math.radians(36)
    view_h=math.radians(27)
    # 画像１
    img1 = cv2.imread(img_path1)
    # 画像２
    img2 = cv2.imread(img_path2)

    kp1=[]
    des1=[]
    kp2=[]
    des2=[]
    matches=[]
    def match(self):
        # A-KAZE検出器の生成
        akaze = cv2.AKAZE_create()

        # 特徴量の検出と特徴量ベクトルの計算
        self.kp1, self.des1 = akaze.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = akaze.detectAndCompute(self.img2, None)

        # Brute-Force Matcher生成
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)


        # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
        self.matches = bf.match(self.des1,self.des2)


    def convert_3d(self,data):

        d=750/350*2.54/math.atan(self.view_h)
        d_from_center = math.sqrt((((data[0]-375)/350*2.54)**2)+(((data[1]-250)/350*2.54)**2))
        D=math.sqrt((d**2)+(d_from_center**2))
        if data[0]-375>0 and data[1]-250<0:#1
            r_r=math.atan(((250-data[1])/350*2.54)/((data[0]-375)/350*2.54))
            all_rotate=self.r_roll+r_r
            r_p=math.asin(d_from_center*math.sin(all_rotate)/D)
            all_pitch=self.r_pitch+r_p
            r_y=math.asin(d_from_center*math.cos(all_rotate)/D)
            all_yaw=self.r_yaw-r_y
        elif data[0]-375>0 and data[1]-250>0:#2
            r_r=math.atan(((250-data[1])/350*2.54)/((data[0]-375)/350*2.54))
            all_rotate=self.r_roll+r_r
            r_p=math.asin(d_from_center*math.sin(all_rotate)/D)
            all_pitch=self.r_pitch-r_p
            r_y=math.asin(d_from_center*math.cos(all_rotate)/D)
            all_yaw=self.r_yaw-r_y
        elif data[0]-375<0 and data[1]-250<0:#3
            r_r=math.atan(((250-data[1])/350*2.54)/((data[0]-375)/350*2.54))
            all_rotate=self.r_roll+r_r
            r_p=math.asin(d_from_center*math.sin(all_rotate)/D)
            all_pitch=self.r_pitch-r_p
            r_y=math.asin(d_from_center*math.cos(all_rotate)/D)
            all_yaw=self.r_yaw+r_y
        elif data[0]-375<0 and data[1]-250>0:#4
            r_r=math.atan(((250-data[1])/350*2.54)/((data[0]-375)/350*2.54))
            all_rotate=self.r_roll+r_r
            r_p=math.asin(d_from_center*math.sin(all_rotate)/D)
            all_pitch=self.r_pitch+r_p
            r_y=math.asin(d_from_center*math.cos(all_rotate)/D)
            all_yaw=self.r_yaw+r_y


        x=D*math.cos(all_pitch)*math.sin(all_yaw)
        y=D*math.cos(all_pitch)*math.cos(all_yaw)
        z=D*math.sin(all_pitch)
        return x,y,z


    def convert(self,data):
        data_converted=[]
        for loc in data:
            data_converted.append(self.convert_3d(loc))
        coe = round(data_converted[0][1]+0.6,5)/round(data_converted[1][1],5)

        return round(data_converted[1][0]*coe,5),round(data_converted[1][1]*coe,5),round(data_converted[1][2]*coe,5)

    def collect(self,matches):
        polygon=[]
        for one in matches:
            n =[self.kp1[one.queryIdx].pt,self.kp2[one.queryIdx] .pt]
            summit=self.convert(n)
            polygon.append(summit)
        poly=np.array(polygon).astype(float)
        self.ax.plot(poly.T[0],poly.T[1],poly.T[2],marker="o",linestyle='None')
        self.ax.can_zoom()
        self.ax.view_init(elev=0, azim=0)
        plt.show()


