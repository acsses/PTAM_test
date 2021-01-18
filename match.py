import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")




r_roll =math.radians(0)
r_yaw =math.radians(0)
r_pitch =math.radians(0)
view_w=math.radians(36.17615)
view_h=math.radians(25.98925)
# 画像１
img1 = cv2.imread("test_1.jpg")
# 画像２
img2 = cv2.imread("test_2.jpg")


img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



# A-KAZE検出器の生成
akaze = cv2.AKAZE_create()


# 特徴量の検出と特徴量ベクトルの計算
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

# Brute-Force Matcher生成
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


# 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
match = bf.knnMatch(des1,des2,k=2)
ratio = 0.5
matches=[]
for m, n in match:
    if m.distance < ratio * n.distance:
        matches.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
cv2.imshow("test",img3)
matches=np.array(matches).flatten().tolist()
cv2.imwrite('img3.jpg', img3)

def convert_3d(data):
    d=375/math.tan(view_w)
    x=((data[0]-375))
    y=d
    z=((250-data[1]))
    
    return round(x,4),round(y,4),round(z,4)

def convert(data):
    data_converted=[]
    for loc in data:
        data_converted.append(convert_3d(loc))
    A=np.matrix([
        [-1*sum(data_converted[1][:2]),sum(data_converted[0][:2])],
        [-1*sum(data_converted[1][1:]),sum(data_converted[0][1:])],
    ])
    Y=np.matrix([
        [-0.6],
        [0],
    ])
    coe = np.linalg.solve(A,Y).reshape(-1,).tolist()
    if data_converted[1][1]*abs(coe[0][1])<210:
        x=1
    else:
        print([i-l for i,l in zip((data_converted[1][0]*coe[0][1],data_converted[1][1]*coe[0][1],data_converted[1][2]*coe[0][1]),(data_converted[1][0]*coe[0][0],data_converted[1][1]*coe[0][0],data_converted[1][2]*coe[0][0]))])
    return data_converted[1][0]*coe[0][0],data_converted[1][1]*coe[0][0],data_converted[1][2]*coe[0][0]

def collect(data):
    polygon=[]
    for one in matches:
        n =[kp1[one.queryIdx].pt,kp2[one.trainIdx].pt]
        summit=convert(n)
        polygon.append(summit)
    polygon=[i for i in polygon if i is not None]
    poly=np.array(polygon).astype(float)
    ax.plot(poly.T[0],poly.T[1],poly.T[2],marker="o",linestyle='None')
    ax.can_zoom()
    ax.view_init(elev=0, azim=0)
    plt.show()
    


collect(matches)