import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure()
ax = Axes3D(fig)

x_1 = [-2,2]
y_1 = [12,16]
X_1,Y_1 = np.meshgrid(x_1,y_1)
x_2 = [-2,2]
y_2 = [-3.4,0.6]
X_2,Y_2 = np.meshgrid(x_2,y_2)
x_3 = [12,16]
y_3 = [-3.4,0.6]
X_3,Y_3 = np.meshgrid(x_3,y_3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_ylim([-20,20])
ax.set_zlim([-20,20])
ax.set_xlim([-20,20])

ax.plot_surface(X_1,Y_1,np.array([[0.6, 0.6], [0.6, 0.6]]),alpha=0.4,color=('g'))
ax.plot_surface(X_1,Y_1,np.array([[-3.4, -3.4], [-3.4, -3.4]]),alpha=0.4,color=('g'))
ax.plot_surface(X_2,np.array([[12, 12], [12, 12]]),Y_2,alpha=0.4,color=('g'))
ax.plot_surface(X_2,np.array([[16, 16], [16, 16]]),Y_2,alpha=0.4,color=('g'))
ax.plot_surface(np.array([[-2, -2], [-2, -2]]),X_3,Y_3,alpha=0.4,color=('g'))
ax.plot_surface(np.array([[2, 2], [2, 2]]),X_3,Y_3,alpha=0.4,color=('g'))




r_roll =math.radians(0)
r_yaw =math.radians(0)
r_pitch =math.radians(0)
view_w=math.radians(36.17615)
view_h=math.radians(25.98925)
range=16
# 画像１
img1 = cv2.imread("test_c_1.JPG")
# 画像２
img2 = cv2.imread("test_c_2.JPG")





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

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2,matches, None, flags=2)

# 画像表示
cv2.imwrite('img1_3.jpeg', img3)

matches=np.array(matches).flatten().tolist()

def convert_3d(data):
    d=3000/math.tan(view_w)
    x=((data[0]-3000))
    y=d
    z=((2000-data[1]))
    
    return x,y,z


def each_delta(a,b):
    A_1=sum(a[:2])
    A_2=sum(a[1:])
    B_1=sum(b[:2])
    B_2=sum(b[1:])
    U=np.array([
        [np.cross([A_1,A_2,a[0]],[B_1,B_2,b[0]]).tolist()[0],np.cross([A_1,A_2,a[1]],[B_1,B_2,b[1]]).tolist()[0],np.cross([A_1,A_2,a[2]],[B_1,B_2,b[2]]).tolist()[0]],
        [np.cross([A_1,A_2,a[0]],[B_1,B_2,b[0]]).tolist()[1],np.cross([A_1,A_2,a[1]],[B_1,B_2,b[1]]).tolist()[1],np.cross([A_1,A_2,a[2]],[B_1,B_2,b[2]]).tolist()[1]],
        [np.cross([A_1,A_2,a[0]],[B_1,B_2,b[0]]).tolist()[2],np.cross([A_1,A_2,a[1]],[B_1,B_2,b[1]]).tolist()[2],np.cross([A_1,A_2,a[2]],[B_1,B_2,b[2]]).tolist()[2]]
    ])
    A=np.array([
        [sum(a[:2]),sum(a[1:])],
        [sum(b[:2]),sum(b[1:])]
    ])
    R=np.array([
        [a[0],a[1],a[2]],
        [b[0],b[1],b[2]],
    ])
    alufa=np.cross(A[:,0],A[:,1])
    Right=np.matrix([
        [U[0,0]+U[0,2],U[0,0]+U[0,1],U[0,1]],
        [U[0,0],U[0,0]+U[0,1]+U[1,2],U[0,1]],
        [U[0,0],U[0,0]+U[0,1],U[0,1]+U[2,2]]
    ])
    Left=np.matrix([
        [-1*(-5*U[0,0]+0*U[0,1]-5*U[0,2])],
        [-1*(-5*U[0,0]+0*U[0,1]+0*U[1,2])],
        [-1*(-5*U[0,0]+0*U[0,1]+0*U[2,2])],
    ])
    coe_delta = (np.linalg.pinv(Right)*Left).reshape(-1,).tolist()
    return (coe_delta[0][0],coe_delta[0][1],coe_delta[0][2])

def convert(data):
    data_converted=[]
    for loc in data:
        data_converted.append(convert_3d(loc))
    d_x,d_y,d_z=each_delta(data_converted[0],data_converted[1])
    A=np.matrix([
        [-1*sum(data_converted[1][:2]),sum(data_converted[0][:2])],
        [-1*sum(data_converted[1][1:]),sum(data_converted[0][1:])],
    ])
    if abs(d_x**2+d_y**2+d_z**2)<range:
        Y=np.matrix([
            [-5+d_x+d_y],
            [0+d_y+d_z],
        ])
        coe = np.linalg.solve(A,Y).reshape(-1,).tolist()
        return data_converted[1][0]*coe[0][0]-5,data_converted[1][1]*coe[0][0],data_converted[1][2]*coe[0][0]
    else:
        Y=np.matrix([
            [-5+2*math.sqrt(range/3)],
            [0+2*math.sqrt(range/3)],
        ])
        coe = np.linalg.solve(A,Y).reshape(-1,).tolist()
        return data_converted[1][0]*coe[0][0]-5,data_converted[1][1]*coe[0][0],data_converted[1][2]*coe[0][0]

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