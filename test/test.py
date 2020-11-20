import cv2

# 画像１
img1 = cv2.imread("test1.jpg")
# 画像２
img2 = cv2.imread("test2.jpg")
#　画像3
img3=cv2.imread("test3.jpg")

# A-KAZE検出器の生成
akaze = cv2.AKAZE_create()

# 特徴量の検出と特徴量ベクトルの計算
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)
kp3, des3 = akaze.detectAndCompute(img3, None)
# Brute-Force Matcher生成
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


# 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
matches_1 = bf.match(des1,des2)
matches_2 = bf.match(des2,des3)

data=[]


for sample in min((matches_1,matches_2),key=len):
    L=[i for i in max((matches_1,matches_2),key=len) if i.trainIdx==sample.queryIdx]
    if L not in data:
        L.append(sample)
        data.append(L)

print(len(data))

