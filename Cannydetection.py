import cv2

# 读取图像
img = cv2.imread('data/render/test/airplane/0003_(9).png', cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img,80,150)
print(v1.shape)
# v2 = cv2.Canny(img,50,100)
cv2.imshow('image.jpg',v1)
cv2.waitKey(0)
cv2.destroyAllWindows()