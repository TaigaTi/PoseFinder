import cv2

image = cv2.imread("data/standing/1.jpeg")

if image is None:
    print("Failed to load image")
else:
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()