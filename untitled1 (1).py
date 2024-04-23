# one blur
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('/content/drive/MyDrive/cv lab experiment/bear.jpg')
median_blur = cv2.medianBlur(image, 5)
plt.figure(figsize=(20,5))
plt.subplot(121).imshow(image)
plt.subplot(122).imshow(median_blur)

#question two (Edge detection)
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('D:\cv i116\dog.jpg')
blurred = cv2.medianBlur(image, 5)
edges = cv2.Canny(blurred, 30, 150)
plt.figure(figsize=(25,5))
plt.subplot(121).imshow(image)
plt.subplot(122).imshow(edges)
plt.show()

# Exp 5
#question five (SIFT)
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("D:\cv i116\dog.jpg")
img = cv2.imread('D:\cv i116\dog.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
img=cv2.drawKeypoints(gray ,kp ,img ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(25, 7))
plt.subplot(121).imshow(image)
plt.subplot(122).imshow(img)
plt.show()

#lane
from google.colab.patches import cv2_imshow
import cv2
import numpy as np

def detect_lane_border(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 100)

    # Define region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    cv2.fillPoly(mask, [np.array(region_of_interest_vertices, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)

    # Draw the detected lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return image

# Read the input image
input_image = cv2.imread('/content/drive/MyDrive/cv lab experiment/lane4.jpg')

# Detect lane borders
output_image = detect_lane_border(input_image)

# Display the result
cv2_imshow(output_image)
# question three (lane detection)

#pca
import pandas as pd
import numpy as np
data=pd.read_csv("/content/drive/MyDrive/cv lab experiment/wine.csv")
data.head()
data.isnull().sum()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.svm import SVC
cls=SVC()
cls.fit(X_train,y_train)


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = cls.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)



from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
from sklearn.svm import SVC
cls=SVC()
cls.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = cls.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# 10
import cv2
# Load pre-trained AdaBoost object detection model
object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Function to detect objects using AdaBoost
def detect_objects(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = object_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame
# Main function for capturing video from webcam
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_objects(frame)
        cv2.imshow('Object Detection using AdaBoost', frame)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()