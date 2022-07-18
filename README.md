# Objectives
Task 1 Face Detection: Given set of images, detect faces in the images and any given test image<br />
Task 2 Face Clustering: Given set of images, cluster similar faces<br />

## Task 1 Face Detection
- For detecting faces, I have used Haar Feature-based Cascade Classifier which uses Haar features for face detection
- I have loaded pretrained classifier using “cv2.cascadeclassifier()”. I have used “detectMultiScale ()” to perform face detection which takes following arguments as inputs:
1. Image Array
2. Scale Factor: Parameter specifying how much the image size is reduced at each image scale.
3. Min Neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
- The detectMultiScale() returns boundary rectangles for the detected faces which I stored in a dictionary. The list of dictionaries containing image name and bounding box is stored in results.json file.

## Task 2 Face Clustering
- Using detected faces from Task 1, for each faces I created 128 dimensional face encoder using face_recognition.face encodings().
- For clustering similar face encodings, I used k-means clustering. 
- The clustering algorithm provided labels of clusters as an output parameter. This labels along with images were stored in clusters.json file.
