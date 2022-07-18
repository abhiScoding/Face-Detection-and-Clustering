'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    # print(input_path)
    result_list = []
    '''
    Your implementation.
    '''
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    all_imgs = [images for images in os.listdir(input_path)]
    for img_name in all_imgs:

        imgArr = cv2.imread(os.path.join(input_path,img_name))
        faces = face_cascade.detectMultiScale(imgArr, 1.2, 5)
        for (x, y , w ,h) in faces:
            resultDict = {}
            resultDict["iname"] = img_name
            [x,y,w,h] =np.array([x,y,w,h]).astype("float")
            resultDict["bbox"] = [x,y,w,h]
            result_list.append(resultDict)

    
    return result_list



'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    K = int(K)
    faces_result_bbox = detect_faces(input_path)
    #all_imgs = [images for images in os.listdir(input_path)]
    #print(faces_result_bbox)
    face_data = []
    for face_elem in faces_result_bbox:
        
        imgArr = cv2.imread(os.path.join(input_path,face_elem['iname']))
        [x,y,w,h] = np.array(face_elem['bbox']).astype("int")
        face_data.append(face_recognition.face_encodings(imgArr, [(x, y,x+w,y+h)] )[0])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    kmean_data = np.array(face_data).astype(np.float32)
    # print(kmean_data.shape)
    compactness,labels,centers =cv2.kmeans(kmean_data, K, None,criteria,10,flags)
    # print(labels)
    # print(compactness)
    for j in range(K):
        element_val ={"cluster_no": j, "elements": []}

        for i in range(len(faces_result_bbox)):
            if labels[i]==j:
                element_val["elements"].append(faces_result_bbox[i]['iname'])
        result_list.append(element_val)
    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
