import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import sys

def readlandmarks(path) :    
    with open(path, 'r') as f:
        points = json.load(f)   
    return points


# Face 1
image_1 = cv2.imread('./images/Im387.png')
img1 = np.copy(image_1)
landmarks_1 = readlandmarks('./landmarks/Im387.json')

for p in landmarks_1:
        cv2.circle(img1, (p[0], p[1]), 4, (0, 255, 0), -1)
cv2.imshow("img1_landmarks", img1)

# Face 2
image_2 = cv2.imread('./images/Im386.png')
img2 = np.copy(image_2)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
landmarks_2 = readlandmarks('./landmarks/Im386.json')
img2_new_face = np.zeros_like(image_2, np.uint8)

for p in landmarks_2:
        cv2.circle(img2, (p[0], p[1]), 4, (0, 255, 0), -1)
cv2.imshow("img2_landmarks", img2)


# Delaunay triangulation
points = np.array(landmarks_1, np.int32)
convexhull = cv2.convexHull(np.array(points, np.int32))
rect = cv2.boundingRect(convexhull)
subdiv = cv2.Subdiv2D(rect)
for p in landmarks_1:
        subdiv.insert(p) 
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype=np.int32)


# find indexes of triangles
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

indexes_triangles = []
for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    
    index_pt1 = np.where((points == pt1).all(axis=1))
    index_pt1 = extract_index_nparray(index_pt1)
    
    index_pt2 = np.where((points == pt2).all(axis=1))
    index_pt2 = extract_index_nparray(index_pt2)
    
    index_pt3 = np.where((points == pt3).all(axis=1))
    index_pt3 = extract_index_nparray(index_pt3)
    
    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        indexes_triangles.append(triangle)


# Triangulation of both faces
for triangle_index in indexes_triangles:
    # Triangulation of the first face
    tr1_pt1 = landmarks_1[triangle_index[0]]
    tr1_pt2 = landmarks_1[triangle_index[1]]
    tr1_pt3 = landmarks_1[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = image_1[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)
    
    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                      [tr1_pt2[0] - x, tr1_pt2[1] - y],
                      [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
    
    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle,
                                       mask=cropped_tr1_mask)
    
    cv2.line(img1, tr1_pt1, tr1_pt2, (0, 0, 255), 2)
    cv2.line(img1, tr1_pt3, tr1_pt2, (0, 0, 255), 2)
    cv2.line(img1, tr1_pt1, tr1_pt3, (0, 0, 255), 2)

    # Triangulation of second face
    tr2_pt1 = landmarks_2[triangle_index[0]]
    tr2_pt2 = landmarks_2[triangle_index[1]]
    tr2_pt3 = landmarks_2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    
    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2
    cropped_triangle2 = image_2[y: y + h, x: x + w]
    cropped_tr2_mask = np.zeros((h, w), np.uint8)
    
    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                       [tr2_pt2[0] - x, tr2_pt2[1] - y],
                       [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
    
    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2,
                                       mask=cropped_tr2_mask)
    
    cv2.line(img2, tr2_pt1, tr2_pt2, (0, 0, 255), 2)
    cv2.line(img2, tr2_pt3, tr2_pt2, (0, 0, 255), 2)
    cv2.line(img2, tr2_pt1, tr2_pt3, (0, 0, 255), 2)

    # Warp triangles
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle,
                                      mask=cropped_tr2_mask)
    
    # Reconstructing destination face
    img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    
    # create a mask to remove the lines between the triangles
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

cv2.imshow("img1_triangles", img1)
cv2.imshow("img2_triangles", img2)

# Face swapped (putting 1st face into 2nd face)
img2_face_mask = np.zeros_like(img2_gray)
convexhull2 = cv2.convexHull(np.array(landmarks_2, np.int32))
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)
img2_head_noface = cv2.bitwise_and(image_2, image_2, mask=img2_face_mask)

result = cv2.add(img2_head_noface, img2_new_face)
cv2.imshow("result_without_line", result)

# seamlessclone
(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
seamlessclone = cv2.seamlessClone(result, image_2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
cv2.imshow("seamlessclone_without_line", seamlessclone)
