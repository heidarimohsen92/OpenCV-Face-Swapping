import cv2
import numpy as np
import json

point_matrix = np.zeros((26,2),np.int)
num_point = len(point_matrix)
counter = 0

# Create point matrix get coordinates of mouse click on image
def mousePoints(event, x, y, flags, params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = int(x), int(y)
        counter = counter + 1

# Read image
img = cv2.imread('./images/Im387.png')

while True:
    # show circle of points
    for x in range (0,num_point):
        cv2.circle(img,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv2.FILLED)
    
    if counter == num_point:
        # save in json
        with open('./landmarks/Im387.json', 'w') as f:
            json.dump(point_matrix.tolist(), f)

    # Showing original image
    cv2.imshow("Original Image ", img)
    cv2.setMouseCallback("Original Image ", mousePoints)
    if cv2.waitKey(20) == ord("q"):
            break
