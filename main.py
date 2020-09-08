import numpy as np 
import dlib
import cv2
import math
from playsound import playsound
import threading

cam = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

overlay_real = cv2.imread("f_n.png")

thread_var = False

def act2(src):
    overlay = cv2.resize(overlay_real, (src.shape[1],src.shape[0]), fx=1, fy=1)
    overlayMask = cv2.cvtColor( overlay, cv2.COLOR_BGR2GRAY )
    res, overlayMask = cv2.threshold( overlayMask, 10, 1, cv2.THRESH_BINARY_INV)

    h,w = overlayMask.shape
    overlayMask = np.repeat( overlayMask, 3).reshape( (h,w,3) )

    src *= overlayMask

    src += overlay

    return src

def sound():
    global thread_var
    thread_var = True
    playsound("final.mp3")
    thread_var = False


while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = detector(gray)

    pb_img = cv2.imread("face_only.png") 

    for face in faces:
        x1 = face.left() 
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom()

        landmarks = predictor(gray,face)
        shape = landmarks

        x = x1
        y = y1
        w = x2-x1
        h = y2-y1

        #for width
        w_x1 = landmarks.part(1).x 
        w_y1 = landmarks.part(1).y 
        #cv2.circle(frame, (w_x1,w_y1), 2, (255, 255, 0), -1)

        w_x2 = landmarks.part(15).x 
        w_y2 = landmarks.part(15).y 
        #cv2.circle(frame, (w_x2,w_y2), 2, (255, 255, 0), -1)

        #final width
        w = w_x2-w_x1

        #calculating height
        p33x = landmarks.part(33).x 
        p33y = landmarks.part(33).y 
        #cv2.circle(frame, (p33x,p33y), 2, (255, 255, 0), -1)

        p21x = landmarks.part(21).x 
        p21y = landmarks.part(21).y 
        #cv2.circle(frame, (p21x,p21y), 2, (255, 255, 0), -1)

        p22x = landmarks.part(22).x 
        p22y = landmarks.part(22).y 
        #cv2.circle(frame, (p22x,p22y), 2, (255, 255, 0), -1)

        mid_x = int(p21x + ((p22x-p21x)/2))
        mid_y = int(p21y + ((p22y-p21y)/2))
        #cv2.circle(frame, (mid_x,mid_y), 2, (255, 255, 0), -1)

        chinx = landmarks.part(8).x 
        chiny = landmarks.part(8).y 
        #cv2.circle(frame, (chinx,chiny), 2, (255, 255, 0), -1)

        head_x = mid_x
        head_y = mid_y - (p33y-mid_y)
        #cv2.circle(frame, (head_x,head_y), 2, (255, 255, 0), -1)

        x = w_x1
        y = head_y
        h = chiny-head_y

        #frame[y:y+h, x:x+w] = applyblur(frame[y:y+h, x:x+w])
        frame[y:y+h, x:x+w] = act2(frame[y:y+h, x:x+w])

        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 15, 120), 5)

    cv2.imshow("face detection",frame)  

    if len(faces)>=1 and thread_var==False:
        thread1 = threading.Thread(target=sound)
        thread1.start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()