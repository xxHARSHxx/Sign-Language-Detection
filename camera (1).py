import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from keras.models import load_model
model=load_model('signdet.h5')
str=""
mphands = mp.solutions.hands
hands = mphands.Hands()
cap = cv2.VideoCapture(0)
s, frame = cap.read()
h, w, c = frame.shape
x_max = 0
y_max = 0
x_min = w
y_min = h
startx=0
starty=0
while True:
    s, frame = cap.read()
    analysis_frame=frame
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 15
            y_max += 15
            x_min -= 15
            x_max += 15
            diff=abs((y_max-y_min)-(x_max-x_min))
            diff=diff//2
            if(y_max-y_min>x_max-x_min):
                startx=x_min-diff
                starty=y_min
                cv2.rectangle(frame, (x_min-diff, y_min), (x_max+diff,y_max), (255, 255, 255), 1)
            else:
                startx=x_min
                starty=y_min-diff
                cv2.rectangle(frame, (x_min, y_min-diff), (x_max, y_max+diff), (255, 255, 255), 1)
    cv2.putText(frame,str,(startx,starty-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,165,255),2)
    cv2.imshow("Frame", frame)
    letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z','del','nothing','space']
    
    if k%256==32:
        #space pressed
        diff=abs((y_max-y_min)-(x_max-x_min))
        diff=diff//2
        if(y_max-y_min>x_max-x_min):
            analysis_frame=analysis_frame[y_min:y_max,x_min-diff:x_max+diff]
        else:
            analysis_frame=analysis_frame[y_min-diff:y_max+diff,x_min:x_max]
        #plt.imshow(analysis_frame)
        #plt.show()
        resized = cv2.resize(analysis_frame, (64, 64 ))
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        cv2.imwrite("hand.png",image)
        image = image.reshape(-1,64,64,3)
        prediction = model.predict(image)
        predarray = np.array(prediction[0])
        #print(predarray)
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        for key,value in letter_prediction_dict.items():
            if value==high1:
                if key=="del":
                    str=""
                elif key=="space":
                    str+=" "
                elif key=="nothing":
                    str=str
                else:
                    str+=key
        print(str)

cap.release()
cv2.destroyAllWindows()
