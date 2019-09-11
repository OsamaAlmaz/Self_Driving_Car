import numpy as np
from PIL import ImageGrab
import cv2
import time

from directKeys import PressKey,W,A,S,D

for i in list(range(4))[::-1]:
    print(i+1)
    time.Sleep(1)

print('down')
PressKey('w')
time.Sleep(3)
print('up')
PressKey('w')









def process_img(original_image):
    processed_img = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img,threshold1= 200,threshold2=300)
    return processed_img

                              
    
    
 

last_time = time.time()

while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    newscreen = process_img(original_img)
    print('Loop took{} seconds'.formate(time.time()-last_time))
    last_time = time.time()
    cv2.imshow('window',new_screen)
    
    #cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
