import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directKeys import PressKey,W,A,S,D
import os

for i in list(range(4))[::-1]:
    print(i+1)
    time.Sleep(1)

print('down')
PressKey('w')
time.Sleep(3)
print('up')
PressKey('w')


def keys_to_output(keys):
    output = [0,0,0]

    if 'A' in keys:
        output[0] =1
    elif 'D' in keys:
        output[2] =1
    else:
        output[1] = 1
    return output

            




def roi(img, verticies):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,verticies,255)
    masked = cv2.bitwise_and(img,mask)
    return masked

def draw_lines(img,lines):
    try:
    for line in lines:
        coords = line[0]
        cv2.line(img,(coords[0],coords[1]),(coords[2],coords[3]),[255,255,255],3)
    except:
        pass
    
def draw_langes(img,lines, color[0,255,255],thickness=3):
    try:
        ys = []
        for i in lines:
            for ii in i:
                ys+= [ii[1],ii[3]]

        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}
        for idx,id in enumerate(lines):
            or xyxy in i:
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m,b = lstsq(A,y_coords)[0]

                x1 = (min_y-b)/m
                x1 = (max_y-b)/m
                line_dict[idx] = [m,b,[int(x1),min_y, int(x2),max_y]]

        final_lanes = {}
        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            if len(final_lanes)==0:
                final_lanes[m]  = [[m,b,line]]
            else:
                found_copy = False
                for other_ms in final_lanes_copy:
                    if not found_copy:
                        if abs(other_ms*1.1)>abs(m)>abs(other_ms*0.9):
                            final_lanes[other_ms].append([m,b,line])
                            found_copy = True
                            break
                        else:
                            final_lanes[m] = [[m,b,line]]
            line_counter = {}
            for lanes in final_lanes:
                line_counter[lanes] = len(final_lanes[lanes])
            top_lanes = sorted(line_counter.item(),key = lambda item: item[1])[::-1]


            lane1_id = top_lanes[0][0]
            lane2_id = top_lanes[1][0]

            def average_lane(lane_data):
                xls = []
                yls = []
                x2s = []
                y2s = []
                for data in lane_data:
                    xls.append(data[2][0])
                    yls.append(data[2][0])
                    x2s.append(data[2][0])
                    y2s.append(data[2][0])
                return int(mean(xls)),int(mean(yls)), int (mean(x2s)),(mean(y2s))
            l1_x1,l1_y1,l1_x2,l1_y2 = average_lane(final_lanes[land1_id])
            l2_x1,l2_y2,l2_x2,l2_y2 = average_lane(final_lanes[land2_id])
            return [[l1_x1,l1_y1,l1_x2,l1_y2] ,[l2_x1,l2_y2,l2_x2,l2_y2]]
        
    
                                  


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img,threshold1= 200,threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    verticies = np.array([[10,500],[10,300],[300,200],[500,200],[800,200],[800,500]])
    
    processed_img = roi(processed_img,[verticies])
    #need to be edges. 
    lines = cv2.HoughLinesP(processed_img,1,np.pi/180,180,20,15)
    draw_lines(processed_img,lines)
    
    
    
    return processed_img

                              
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    
def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    PressKey(A)
 
def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    
def slow_ya_roll():
    Release(W)
    ReleaseKey(A)
    ReleaseKey(D)

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('file does exits')
    training_data = list(np.load(file_name))
    
else:
    print('file does not exits')
    training_data = []


def main:
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()

    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(80,60))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        last_time = time.time()
        
        newscreen,original_image,m1,m2 = process_img(original_img)
        print
        print('Loop took{} seconds'.formate(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window',new_screen)
        if m1<0 and m2<0:
            right()
        elif m1 >0 and m2>0:
            left()
        else:
            straight()
            
        #cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            breakpoint

        if len(training_data)%500 == 0 :
            print(len(training_data))
            np.save(file_name ,training_data)
            
