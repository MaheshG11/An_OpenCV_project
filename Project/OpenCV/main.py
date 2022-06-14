import cv2 as cv
import math
import numpy as np
import imutils
import cv2.aruco as aruco

'''The function Colour_in_Range checks if the given color(BGR value) is in range of lower_limit(given) and 
upper_limit(given)'''
def Colour_in_Range(given_colour,lower_limit,upper_limit):
    if given_colour[0] in range(lower_limit[0],upper_limit[0]+1):
        if given_colour[1] in range(lower_limit[1], upper_limit[1]+1):
            if given_colour[2] in range(lower_limit[2], upper_limit[2]+1):
                return True
    else :
        return False



'''The function IsEqual(s) takes contours of a shape(that is approx) as input and checks if the adjacent 
sides are equal. if equal function returns True else it returns False '''
def IsEqual(s):
    dx=math.sqrt(math.pow((s[0][0][1]-s[1][0][1]),2)+math.pow((s[0][0][0]-s[1][0][0]),2))
    dy= math.sqrt(math.pow((s[1][0][1] - s[2][0][1]), 2) + math.pow((s[1][0][0] - s[2][0][0]), 2))
    if abs(dx-dy)<3:
        return True
    else :
        return False



'''The function IsEqual(s) takes contours of a shape(that is approx) as input and checks if 3 of
 the angles are at 90 degrees. if 90 degrees function returns 
True else it returns False '''
def IsRightA(s):
    a1=math.atan((s[0][0][1]-s[1][0][1])/(s[0][0][0]-s[1][0][0]))
    a2= math.atan((s[1][0][1] - s[2][0][1]) / (s[1][0][0] - s[2][0][0]))
    a3= math.atan((s[2][0][1] - s[3][0][1]) / (s[2][0][0] - s[3][0][0]))
    a4= math.atan((s[0][0][1] - s[3][0][1]) / (s[0][0][0] - s[3][0][0]))
    if ((abs(a1-a2)>1.567)&(abs(a1-a2)<1.573))&((abs(a3-a4)>1.567)&(abs(a3-a4)<1.573))&((abs(a2-a3)>1.567)&(abs(a2-a3)<1.573)):
        return True
    else :
        return False



'''This function takes aruco image ,length and width of aruco required,length and height of bounding rectangle
    , angle at which aruco is required to be imported in the image, and returns the image in the dimensions of
     bounding rectangle and aruco inclined at an required angle'''
def Extract_aruco_image(img2, aruco_length, aruco_width, length_rect, height_rect, angle):
    img1 = img2.copy()
    gimg = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    r, t = cv.threshold(gimg, 220, 255, cv.THRESH_BINARY_INV)
    c, h = cv.findContours(t, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    l = []
    for i in c:
        l.append(cv.contourArea(i))
    i = l.index(max(l))
    peri = cv.arcLength(c[i], True)
    approx = cv.approxPolyDP(c[i], peri * 0.04, True)
    a2 = math.atan((approx[1][0][1] - approx[2][0][1]) / (approx[1][0][0] - approx[2][0][0]))
    a2 = 180 / 3.1416 * a2
    img1 = imutils.rotate(img1, a2)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    r, t = cv.threshold(img1, 220, 255, cv.THRESH_BINARY_INV)
    c, h = cv.findContours(t, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    l = []
    for i in c:
        l.append(cv.contourArea(i))
    i = l.index(max(l))
    peri = cv.arcLength(c[i], True)
    approx = cv.approxPolyDP(c[i], peri * 0.04, True)

    img1 = img1[approx[0][0][0]:approx[2][0][0], approx[0][0][1]:approx[2][0][1]]
    img1 = cv.resize(img1, (aruco_width,aruco_length))

    blank = np.zeros((length_rect, height_rect), np.uint8)
    blank[int((length_rect - aruco_length) / 2):int((length_rect + aruco_length) / 2),
    int((height_rect - aruco_width) / 2):int((height_rect + aruco_width) / 2)] = \
        blank[int((length_rect - aruco_length) / 2):int((length_rect + aruco_length) / 2),
        int((height_rect - aruco_width) / 2):int((height_rect + aruco_width) / 2)] + img1
    blank = imutils.rotate(blank, angle)
    blank = cv.cvtColor(blank, cv.COLOR_GRAY2BGR)
    return blank



'''
Takes image in which aruco is to be placed and list of aruco images as input and returns the 
given image with aruco paced in it '''

def shape_of_image(img2,list2):
    img1=img2.copy()
    gimg=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    r,t=cv.threshold(gimg,230,255,cv.THRESH_BINARY_INV)
    c,h=cv.findContours(t,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    for i in c:
        if cv.contourArea(i)>1000:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, peri * 0.03, True)
            if len(approx)==4:
                if IsRightA(approx):
                    if IsEqual(approx):
                        dx = math.sqrt(math.pow((approx[0][0][1] - approx[1][0][1]), 2) + math.pow(
                            (approx[0][0][0] - approx[1][0][0]), 2))
                        dy = math.sqrt(math.pow((approx[1][0][1] - approx[2][0][1]), 2) + math.pow((approx[1][0][0] - approx[2][0][0]),2))
                        a2 = math.atan((approx[1][0][1] - approx[2][0][1]) / (approx[1][0][0] - approx[2][0][0]))
                        dx=int(dx)
                        dy=int(dy)
                        a2 = 180 / 3.1416 * a2

                        x, y, w, h = cv.boundingRect(i)

                        if Colour_in_Range(img1[int(y+(h/2)),int(x + (w/2))],(0,128,0),(152, 255, 154)):#green
                            ind=list2.index(1)
                            aruco_img=Extract_aruco_image(list1[ind],dx,dy,h,w,-a2)

                            cv.putText(aruco_img,f"ID:{list2[ind][0][0]}",(int(dx/2),int(dy/2)),
                                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif Colour_in_Range(img1[int(y+(h/2)),int(x + (w/2))],(0,100,200),(153,204,255)):#orange
                            ind = list2.index(2)
                            aruco_img = Extract_aruco_image(list1[ind], dx, dy, h, w, -a2)
                            cv.putText(aruco_img, f"ID:{list2[ind][0][0]}", (int(dx / 2), int(dy / 2)),
                                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif Colour_in_Range(img1[int(y+(h/2)),int(x + (w/2))], (0, 0, 0), (20,20,20)):  #black

                            ind = list2.index(3)
                            aruco_img = Extract_aruco_image(list1[ind], dx, dy, h, w, -a2)
                            cv.putText(aruco_img, f"ID:{list2[ind][0][0]}", (int(dx / 2), int(dy / 2)),
                                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif Colour_in_Range(img1[int(y+(h/2)),int(x + (w/2))], (200, 200, 200), (250, 250, 250)):  # pink-peach
                            ind = list2.index(4)
                            aruco_img = Extract_aruco_image(list1[ind], dx, dy, h, w, -a2)
                            cv.putText(aruco_img, f"ID:{list2[ind][0][0]}", (int(dx / 2), int(dy / 2)),
                                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv.drawContours(img1,[approx],-1,(0,0,0),-1)
                        img1[y:y+h,x:x+w]=img1[y:y+h,x:x+w]+aruco_img
    return img1





'''Returns aruco id for a given aruco image'''
def fa(img):
    g1=img.copy()
    key=getattr(aruco,f'DICT_5X5_250')
    ardi=aruco.Dictionary_get(key)
    arucoparam=aruco.DetectorParameters_create()
    (cor,ids,rej)=cv.aruco.detectMarkers(img,ardi,parameters= arucoparam)
    cv.putText(g1,f'{ids}',(int(591/2),int(591/2)),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return ids



# MAIN
aruco3 = cv.imread("Data\\3.jpg")
aruco4 = cv.imread("Data\\4.jpg")
aruco1 = cv.imread("Data\\1.jpg")
aruco2 = cv.imread("Data\\2.jpg")
list1=(aruco1,aruco2,aruco3,aruco4) #list of aruco images

list2 = []
for i in list1:
    list2.append(fa(i))
''' list2 is of aruco ids such that it corresponds the aruco images of list 1'''

if __name__=="__main__":
    print("The image may open in background")
    print("Make Sure NOT to include double inverted commas(\"\") in path name")
    path=input("Enter the path of the image: ")
    path=cv.imread(str(path))
    cv.namedWindow("Final", cv.WINDOW_NORMAL)
    output=shape_of_image(path,list2)

    cv.imshow("Final", output)
    cv.imwrite("Final.jpg",output)
    cv.namedWindow("Given image", cv.WINDOW_NORMAL)
    cv.imshow("Given image",path)

    print("The image may Have opened in background")
    cv.waitKey(0)
    cv.destroyAllWindows()