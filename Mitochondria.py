import numpy as np
from numpy.linalg import *
import cv2
import pandas as pd
from scipy import ndimage
from operator import itemgetter, attrgetter
import csv
import sys
#output in csv format
pi=3.141592653589793
thrs1 = 80
thrs2 = 150
objects = []
collection_r2=[]
collection_velocity=[]
collection_MT = {}
global Event_F , Event_S
Event_F = []
Event_S = []
r1, r2, r3, r4, r5 = [], [], [], [], []
m_pt = 175
kernel = np.ones((3,3),np.uint8)
'''
R1 = [0:70,90:200],
R2 = [100:180,270:400],
R3 = [200:280,430:512],
R4 = [250:350,120:250],
R5 = [350:,290:]
'''
def exchange(i, j):
    return j, i
# %%  One-to-one Transformation (Filtering)
class Cell:
    def __init__(self):       
        #x is the x coordinate
        #y is the y coordinate
        #pca1 pca2 pca transformation value
        #fusion flag default is zero
        self.x=[]
        self.y=[]
        self.pca1=0
        self.pca2=0
        self.pca3=0
        self.fusion_flag=0
        self.frame=0
    def new_object(self,x_,y_,pca1_,pca2_,pca3_):
        self.x.append(x_)
        self.y.append(y_)
        self.frame+=1
        self.pca1=pca1_
        self.pca2=pca2_
        self.pca3=pca3_
    def pca_score(self):
        ret=(0.8*self.pca1)+(0.5*self.pca2)+(0.3*self.pca3)
        return ret       
    def fusion_update(self,para):
        if(para):
            self.fusion_flag=1
      
    def new_frame(self,x_,y_,pca1_,pca2_,pca3_):
        self.frame+=1
        self.x.append(x_)
        self.y.append(y_)        
        if(self.fusion_flag==0):
            self.pca1=pca1_
            self.pca2=pca2_
            self.pca3=pca3_
    def top_x(self):
        retx=self.frame-1
        return self.x[retx]
    def top_y(self):
        rety=self.frame-1
        return self.y[rety]
def extract(img,cont_frame):
    height, width=cont_frame.shape
    mmax_x=-100
    mmin_x=1000000
    mmax_y=-100
    mmin_y=1000000
    for x in range(height-1):
        for y in range(width-1):
            if (cont_frame[x][y]==1):
                if(x>mmax_x):
                    mmax_x=x
                elif (mmin_x>x):
                    mmin_x=x
                if(y>mmax_y):
                    mmax_y=y
                elif (mmin_y>y):
                    mmin_y=y              
    #construct the square frame for PCA to apply
    #img[y: y + h, x: x + w] 
    new_region=img[mmin_x:mmax_x,mmin_y:mmax_y] 
#    new_region=img[mmin_y:mmax_y,mmin_x:mmax_x]
    return new_region

def img2vector(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows,cols = gray_image.shape
    imgVector = np.asarray(gray_image) 
              
    return imgVector
#fix for error
def error_fixing(prev,cur):
    if abs(prev-cur)>0.1:
        cur=0.5*(prev+cur)
    return cur

def PCA(dataMat,topNfeat= 3):
    meanVals=np.mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=np.cov(meanRemoved, rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    eigValInd=np.argsort(eigVals)
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat ,reconMat

def T(x,m,a,b):                
    x=np.float32(x)
    x[x<50]=0
    x[x>150]=255
    y = x.copy()
    x = x/255
    y = y/m
    x[x<(m/255)]= y[y<1]**a                          ## 0.8  
    x[x>=(m/255)] = (x[x>=(m/255)]-m/255)**b+m/255   ## 1.2
    x = np.uint8(x*255)
    x[x<50]=0
    return x
    
def nothing(x):
    pass

# %%   Remove redundant(too small) contours

def m_contours(u):
    y = []     #numOfinvalidcnt 
    
    for i in range(len(u)):
        con_size = np.size(u[i])
        if con_size > 5:

            cnt = unique_rows(np.squeeze(u[i]))  
            cnt_shape = list(cnt.shape)
            cnt_shape.insert(1,1)
            cnt_shape = tuple(cnt_shape)
            cnt = np.reshape(cnt,cnt_shape)
            
            con_size = np.size(cnt)
            if con_size < 13:
                y.insert(0,i)
            else:
                u[i]=cnt
        else:
            y.insert(0,i)
            

    for j in y:
        u.pop(j)
        
    return u

# %%   Matching and MatchingII are essentially the same 

#responsible for labeling numbers
#using for first few frame
def Matching(odd1, odd2, odd3, odd_r, new_x, new_y, j, Area):
    
    odd_r.sort(key=itemgetter(-1))
    dist1 = np.sqrt(odd_r[-1][-1])*3    ### distance indicator
    odd_r.sort(key=itemgetter(3))
    dist2 = 1000
    k1 = 0                              # k1 is a checker 
    arr = []
    
    for i in range(len(odd_r)):
        a , b = odd_r[i][0], odd_r[i][1] 
        dist = np.sqrt((new_x-a)**2+(new_y-b)**2)
        if dist < dist1 :
            dist1 = dist
            arr.append(odd_r[i][-1])
            if dist > 0.:
                speed_y , speed_x = (new_x-a)*pixelsize/seco , (new_y-b)*pixelsize/seco
            else:
                speed_y , speed_x = 0,0
            next_py,next_px = new_x, new_y                    ## math.isnan() to chaeck for nan
            k1 = 1
            
            #l3 for labeling of number in every frame
            #l1 is corrospond to last frame
            #l2 is current frame ordering
            #l3 is assigned value at starting point
            l1 , l2 ,l3 = odd_r[i][3] , j+1  , odd_r[i][-3]   ## i is the old index , j is the new index
        elif dist > dist1:
            dist2 = dist            
    if k1 == 0:
        next_py,next_px = new_x, new_y
        speed_y , speed_x = 50 , 50
        l1 , l2 , l3= 0 , i+1 , 0
    else:
        pass
    if dist1 < dist2:
        z= [next_py,next_px,l1,l2, int(dist1*1000)/1000  , l3 ]
        v= [next_py,next_px, speed_x, speed_y,l1,l2 , int(dist1*1000)/1000  ,l3 ]
    else:
        z= [next_py,next_px,l1,l2, int(dist2*1000)/1000  , l3 ]
        v= [next_py,next_px, speed_x, speed_y,l1,l2 , int(dist2*1000)/1000  ,l3 ]        
    return  z,v
#using for later frame
def MatchingII(odd1, odd2, odd3, odd4, new_x, new_y, j, Area):
    odd4.sort(key=itemgetter(-1))
    dist1 = np.sqrt(odd4[-1][-1])*3    # parametrise this value
    dist2 = dist1
    odd4.sort(key=itemgetter(3))
    
    k1 = 0                             # k is a checker 
    arr = []

    for i in range(len(odd4)):
        a , b= odd4[i][0], odd4[i][1]
        dist = np.sqrt((new_x-a)**2+(new_y-b)**2)
#        if dist < dist2 :
#            arr.append([odd4[i][-1],dist])              
        if dist < dist1 :
            dist1 = dist
            arr.append(odd4[i])      
            if dist > 0.:
                speed_y , speed_x = (new_x-a)*pixelsize/seco , (new_y-b)*pixelsize/seco
            else:
                speed_y , speed_x = 0,0
            next_py,next_px = new_x, new_y                  ## math.isnan() to chaeck for nan
            k1 = 1
            l1 , l2 ,l3 = odd4[i][3] , j+1  , odd4[i][-3]   ## i is the old index , j is the new index
        elif dist < dist2:
            dist2 = dist   
    if k1 == 0:
        next_py,next_px = new_x, new_y
        speed_y , speed_x = 50 , 50
        l1 , l2 , l3= 0 , i+1 , 0
    else:
        pass

    if dist1 < dist2:
        z= [next_py,next_px,l1,l2, int(dist1*1000)/1000 , l3 ]
        v= [next_py,next_px, speed_x, speed_y,l1,l2 , int(dist1*1000)/1000  ,l3 ]
    else:
        z= [next_py,next_px,l1,l2, int(dist2*1000)/1000  , l3 ]
        v= [next_py,next_px, speed_x, speed_y,l1,l2 , int(dist2*1000)/1000  ,l3 ]
    return  z,v, status

# %%       Cheking for fusion

def Match_F(tagNum,odd3,odd4):
    odd3_left = []
    odd4_left = []
    
    ## Find (tagNum)th MT in odd3
    for i in range(len(odd3)):
        if odd3[i][3] == tagNum:
            odd3_left.append(odd3[i])
            a , b  = odd3[i][0] , odd3[i][1]
    
    ## Find the possible MT undergoes fusion
    for i in range(len(odd4)):
        if odd4[i][4] > 1.5:                ## 1.5 is the distance about a diagonally shift
            odd4_left.append(odd4[i])

    ## Find the MT in odd4 that is closest to (tagNum)th MT in odd3
    diff = 20
    l = 0
    for i in range(len(odd4_left)):
        diff1 = np.sqrt((a-odd4_left[i][0])**2+(b-odd4_left[i][1])**2)
        if diff1 < diff:
            diff = diff1
            l = i
    ##  Then, odd4_left[l] is the one we need
    
    ## Compare the change in area
    if odd4_left != []:
        tagInodd3 = odd4_left[l][2]
        area = odd4_left[l][-1]

        for i in range(len(odd3)):
            if odd3[i][3] == tagInodd3:
                odd3_left.append(odd3[i])    
        
        area1 = 0 
        for i in range(len(odd3_left)):
            area1 += odd3_left[i][-1]
        if abs(area - area1)/area < 0.2: 
            i = odd4.index(odd4_left[l])
            odd4[i][-2] = 'F'
            Event_F.append([odd4[i],cap.get(1)-2])
#            Event_F.append([int(cap.get(1)-2),odd4[i][3]])
        Event_F.append(0)
            
    return odd4

# %%      Checking for splitting (may be useless)
def Match_S(tagNum,odd3,odd4):
    odd3_left = []
    odd4_left = []
    
    for i in range(len(odd4)):
        if odd4[i][2] == tagNum:
            if odd4[i][4] > 1.5:
                odd4_left.append(odd4[i])
                
    ## Compare the change in area            
    area1 = 0
    for i in range(len(odd4_left)):
        area1 += odd4_left[i][-1]
        
    odd3_left = odd3[tagNum-1]
    area = odd3_left[-1]
    
    if abs(area - area1)/area < 0.3: 
        for i in range(len(odd4_left)):
            odd4_left[i][-2] = 'S'
            Event_S.append([odd4_left[i],cap.get(1)-2])
#            Event_S.append([int(cap.get(1)-2),odd4_left[i][3]])
    Event_S.append(0)
    return odd4


# %%  


def MT_individualize(odd3,odd4_i):
    if odd3 == []:
        pass
        

    
    

# %%  
def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


# %%    The main loop

videofile=input("Please enter the location of the video file:")

cap = cv2.VideoCapture(r'%s' %videofile)

pixelsize=input("Please enter the pixel size in nm:")
seco=input("Please enter the time separation between frames in s:")
#cv2.namedWindow('original')
#cv2.createTrackbar('Threshold','video',0,255,nothing)
#for i in range(380):
#    flag, img = cap.read()
 
while True:
    flag, img = cap.read()  
    if flag == True:       
        img = img[0:256,:]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        dim = list(img.shape)
#        max_bf = np.max(gray)
#        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
#        max_aft = np.max(gray)
#        gray = gray + (max_bf - max_aft)
#        gray[gray<50] = 0

#        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        #tv = cv2.getTrackbarPos('Threshold','video')
#        enlarge = cv2.pyrUp(gray,dstsize=(dim[1]*2,dim[0]*2))
        
        gray1 = T(gray,m_pt,0.8,1.05)
        check1 = gray1.copy()
        gray1 = cv2.GaussianBlur(gray1,(3,3),0)    
        check2 = gray1.copy()
               
        ret,thresh1=cv2.threshold(gray1,40,255,cv2.THRESH_TOZERO)    ####THRESH_BINARY
        #ret, BW = cv2.threshold(gray1,150,255,cv2.THRESH_TRUNC)
        dim = np.shape(thresh1)
        check3 = thresh1.copy()
        thresh1 = 255-thresh1
        check4 = thresh1.copy()
        laplacian = cv2.Laplacian(thresh1,cv2.CV_8U,ksize=3)
        laplacian1 = laplacian.copy()
        
        
#        laplacian1 = np.uint8(laplacian1)

        laplacian2 = cv2.GaussianBlur(laplacian1,(5,5),0)
#        laplacian2 = cv2.morphologyEx(laplacian2, cv2.MORPH_OPEN, kernel)
        laplacian2[laplacian2< 60] = 0
        laplacian3 = laplacian2.copy()
        
        edge = cv2.Canny(laplacian2, thrs1, thrs2, apertureSize=5)  # thresh1 or laplacian1 or laplacian2
        
        vis = img.copy()
        unitkernal = np.zeros((256,512))
        vis = np.uint8(vis)
#        vis1 = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
#        vis[edge != 0] = (0, 255, 0)
        image, contours, hierarchy = cv2.findContours(edge, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
        contours = m_contours(contours)  ## Remove redundant  contours
                
# %%     Obtain MT data in the present frame

        r5=[]
        velocity = []
        no_of_contours=len(contours)
        for i in range(len(contours)):            
            cnt = contours[i]
            #unikernel is for drawing in a blank
            cv2.drawContours(unitkernal,[cnt],-1,1,-1,1,maxLevel=2)
            #for pca process
            vis_1=img2vector(vis)
            pca_input=extract(vis_1,unitkernal)
            #lowDDataMat_,reconMat_=PCA(pca_input,3)
            #input to the structure            
            Area = sum(sum(unitkernal))
            M00 = sum(sum(unitkernal*img[:,:,2]))    ## img[:,:,2]
            M10 , M01 = ndimage.measurements.center_of_mass(unitkernal*img[:,:,2])
            
            if M00 != 0:
                #CG calculated
                cx = M01
                cy = M10
                status = 'N'
#                r1 = [cx,cy]
                if r4 == []:
                    
                    z =[cx,cy,0,i+1, i+1, status, Area]
                    r5.append(z)
                                 
                    ## global['a'] = 15
                    
                    if (i+1)//10 < 1:
                        collection_MT['MT_0'+str(i+1)] = []
                        collection_MT['MT_0'+str(i+1)].append(z)
                    else:
                        collection_MT['MT_'+str(i+1)] = []
                        collection_MT['MT_'+str(i+1)].append(z)
                elif r1 == []:
                    z  , v = Matching(r1, r2 , r3 , r4, cx, cy, i, Area)
                    z.append(status)
                    z.append(Area)
                    r5.append(z)
                    velocity.append(v)
                else:
                    z , v , status = MatchingII(r1, r2 , r3 , r4, cx, cy, i, Area)
                    z.append(status)
                    z.append(Area)
                    r5.append(z)
                    velocity.append(v)                                             
                cv2.circle(vis,(int(cx),int(cy)),2,(0,0,255),-1)
                unitkernal[:,:] = 0 
                cv2.drawContours(vis,cnt,-1,(0,255,0),1)
                font = cv2.FONT_HERSHEY_SIMPLEX
#                cv2.putText(vis,str(r5[i][-3])+''+str(r5[i][-2]),(int(cx)-40,int(cy)+20), font, 0.8,(255,255,0),2,cv2.LINE_AA)
        
# %%    Fusion and splitting checking algorithm
       
        if r3 !=[]:
            numList_r3 = []
            numList_r4 = []
            list_r3_F = []
            dist_diff = []
            
            maxNum = max([r3[-1][3],r4[-1][3]])+1
            
            ### Look for any addition or reduction of contour in r3 and r4 
            numWithchanges = []
            for k in range(maxNum):
                numList_r3.append(0)
                numList_r4.append(0)
                for i in range(len(r3)):
                    if r3[i][3] == k:
                        numList_r3[-1] = numList_r3[-1]+1
                        
                    if r3[i][-2] == 'F':
                        list_r3_F.append(r3[i][3])
                
                for j in range(len(r4)):
                    if r4[j][2] == k:
                        numList_r4[-1] = numList_r4[-1]+1
                    
                    for l in range(len(list_r3_F)):
                        if list_r3_F[l] == r4[j][2]:
                            if r4[j][4] < 1.5:
                                r4[j][-2] = 'F'


                ###  To identify any fusing MT    
                if numList_r3[-1] > numList_r4[-1]:
                    # k is the one 
                    numWithchanges.append(k)
                    r4 = Match_F(k,r3,r4)
                    
                    
                ###  To identify any splitting MT
                elif numList_r3[-1] < numList_r4[-1]:
                    numWithchanges.append(k)
                    r4 = Match_S(k,r3,r4)

#%%            
        ## Store fusion events
        ## No fusion before splitting, add # of MT
        ## Find out the corresponding MT before fuse and after split                       
        for i in range(len(r4)): 
            for key in collection_MT:
                if collection_MT[key][-1][3] == r4[i][2]:
                    collection_MT[key].append(r4[i])
        
        ## Displaying useful information on the video
            cv2.putText(vis,str(r4[i][-3])+''+str(r4[i][-2]),(int(r4[i][0])-40,int(r4[i][1])+20), font, 0.5,(255,255,0),1,cv2.LINE_AA)    
        cv2.putText(vis,"Frame : "+str(int(cap.get(1)))+"/2000",(5,245), font, 0.6,(255,255,0),1,cv2.LINE_AA)

        r1 , r2 ,r3 , r4 , r5 = r2 , r3 , r4 , r5 , []        
        collection_r2.extend([r4])
        collection_velocity.extend([velocity])
        ## Show video
#        cv2.imshow('original',img)
        cv2.imshow('edited',vis)


        ## Output result to a csv file
        
        with open("collection_r2.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows(collection_r2[-1])
        with open("collection_velocity.csv", "a") as foo:
            boo = csv.writer(foo)
            boo.writerows(collection_velocity[-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
    
cap.release()
cv2.destroyAllWindows()
    




