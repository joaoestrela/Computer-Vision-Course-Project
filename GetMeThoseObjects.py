import cv2
import numpy as np
import argparse
import ntpath

# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
    length=100, fill=u'\u2588', header=''):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        header      - Optional  : header string (Str)
    """
    # Clear the current line and print the header
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    # Generate and print the bar
    bar = fill * filledLength + u'-' * (length - filledLength)
    print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Lines on Complete
    if iteration == total:
        print("Your " + input_image_file_name+' has been separeted with success !!!')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-o", "--output", required = False, default="./resultObjects/", help = "Output folder for objects images")
args = vars(ap.parse_args())
total_steps = 6
step = 0
input_image_dir = ntpath.dirname(args["image"])
input_image_file_name = ntpath.basename(args["image"])
output_image_dir = args["output"]
imageFormat = input_image_file_name.split(".")[1]
input_image_file_name = input_image_file_name.split(".")[0]
print("You are using GetMeThoseObjects 1.0 by Joao Estrela @ DCC.FC.UP.PT")

img = cv2.imread(args["image"],-1)
height, width, channels = img.shape
step +=1
printProgressBar(step, total_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Getting top left corner for sampleing background
cornerSample = img[0:int(min(height,width)*0.15),0:int(min(height,width)*0.15)]
hsv_corner = cv2.cvtColor(cornerSample,cv2.COLOR_BGR2HSV)
hueMax = hsv_corner[:,:,0].max()
hueMin = hsv_corner[:,:,0].min()
lowerBound = np.array([hueMin-15,75,75], np.uint8)
upperBound = np.array([hueMax+15,255,255], np.uint8)
step +=1
printProgressBar(step, total_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Making a mask thresholding based on the corner sample
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_img,lowerBound,upperBound)
mask = cv2.bitwise_not(mask,mask)
step +=1
printProgressBar(step, total_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Using some morphological transformations to get a cleaner mask with less noise
kernel = np.ones((3,3),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
step +=1
printProgressBar(step, total_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Finding contours on mask to get diferent objetcs
im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in contours]
step +=1
printProgressBar(step, total_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Making each object its own image
objectN = 0
for i in range(len(areas)):
    # Only using good contours that have a good area, sorry small objects you will be missed :(
    if(areas[i] > 50*50):
        objectN +=1
        cnt=contours[i]
        approxMask = np.ones((height,width), np.uint8)
        cv2.drawContours(approxMask, [cnt], -1,255, -1, lineType = cv2.LINE_AA)
        approxMask = cv2.bitwise_and(approxMask,mask)
        x,y,w,h = cv2.boundingRect(cnt)
        res = cv2.merge((img[:,:,0],img[:,:,1],img[:,:,2],approxMask))
        cv2.imwrite(output_image_dir+"object"+str(objectN)+"_"+input_image_file_name+".png",res[y:y+h,x:x+w])
step +=1
printProgressBar(step, total_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)
