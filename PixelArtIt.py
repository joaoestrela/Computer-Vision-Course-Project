# -*- coding: utf-8 -*-
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
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
        print("Your " + input_image_file_name+' has been pixelated with success !!!')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-o", "--output", required = False, default="_", help = "Output to the image")
ap.add_argument("-s", "--saturation", required = False, nargs='?', default=1.25, type = int, help = "% of saturation change")
ap.add_argument("-c", "--clusters", required = False, nargs='?', default=8, type = int, help = "# of clusters")
ap.add_argument("-d", "--downscaling", required = False, nargs='?', default=0.5, type = float, help = "% of downscaling")
args = vars(ap.parse_args())

input_image_dir = ntpath.dirname(args["image"])
input_image_file_name = ntpath.basename(args["image"])
output_image_dir = ntpath.dirname(args["output"])
output_image_file_name = ntpath.basename(args["output"])
if(output_image_file_name == "_"): output_image_file_name = input_image_file_name.split(".")[0]+"_"
print("You are using PixelArtIt 1.0 by Joao Estrela @ DCC.FC.UP.PT")
print("Your " + input_image_file_name+' is going to be pixelated !!!')
printProgressBar(1, 6, prefix = 'Progress:', suffix = 'Complete', length = 50)
img = cv2.imread(args["image"])
height, width, channels = img.shape

printProgressBar(2, 6, prefix = 'Progress:', suffix = 'Complete', length = 50)
# SATURATION - Giving the image a more vivid look for easier quantitization
saturated = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
for i in range(0, height):
    for j in range(0, width):
        if(saturated[i,j,1] * args["saturation"] < 0):
            saturated[i,j,1] = 0
        elif(saturated[i,j,1] * args["saturation"] > 255):
            saturated[i,j,1] = 255;
        else:
            saturated[i,j,1] = int(saturated[i,j,1] * args["saturation"])
saturated = cv2.cvtColor(saturated, cv2.COLOR_HSV2RGB)
cv2.imwrite(output_image_dir+output_image_file_name+"saturated.jpg", saturated)

printProgressBar(3, 6, prefix = 'Progress:', suffix = 'Complete', length = 50)
# QUANTITIZATION BY K-CLUSTEING - Making only the k most relevant colors appear
quant = cv2.cvtColor(saturated, cv2.COLOR_BGR2LAB)
quant = quant.reshape((quant.shape[0] * quant.shape[1], 3))
clt = MiniBatchKMeans(args["clusters"])
labels = clt.fit_predict(quant)
quant = clt.cluster_centers_.astype("uint8")[labels]
quant = quant.reshape((height, width, 3))
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
cv2.imwrite(output_image_dir+output_image_file_name+"quantized.jpg",quant)

printProgressBar(4, 6, prefix = 'Progress:', suffix = 'Complete', length = 50)
# Nearest Neighbor Downscaling - Giving a pixelated look
downscaled = cv2.resize(quant,(int(width*args["downscaling"]),int(height*args["downscaling"])), interpolation = cv2.INTER_NEAREST)
cv2.imwrite(output_image_dir+output_image_file_name+"downscaled.jpg",downscaled)
printProgressBar(5, 6, prefix = 'Progress:', suffix = 'Complete', length = 50)
# Nearest Neighbor Upscaling - Returning to normal size while preserving the pixelated look
res = cv2.resize(downscaled,(width,height), interpolation = cv2.INTER_NEAREST)
printProgressBar(6, 6, prefix = 'Progress:', suffix = 'Complete', length = 50)
#Saving the result
cv2.imwrite(output_image_dir+output_image_file_name+"output.jpg",res)
