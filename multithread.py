# USAGE
# python write_to_video.py --output example.avi
# python write_to_video.py --output example.avi --picamera 1

# import the necessary packages
# from __future__ import print_function
# from __future__ import print_function
from imutils.video import VideoStream
# import numpy as np
import argparse
import imutils
import time
from pylibdmtx.pylibdmtx import decode
import cv2
import numpy as np
import zbar
import math
scanner = zbar.Scanner()
count = 0


# def rotate_about_center(src,rect,scale=1):
#     w = src.shape[1]
#     h = src.shape[0]
#     angle = rect[2]
#     rangle = np.deg2rad(angle)  # angle in radians
#     # print "angle",angle
#     ## now calculate new image width and height
#     nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
#     nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
#     ## ask OpenCV for the rotation matrix
#     rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
#     ## calculate the move from the old center to the new center combined
#     ## with the rotation
#     rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
#     ## the move only affects the translation, so update the translation
#     ## part of the transform
#     rot_mat[0,2] += rot_move[0]
#     rot_mat[1,2] += rot_move[1]
#     img_rot = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
#     rect0 = (rect[0], rect[1], 0.0)
#     box = cv2.boxPoints(rect)
#     # case = np.array(box, dtype="int")
#     pts = np.array(cv2.transform(np.array([box],dtype=int), rot_mat))[0]
#     pts[pts < 0] = 0
#     # crop
#     img_crop = img_rot[pts[1][1]:pts[0][1],
#                        pts[1][0]:pts[2][0]]
#     return img_crop,box


def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    # print "angle",angle
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    # print "M", type(M)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # print "hey"
    ## rotate bounding box
    # rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    # case = np.array(box, dtype="int")
    pts = np.array(cv2.transform(np.array([box],dtype=int), M))[0]
    pts[pts < 0] = 0

    ## crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop,box


def process(image):
    # colour = image.copy()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("crop",image)
    org = image.copy()
    #  # compute the Scharr gradient magnitude representation of the images
    # # in both the x and y direction
    # gradX = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    # gradY = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    #
    # # subtract the y-gradient from the x-gradient
    # gradient = cv2.subtract(gradX, gradY)
    # image = cv2.convertScaleAbs(gradient)
    # cv2.imshow("gradient", image)

    # image = cv2.blur(image, (9, 9))
    # cv2.imshow("blur", blurred)
    (_, thresh) = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh.copy())


    ## close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("MorphClosed", closed.copy())

    ## perform a series of erosions and dilations
    closed = cv2.dilate(closed, None, iterations = 7)
    closed = cv2.erode(closed, None, iterations = 4)
    # closed = cv2.dilate(closed, None, iterations = 7)
    # cv2.imshow("Erode Dilate", closed.copy())

    ## find the contours in the thresholded image, theqn sort the contours
    ##  by their area, keeping only the largest one
    (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # print "c",type(c)
    ## compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    cropped_img,box = crop_minAreaRect(org, rect)
    if len(cropped_img)==0:
        return None
    # cropped_img,box = rotate_about_center(org,rect)
    ## reject rectangles with length abd breadth less than dimen
    dimen = cv2.boundingRect(c)
    if dimen[3] < 5 or dimen[2]<5:
        print "samll area"
        return None
    # cropped_img = org[dimen[1]:(dimen[1] + dimen[3]), dimen[0]:(dimen[0] + dimen[2])]
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="int")
    # np.float32(frame)
    # print type(cropped_img)
    # cropped_img=np.float32(cropped_img)
    # angle = -(90+rect[2])
    # angle = 0
    # print type(cropped_img)
    # print "box", box
    ## draw a bounding box arounded the detected  barcode and display the
    ##  image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imshow("image",cropped_img)
    cv2.imshow("crop",image)
    ans = detect(cropped_img)
    if len(cropped_img) !=0:
        ans = detect(cropped_img)
        return ans
    else: return False


# def detect(frame):
#     ans = decode(frame)
#     if len(ans) == 0:
#         ans = scanner.scan(frame)
#         if len(ans) != 0:
#             print "Decode Barcode", ans
#             return True
#         else:
#             return False
#     print "Decode 2D barcode", ans
#     return True

def detect(frame):
    ##using pylibdmtx
    ans = decode(frame)
    if len(ans)!=0:
        print "Decode 2D Barcode", ans
        return True
    ##using zbar
    if len(ans)==0:
        ans = scanner.scan(frame)
        if len(ans) ==0:
            return False
        print "Decode barcode", ans
        return True
    return False

## construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())


## initialize the video stream and allow the cameraq
## sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)


## loop over frames from the video stream
while True:
    ## grab the frame from the video stream and resize it to have a
    frame = vs.read()
    frame = imutils.resize(frame, width=1080)
    # ans = detect(frame)
    # print "decode",ans
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("crop",image)
    print frame.shape()
    ans = process(frame)
    if ans == None:
        pass
    if ans != False:
        global count
        count += 1
        print count
        # print "decode",ans
    else: print "False"    # print "decode",ans

    # show the frames
    # cv2.imshow("Frame", frame)
    # cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    ## if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

## do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
