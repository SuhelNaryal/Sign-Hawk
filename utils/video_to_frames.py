import cv2
import os

files = ['test3.mp4', 'test4.mp4']

for file in files:
    videostream = cv2.VideoCapture(file)
    fname = file.rsplit('.', maxsplit=1)[0]
    os.makedirs(fname)
    count = 0
    while True:
        ret, frame = videostream.read()
        if not ret:
            break
        if count%5 == 0:
            cv2.imwrite(fname+"/"+str(count)+".jpg", frame)
        count += 1