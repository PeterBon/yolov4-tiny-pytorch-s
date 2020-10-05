#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import time

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        t1 = time.time()
        r_image = yolo.detect_image(image)
        t2 = time.time()
        print(t2-t1)
        r_image.show()
