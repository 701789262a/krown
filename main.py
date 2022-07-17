import datetime

import numpy
from stl import mesh
from PIL import Image, PngImagePlugin
import cv2
import threading

PngImagePlugin=True

centers=[]

def main():
    im = Image.new('RGB',(5000,5000))
    your_mesh = mesh.Mesh.from_file('lowercadinu.stl')
    height=[]
    for triangle in your_mesh.data:
        #print(triangle[1][0][0])
        height.append(triangle[1][0][1])
        #norm = numpy.linalg.norm(triangle[0])
        #area=0.5*numpy.multiply(numpy.linalg.norm(numpy.array(triangle[1][0])-numpy.array(triangle[1][1])),numpy.linalg.norm(numpy.array(triangle[1][0])-numpy.array(triangle[1][2])))
        #im.putpixel((int(triangle[1][0][0]*100),int(triangle[1][0][2]*100)),(int(norm*17000),int(norm*17000),int(norm*17000),255))
        #print(area)
    step=0.5
    for j in numpy.arange(min(height),max(height),step):
        s=threading.Thread(target=huges,args=(j,your_mesh,step))
        s.start()

    while True:
        if len(centers)>=2:
            print(centers)
            v0=numpy.array(centers[1])-numpy.array(centers[0])
            v1 = numpy.array(centers[1])-numpy.array(centers[0])
            v1[0]=0
            v2 = numpy.array(centers[1])-numpy.array(centers[0])
            v2[2]=0
            print(v0,v1,v2)
            unit_vector_1 = numpy.array(v1) / numpy.linalg.norm(v1)
            dot_product = numpy.dot(unit_vector_1, [0, 1, 0])
            angle = numpy.arccos(dot_product)
            print("angle v1",str(angle))
            unit_vector_2 = numpy.array(v2) / numpy.linalg.norm(v2)
            dot_product = numpy.dot(unit_vector_2, [0, 1, 0])
            angle = numpy.arccos(dot_product)
            print("angle v2",str(angle))
            break
    #print(your_mesh.data,len(your_mesh),your_mesh.get_header(''))
    #print(max(height),min(height))
    #mesh.Mesh.save(your_mesh,'meshaggiornata.stl')
    im.save('culo212.png')


def huges(j,your_mesh,step):
    im = Image.new('RGB', (5000, 5000))
    print(j)
    for triangle in your_mesh.data:
        if triangle[1][0][1] < j and triangle[1][0][1] > j - step:
            im.putpixel((int(triangle[1][0][0] * 100), int(triangle[1][0][2] * 100)), (255, 255, 255, 255))
            im.putpixel((int(triangle[1][0][0] * 100), int(triangle[1][0][2] * 100) + 1), (255, 255, 255, 255))
            im.putpixel((int(triangle[1][0][0] * 100) + 1, int(triangle[1][0][2] * 100)), (255, 255, 255, 255))
            im.putpixel((int(triangle[1][0][0] * 100) + 1, int(triangle[1][0][2] * 100) + 1), (255, 255, 255, 255))
            im.putpixel((int(triangle[1][0][0] * 100) + 2, int(triangle[1][0][2] * 100) + 2), (255, 255, 255, 255))
            im.putpixel((int(triangle[1][0][0] * 100) + 2, int(triangle[1][0][2] * 100) + 1), (255, 255, 255, 255))
            im.putpixel((int(triangle[1][0][0] * 100) + 1, int(triangle[1][0][2] * 100) + 2), (255, 255, 255, 255))
            im.putpixel((int(triangle[1][0][0] * 100), int(triangle[1][0][2] * 100) + 2), (255, 255, 255, 255))
            im.putpixel((int(triangle[1][0][0] * 100) + 2, int(triangle[1][0][2] * 100)), (255, 255, 255, 255))
    dt = str(int(datetime.datetime.timestamp(datetime.datetime.now())))

    # im.save(dt+str(y)+'.png')
    # img=cv.imread(dt+str(y)+'.png')
    # im.show()
    numpy_image = numpy.array(im)
    img = numpy_image[:, :, ::-1].copy()
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 1)
    gray = cv2.Canny(gray, 1, 1)

    # gray=cv.medianBlur(gray,5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2.2, 500, param1=10, param2=75, minRadius=100, maxRadius=120)
    try:
        detected_circles = numpy.uint16(numpy.around(circles))
        for (x, y, r) in detected_circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
            cv2.circle(output, (x, y), 2, (0, 255, 255), 3)
            centers.append([x,j*100,y])
        cv2.imwrite(f'outcv2proc-' + str(j).replace('.', '-') + '.png', output)
        # cv.imshow('output',output)
        # cv.waitkey(0)
        # cv.destroyAllWindows()
    except:
        pass


if __name__=='__main__':
    main()