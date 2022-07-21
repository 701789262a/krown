import json
import logging
import math
import pickle
import threading
from datetime import datetime

from sympy import Point3D, Plane, Line3D
import matlab.engine
import matplotlib.pyplot
import numpy
import numpy as np
from PIL import Image, ImageOps
from scipy import interpolate
from stl import mesh
import sklearn.cluster
import scipy
import vtk
import matplotlib.pyplot as plt
from pyoctree import pyoctree as ot
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

tangent_array = []
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
points=[]
x_dist=[]
y_dist=[]
def main():
    """Load STL for ray scan"""

    lineimg = Image.new('1', (5000, 5000))
    your_mesh = mesh.Mesh.from_file('lowercadinu.stl')

    im = Image.new('1', (5000, 5000))
    j = []
    #your_mesh.translate([37, 0, 27])
    """
    for triangle in your_mesh.data:
        # print(triangle[1][0][0])
        j.append(triangle[1][0][1])
        im.putpixel((int(triangle[1][0][0] * 50), int(triangle[1][0][2] * 50)), 1)
    imgarr = numpy.array(im.rotate(-90))
    func_centroid_x = []
    func_centroid_y = []
    c = 0
    for col in imgarr:
        arr = []
        x_c = 0
        for x in col:
            if x:
                arr.append([c, x_c])

            x_c += 1
        if len(arr) > 10:
            kmean = sklearn.cluster.KMeans(n_clusters=1, init='k-means++', random_state=0).fit(arr)
            if int(kmean.cluster_centers_[0][0]) % 500 == 0:
                func_centroid_x.append(int(kmean.cluster_centers_[0][0]))
                func_centroid_y.append(5000 - int(kmean.cluster_centers_[0][1]))
            # print(kmean.cluster_centers_[0])
            lineimg.putpixel((int(kmean.cluster_centers_[0][0]), int(kmean.cluster_centers_[0][1])), 1)
            # print(c,kmean)
        c += 1
    lineimg = ImageOps.mirror(lineimg)
    lineimg.save('cendroid.png')
    f = interpolate.CubicSpline(func_centroid_x, func_centroid_y)"""
    with open('spline_coords.txt') as f:
        j= json.load(f)
        x_spline_from_file = j['spline_coords']['x']
        y_spline_from_file = j['spline_coords']['y']
        x_spline_from_file.reverse()
        y_spline_from_file.reverse()
    f = interpolate.CubicSpline(x_spline_from_file, y_spline_from_file)
    #with open('interp_f.pck','wb')as file_handle:
    #    pickle.dump(f,file_handle)

    """with open('interp_f.pck', 'rb') as file_handle:
        f = pickle.load(file_handle)"""

    #print(min(func_centroid_x), max(func_centroid_x))
    im.save('robbone.png')
    xnew = np.arange(-50, 50, 100)
    ynew = f(xnew)
    ynew_der = f(xnew, 1)

    #plt.plot(xnew, ynew, '-')
    #plt.plot(xnew, ynew_der, '-')
    j = -6.2
    abut_height = j
    center_abut_x = 10.65
    radius_abut = 3
    error_abut = 0.3
    precision = 0.1
    steps = 50
    max_dist=6
    c = 0
    disarray_x = []
    disarray_y = []
    while True:
        if c > steps:
            break
        tangent_array = numpy.array([1, f(center_abut_x - radius_abut - error_abut - (precision * c), 1)])
        orthogonal_array = np.flip(tangent_array) * np.array([1, -1])
        xrange = np.linspace(1000, 2700, 100)
        """Aggiungere tangente a matplotlib"""
        # print(tangent_array, orthogonal_array)
        logging.info(
            "%s is evalued on X:%s;\tEvaluted on Y:%s" % (c, center_abut_x - radius_abut - error_abut + (precision * c),
                                                          f(center_abut_x - radius_abut - error_abut + (
                                                                  precision * c))))

        """plt.plot(xrange, tangent_func(xrange, center_abut_x - radius_abut - error_abut - (precision * c),
                                     f(center_abut_x - radius_abut - error_abut - (precision * c)), tangent_array),
                'C1--')
        plt.plot(xrange, tangent_func(xrange, center_abut_x - radius_abut - error_abut - (precision * c),
                                      f(center_abut_x - radius_abut - error_abut - (precision * c)), orthogonal_array),
                 'C1--')
        """
        """incident_point(your_mesh,
                       [orthogonal_array[0], j, orthogonal_array[1]],
                       [((center_abut_x - radius_abut - error_abut + (precision * c)) / 50) + 37, -6.2,
                        (f(center_abut_x - radius_abut - error_abut + (precision * c)) / 50) + 27])"""
        incident_point(your_mesh,
                       [orthogonal_array[0], 0, -orthogonal_array[1]],
                       [center_abut_x - radius_abut - error_abut - (precision * c),j,int(f(center_abut_x - radius_abut - error_abut - (precision * c)))],max_dist)

        logging.info('\n')

        c += 1
    d = interpolate.CubicSpline(list(reversed(x_dist)), list(reversed(y_dist)))
    d_x = numpy.arange(int(numpy.floor(x_dist[0])),int(numpy.ceil(x_dist[-1])),1)
    plt.plot(x_dist,d(d_x), color='red')
    plt.show()

#def compute_centers():


def tangent_func(xrange, x1, y1, arr):
    dir = (1 / (arr[0])) * (arr[1])
    return (dir * (xrange - x1)) + y1


def incident_point(stl, arr, application_point,max_dist):
    c = 0
    eng = matlab.engine.start_matlab()
    for plane in stl.data:
        #plane_normal = Plane(Point3D(plane[1][0], evaluate=False), Point3D(plane[1][1], evaluate=False),
        #                     Point3D(plane[1][2], evaluate=False), evaluate=False)
        if numpy.linalg.norm(numpy.array(plane[1][0])-numpy.array(application_point))<max_dist:
            threading.Thread(target=threaded,args=(plane, application_point,arr,eng,c,)).start()
        if len(points)==2:
            break
        c += 1
    if len(points)==0:
        with open('dis.txt', 'a') as d:
            d.write(str(0) + '\n')
        x_dist.append(application_point[0])
        y_dist.append(0)
    if len(points)==2:
        with open('dis.txt', 'a') as d:
            d.write(str(numpy.linalg.norm(numpy.array(points[0]) - numpy.array(points[1]))) + '\n')
        x_dist.append(application_point[0])
        y_dist.append(numpy.linalg.norm(numpy.array(points[0]) - numpy.array(points[1])))
        points.pop()
        points.pop()
    logging.info("no intersection found!")


def threaded(plane, application_point,arr,eng,c):
    plane_normal = plane[0]
    # line = Line3D(application_point, direction_ratio=array)
    # intr = plane_normal.intersection(line)

    intersection = line_plane_inter(np.array(arr).tolist(),
                                               np.array(application_point).tolist(),
                                               [float(plane_normal[0]),
                                                float(plane_normal[1]),
                                                float(plane_normal[2])],
                                               np.array(plane[1][0]).tolist())
    polygon = Polygon([tuple(plane[1][0]), tuple(plane[1][1]), tuple(plane[1][2])])
    logging.info(c)
    logging.info(
        f"{arr} applicated on {application_point} intersect {plane_normal} on {intersection};")


    if polygon.contains(Point(float(intersection[0]),float(intersection[1]),float(intersection[2]))) and distance([float(intersection[0]),float(intersection[1]),float(intersection[2])],application_point)<20 :
        with open('res', 'a') as w:
            w.write(
                f"{arr} applicated on {application_point} intersect {plane_normal} on {intersection} within boundaries 2 {str(datetime.now())}\n")
        points.append(intersection)

def point_on_triangle(point, p1,p2,p3):
    A = area(numpy.array(p1)-numpy.array(p2),numpy.array(p1)-numpy.array(p3))
    A1 = area(numpy.array(p1)-numpy.array(point),numpy.array(p1)-numpy.array(p3))
    A2 = area(numpy.array(p2)-numpy.array(point),numpy.array(p2)-numpy.array(p3))
    A3 = area(numpy.array(p1)-numpy.array(p2),numpy.array(p1)-numpy.array(point))
    if A1 + A2 + A3 == A:
        return True
    return False


def distance(v1,v2):
    distance_array=numpy.array(v1)-numpy.array(v2)
    return numpy.linalg.norm(distance_array)

def area(v1,v2):
    return numpy.linalg.norm(abs(numpy.cross(v1,v2)))/2

def line_plane_inter(u,N,n,M):
    d=-numpy.dot(n,M)
    if not(numpy.dot(n,N)):
        if not numpy.dot(n,N)+d==0:
            return []
        else:
            return M
    else:
        t=-(d+numpy.dot(n,N))/numpy.dot(n,u)
        I=numpy.array(N)+numpy.array(u)*t
        return I

if __name__ == '__main__':
    main()
