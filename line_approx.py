import logging
import math
import threading

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


def main():
    """Load STL for ray scan"""
    im = Image.new('1', (5000, 5000))
    lineimg = Image.new('1', (5000, 5000))
    your_mesh = mesh.Mesh.from_file('lowercadinu.stl')

    j = []
    your_mesh.translate([37, 0, 27])
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
    f = interpolate.CubicSpline(func_centroid_x, func_centroid_y)
    print(min(func_centroid_x), max(func_centroid_x))
    im.save('robbone.png')
    xnew = np.arange(450, 3350, 20)
    ynew = f(xnew)
    ynew_der = f(xnew, 1)

    plt.plot(xnew, ynew, '-')
    plt.plot(xnew, ynew_der, '-')
    j = -6.2
    abut_height = j
    center_abut_x = 2382
    radius_abut = 110
    error_abut = 7
    precision = 20
    steps = 200
    c = 0
    disarray_x = []
    disarray_y = []
    while True:
        if c > steps:
            break
        tangent_array = numpy.array([1, f(center_abut_x - radius_abut - error_abut + (precision * c), 1)])
        orthogonal_array = np.flip(tangent_array) * np.array([1, -1])
        xrange = np.linspace(1000, 2700, 100)
        """Aggiungere tangente a matplotlib"""
        # print(tangent_array, orthogonal_array)
        logging.info(
            "%s is evalued on X:%s;\tEvaluted on Y:%s" % (c, center_abut_x - radius_abut - error_abut + (precision * c),
                                                          f(center_abut_x - radius_abut - error_abut + (
                                                                  precision * c))))

        # plt.plot(xrange, tangent_func(xrange, center_abut_x - radius_abut - error_abut - (precision * c),
        #                              f(center_abut_x - radius_abut - error_abut - (precision * c)), tangent_array),
        #         'C1--')
        # plt.plot(xrange, tangent_func(xrange, center_abut_x - radius_abut - error_abut - (precision * c),
        #                              f(center_abut_x - radius_abut - error_abut - (precision * c)), orthogonal_array),
        #         'C1--')
        incident_point(your_mesh, [-orthogonal_array[0], j, -orthogonal_array[1]],
                       [((-center_abut_x - radius_abut - error_abut + (precision * c)) / 50) + 37, -6.2,
                        (-f(center_abut_x - radius_abut - error_abut + (precision * c)) / 50) + 27])
        computational_vtk_mesh = mesh.Mesh.from_file('lowercadinu.stl')
        computational_vtk_mesh.translate([((-center_abut_x - radius_abut - error_abut + (precision * c)) / 50) + 37, 0,
                                          (-f(center_abut_x - radius_abut - error_abut + (precision * c)) / 50) + 27])
        computational_vtk_mesh.save(str(c) + '.stl')
        tree = generate_tree(c)
        # plt.show()
        # break
        # application_point = [center_abut_x - radius_abut - error_abut - (precision * c), j,
        #                     f(center_abut_x - radius_abut - error_abut - (precision * c))]
        # applicated_array = orthogonal_array + numpy.array(application_point)
        # rayPointList = np.array([[[x, y, zs], [x, y, ze]]], dtype=np.float32)
        # print(rayPointList)
        # break

        ray = \
            numpy.array(
                [[[-orthogonal_array[0], j, -orthogonal_array[1]], [orthogonal_array[0], j, orthogonal_array[1]]]],
                dtype=np.float32)[0]
        logging.info("Ray:%s" % ray)
        i = tree.rayIntersection(ray)
        if len(i) == 2:
            logging.info("Intersection in P1:%s;\tP2:%s" % (i[0].p, i[1].p))
            dis = math.dist(i[0].p, i[1].p)
            disarray_x.append(c)
            disarray_y.append(dis)
            logging.info("Distance:%s" % dis)
        else:
            logging.info("Intersection not found!")
            disarray_x.append(c)
            disarray_y.append(0)
        logging.info('\n')

        c += 1
    plt.plot(disarray_x, disarray_y, color='red')
    plt.show()


def tangent_func(xrange, x1, y1, arr):
    dir = (1 / (arr[0])) * (arr[1])
    return (dir * (xrange - x1)) + y1


def incident_point(stl, array, application_point):
    c=0
    for plane in stl.data:
        threading.Thread(target=threaded_point_finder, args=(c, plane, array, application_point)).start()

        c += 1
    logging.info("no intersection found!")

def threaded_point_finder(c,plane, array, application_point):
    eng=matlab.engine.start_matlab()
    plane_normal = Plane(Point3D(plane[1][0],evaluate=False), Point3D(plane[1][1],evaluate=False), Point3D(plane[1][2],evaluate=False),evaluate=False)
    line = Line3D(application_point, direction_ratio=array)
    #intr = plane_normal.intersection(line)
    intersection = eng.line_plane_intersection(array,application_point,plane_normal.normal_vector,plane[1][0])
    polygon = Polygon([tuple(plane[1][0]), tuple(plane[1][1]), tuple(plane[1][2])])
    logging.info(c)
    logging.info(
        f"{array} applicated on {application_point} intersect {plane_normal.normal_vector} on {intersection}; point on plane {plane[1][0]}")
    if polygon.contains(Point(intersection)):
        logging.info(
            f"{array} applicated on {application_point} intersect {plane_normal.normal_vector} on {intersection} within boundaries")
        with open('res','a') as w:
            w.write(f"{array} applicated on {application_point} intersect {plane_normal.normal_vector} on {intersection} within boundaries\n")

def generate_tree(c):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(c) + ".stl")
    reader.MergingOn()
    reader.Update()
    stl = reader.GetOutput()

    numPoints = stl.GetNumberOfPoints()
    pointCoords = np.zeros((numPoints, 3), dtype=float)
    for i in range(numPoints):
        pointCoords[i, :] = stl.GetPoint(i)

    # 2. Get polygon connectivity
    numPolys = stl.GetNumberOfCells()
    connectivity = np.zeros((numPolys, 3), dtype=np.int32)
    for i in range(numPolys):
        atri = stl.GetCell(i)
        ids = atri.GetPointIds()
        for j in range(3):
            connectivity[i, j] = ids.GetId(j)

    tree = ot.PyOctree(pointCoords, connectivity)
    return tree


if __name__ == '__main__':
    main()
