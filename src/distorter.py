"""
    DPRL Math Symbol Recognizers 
    Copyright (c) 2012-2014 Kenny Davila, Richard Zanibbi

    This file is part of DPRL Math Symbol Recognizers.

    DPRL Math Symbol Recognizers is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DPRL Math Symbol Recognizers is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DPRL Math Symbol Recognizers.  If not, see <http://www.gnu.org/licenses/>.

    Contact:
        - Kenny Davila: kxd7282@rit.edu
        - Richard Zanibbi: rlaz@cs.rit.edu 
"""
import numpy as np
import ctypes
import scipy.ndimage as ndimage
from mathSymbol import *

#=====================================================================
#  Generates distorted versions of samples
#
#  Created by:
#      - Kenny Davila (Jan 14, 2012-2014)
#  Modified By:
#      - Kenny Davila (Jan 14, 2012-2014)
#      - Kenny Davila (Jan 20, 2012-2014)
#
#=====================================================================

distorter_lib = ctypes.CDLL('./distorter_lib.so')
distorter_lib.distorter_init()


class Distorter:
    def __init__(self):
        self.noise_size = 64  # 128
        self.noise_map_size = 8
        self.noise_map_depth = 6

    def getBoundingBox(self, points):
        min_x = points[0][0]
        max_x = points[0][0]
        min_y = points[0][1]
        max_y = points[0][1]

        for i in range(1, len(points)):
            x, y = points[i]

            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

        return (min_x, max_x, min_y, max_y)

    def distortPoints(self, points, max_diagonal):
        new_points = []

        #check original box...
        min_x, max_x, min_y, max_y = self.getBoundingBox(points)
        w = max_x - min_x
        h = max_y - min_y

        if w == 0.0:
            w = 0.000001

        if h == 0.0:
            h = 0.000001

        line_dist = math.sqrt(w ** 2 + h ** 2) * max_diagonal

        #Get noise maps...
        #...x....
        #noise_x = self.getNoiseMap(self.noise_size, self.noise_map_size, self.noise_map_depth)
        noise_x = np.zeros((self.noise_size, self.noise_size))
        noise_x_p = noise_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        distorter_lib.distorter_create_noise_map(noise_x_p, self.noise_size, self.noise_map_size, self.noise_map_depth)
        #...smooth x....
        noise_x = ndimage.gaussian_filter(noise_x, sigma=(2, 2), order=0)

        #...y....
        #noise_y = self.getNoiseMap(self.noise_size, self.noise_map_size, self.noise_map_depth)
        noise_y = np.zeros((self.noise_size, self.noise_size))
        noise_y_p = noise_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        distorter_lib.distorter_create_noise_map(noise_y_p, self.noise_size, self.noise_map_size, self.noise_map_depth)
        #...smooth y....
        noise_y = ndimage.gaussian_filter(noise_y, sigma=(2, 2), order=0)

        #now, apply distortion....
        #...for each point....
        for i in range(len(points)):
            x, y = points[i]

            #compute relative position...
            p_x = (x - min_x) / w
            p_y = (y - min_y) / h

            #compute noise position...
            p_nx = int(p_x * self.noise_size)
            p_ny = int(p_y * self.noise_size)
            if p_nx >= self.noise_size:
                p_nx = self.noise_size - 1
            if p_ny >= self.noise_size:
                p_ny = self.noise_size - 1

            #get distortion...
            dist_x = noise_x[p_nx, p_ny]
            dist_y = noise_y[p_nx, p_ny]


            #print "Pre: (" + str(dist_x) + ", " + str(dist_y) + ")"
            """
            #normalize...
            norm = math.sqrt(dist_x * dist_x + dist_y * dist_y)
            if norm > 0.0:
                dist_x /= norm
                dist_y /= norm
            else:
                dist_x = 0.0
                dist_y = 0.0
            """
            #print "Post: (" + str(dist_x) + ", " + str(dist_y) + ")"


            off_x = dist_x * line_dist
            off_y = dist_y * line_dist

            new_x = x + off_x
            new_y = y + off_y

            new_points.append((new_x, new_y))

        return new_points

    #create a distorted version of the given symbol...
    def distortSymbol(self, symbol, max_diagonal):

        #create distorted traces...
        all_traces = []
        for t in symbol.traces:
            new_points = self.distortPoints(t.original_points, max_diagonal)

            new_trace = TraceInfo(t.id, new_points)

            #try smoothing the distorted version...
            new_trace.removeDuplicatedPoints()

            #Add points to the trace...
            new_trace.addMissingPoints()

            #Apply smoothing to the trace...
            new_trace.applySmoothing()

            #it should not ... but .....
            if new_trace.hasDuplicatedPoints():
                #...remove them! ....
                new_trace.removeDuplicatedPoints()

            all_traces.append(new_trace)

        #now, create the new symbol...
        new_symbol = MathSymbol(symbol.id, all_traces, symbol.truth)
        new_symbol.normalize()

        return new_symbol
