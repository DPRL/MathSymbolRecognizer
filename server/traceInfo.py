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
import math

#=====================================================================
#  This class represents a traces, and also performs all the operations
#  relative to it
#
#  Created by:
#      - Kenny Davila (Oct, 2012)
#  Modified By:
#      - Kenny Davila (Oct 19, 2013)
#      - Kenny Davila (Jan 14, 2012-2014)
#        - Added "original_points" field
#        - Added function to swap between original and processed points
#      - Kenny Davila (Jan 15, 2012-2014)
#        - Aspect Ratio is now normalized
#        - Orientation Hist. accept now non-squared grids
#        - Grid of distances now computes the average distance instead of min
#      - Kenny Davila (Jan 16, 2012-2014)
#        - 2D histogram now acepts multiple non-squared grids
#
#=====================================================================
#
# References to papers for the original methods....
# [1] - Preprocessing Techniques for Online Handwriting Recognition,
#      B.Q. Huang and Y.B. Zhang and M-T. Kechadi
#
#
#=====================================================================

class TraceInfo:    
    #constructor
    def __init__(self, trace_id, trace_points):
        self.id = trace_id
        self.points = trace_points
        self.sharp_points = None
        self.bounding_box = None
        self.segments = None
        
        self.original_points = list(trace_points)

    #================================================================
    #  swap between original and current points
    #  warning: This leaves the trace on a very unconsistent state
    #           and should be used only for debugging purposes...
    #================================================================
    def swapPoints(self):
        tempo = self.points
        self.points = self.original_points
        self.original_points = tempo

        #out of date...
        self.bounding_box = None

    #Gets current boundaries for the trace
    def getBoundaries(self):
        #only compute if it has never been computer or if it has changed...
        if self.bounding_box == None:
            #assume first point to define all boundaries initially..
            minX, minY = self.points[0]
            maxX, maxY = self.points[0]
    
            #for each point...
            for x, y in self.points:
                if x < minX:
                    minX = x
                if x > maxX:
                    maxX = x
                if y < minY:
                    minY = y
                if y > maxY:
                    maxY = y
            
            self.bounding_box = (minX, maxX, minY, maxY) 

        return self.bounding_box    
            
    #Checks for duplicated points
    def hasDuplicatedPoints(self):
        duplicated = False
        for i in range(len(self.points) - 1):
            for j in range(i + 1, len(self.points)):
                if self.points[i][0] == self.points[j][0] and \
                   self.points[i][1] == self.points[j][1]:
                    duplicated = True
        return duplicated

    #Convert to string representation
    def __str__(self):
        result = ''
        
        for x, y in self.points:
            result += str(x) + "," + str(y) + "\r\n" 
            
        return result

    #remove duplicated points (pre processing)
    def removeDuplicatedPoints(self):

        minX, maxX, minY, maxY = self.getBoundaries()
        w = maxX - minX
        h = maxY - minY
        diagonal = math.sqrt( w * w + h * h )
        
        i = 0
        while i < len(self.points):
            j = i + 1
            while j < len(self.points):
                if self.distance(i, j) > diagonal * 0.1:
                    break
                
                if self.points[i] == self.points[j]:
                    del self.points[j]
                else:
                    j += 1
            i += 1            
    
    #Add points where there are missing points (pre processing)
    def addMissingPoints(self):
        #to avoid problems....
        self.removeDuplicatedPoints()
        
        #calculate Le (average segment length) as defined in [1]
        Le = 0;
        for i in range(len(self.points) - 1):
            Le += self.distance( i, i + 1)
        Le /= len(self.points)
        
        #the distance used to insert points...
        d = 0.95 * Le
        i = 0        
        while i < len(self.points) - 1:
            #search point to interpolate ...
            n = 1
            lenght = self.distance(i, i + n)
            sum = 0
            
            while sum + lenght < d and i + n + 1 < len(self.points):
                n += 1
                sum += lenght
                lenght = self.distance(i + n - 1, i + n)
                    
            diff = d - sum 
                                    
            #insert a point between i + n- 1 and i +n at distance diff                                
            #use linear interpolation...                                        
            w2 = diff / lenght            
            
            if w2 < 1.0:
                xp = self.points[i + n - 1][0] * (1- w2) + self.points[i + n][0] * w2
                yp = self.points[i + n - 1][1] * (1- w2) + self.points[i + n][1] * w2                  
                
                #check for collision with next point...
                insert = True
                                                
                if i + n < len(self.points):
                    if xp == self.points[i + n][0] and \
                       yp == self.points[i + n][1]:
                        #weird case where a point after interpolated falls of the same 
                        #coordinates as next point, don't insert it
                        insert = False                                            
                             
                                                       
                if insert:
                    self.points.insert(i + n, (xp, yp) )                
                        
            else:
                #at the end, no point added but erase the one at the end
                n += 1
            
            #now erase points from i + 1 .. f + n - 1
            toErase = n - 1
            for j in range(toErase):
                del self.points[i + 1]  
                
            i += 1        
        
    #returns the distance between two points in the curve
    def distance(self, i, j):
        x1, y1 = self.points[i]
        x2, y2 = self.points[j]
        
        return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2)) 

    #smooths the curve (pre-processing)
    def applySmoothing(self):
        #Detect sharp points and extract them....
        sharp_points = self.getSharpPoints()
        
        #Remove hooks using sharp points information
        self.removeHooks(sharp_points)
        
        #Finally re-sample
        self.splineResample(sharp_points, 2)
        
        #bounding box might have changed...
        self.bounding_box = None

    #removes the hooks at the beginning and at the end (pre-processing)
    def removeHooks(self, sharp_points):
        if len(sharp_points) >= 4:
            #remove hooks....
            
            #Check for hooks in the segment b (beginning)
            #and at the segment e (ending)
            
            #get the diagonal length....
            minX, maxX, minY, maxY  = self.getBoundaries()
            ld = math.sqrt( math.pow(maxX - minX, 2) + math.pow(maxY - minY, 2) )
            
            #at beginning
            betha_b0 =  self.slopeAngle(sharp_points[0][1], sharp_points[1][1])
            betha_b1 = self.slopeAngle(sharp_points[1][1], sharp_points[2][1]);            
            lseg_b = self.distance(sharp_points[0][0], sharp_points[1][0])
            lambda_b = self.angularDifference(betha_b0, betha_b1)
            
            #At ending
            betha_e0 =  self.slopeAngle(sharp_points[-1][1], sharp_points[-2][1])
            betha_e1 = self.slopeAngle(sharp_points[-2][1], sharp_points[-3][1]);            
            lseg_e = self.distance(sharp_points[-1][0], sharp_points[-2][0])
            lambda_e = self.angularDifference(betha_e0, betha_e1)
            
            if lambda_e > math.pi / 4 and lseg_e < 0.07 * ld:
                #remove sharp point at the end...
                del sharp_points[-1]
            
            if lambda_b > math.pi / 4 and lseg_b < 0.07 * ld:
                #remove sharp point at the beginning
                del sharp_points[0]
        
        return sharp_points
            
    #uses the algorithm defined in [1] to find the sharp points of the trace...
    # returns a list of tuples of the form (index, (x, y))  for all the sharp points
    def getSharpPoints(self):
        #the first is a sharp point
        sharpPoints = [ (0, self.points[0]) ]
        
        #now calculate the slope angles between each pair of consecutive points...
        alpha = []
        for i in range(len(self.points) - 1):
            alpha.append( self.slopeAngle(self.points[i], self.points[i + 1]) )
            
        #check
        if len(alpha) <= 1:
            #very special case where the trace is one single point
            #no more sharp points than itself...
            return sharpPoints 
            
        #now detect sharp points...
        theta = [ (alpha[0] - alpha[1]) ]            
        for k in range(1, len(self.points) - 1):
            #use two different tests to detect sharp point...
            #1) change in writing direction (as defined in [1])
            #2) difference in writing direction angle between current point and
            #   and last sharp point higher than a threshold            
            addPoint = False
            
            #for 1) calculate difference in writing direction between 
            #current point and previous point
            if (k < len(self.points) - 2):
                theta.append( alpha[k] - alpha[k + 1] )
                
                if theta[k] != 0.0 and k > 0:
                    delta = theta[k] * theta[k - 1]
                    
                    if (delta <= 0.0 and theta[k - 1] != 0.0):
                        #direction at which the pen is moving has changed                        
                        addPoint = True
                
            #for 2) calculate the difference of angle between 
            #current point and last sharp point
            phi = self.angularDifference(alpha[sharpPoints[-1][0]], alpha[k])
            
            #circular angle....
            if (phi > math.pi):
                phi = math.pi * 2 - phi
                
            if (phi >= math.pi / 8):
                addPoint = True
                
                
            if addPoint:
                sharpPoints.append( (k, self.points[k] ) )                                    

        #the last point is a sharp point
        sharpPoints.append( (len(self.points) - 1, self.points[-1]  ) )
        
        return sharpPoints

    #Returns the angle of the slope of the line that connects p1 and p2
    def slopeAngle(self, p1, p2):
        x = p2[0] - p1[0]
        y = p2[1] - p1[1]

        return math.atan2(y, x)

    #find the absolute minimum difference between two angles
    def angularDifference(self, alpha, beta):
        diff = abs(alpha - beta)
        
        #circular ....
        while (diff > math.pi):
            diff = math.pi * 2 - diff
        
        return diff

    #finds the signed minimum difference between two given angles
    def signedAngularDifference(self, alpha, beta):
        diff = alpha - beta
        
        if diff > math.pi:
            diff = alpha - (beta + math.pi * 2)

        if diff < -math.pi:
            diff = (alpha + math.pi * 2) - beta
                
        return diff

    #linear interpolation
    def lerp(self, p1, p2, t):
        x = p1[0] * (1- t) + p2[0] * t
        y = p1[1] * (1 -t) + p2[1] * t

        return (x, y)

    #Spline interpolation using Catmull-Rom method
    def catmullRom(self, p1, p2, p3, p4, t):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        xt =  0.5 * ((-x1 + 3*x2 -3*x3 + x4)*t*t*t \
               + (2*x1 -5*x2 + 4*x3 - x4)*t*t \
               + (-x1+x3)*t \
               + 2*x2)
        yt =  0.5 * ((-y1 + 3*y2 -3*y3 + y4)*t*t*t \
               + (2*y1 -5*y2 + 4*y3 - y4)*t*t \
               + (-y1+y3)*t \
               + 2*y2)
    
        return (xt, yt)  

    #Resample the trace using splines (pre-processing)
    def splineResample(self, sharp_points, subDivisions):                
        new_points = []
        
        self.sharp_points = sharp_points
        
        #check special case: Only one sharp_point
        if len(sharp_points) == 1:
            self.points = [ sharp_points[0][1] ]
            return
        
        for i in range(len(sharp_points)):
            #add the sharp point
            new_points.append( sharp_points[i][1] )
            
            if i < len(sharp_points) - 1:
                innerPoints = (sharp_points[i + 1][0] - sharp_points[i][0]) * subDivisions
                tStep = 1.0 / innerPoints
            
            #depending on the current range...
            if i == 0 or i == len(sharp_points) - 2:
                #between first and second sharp points...
                # or between the last two sharp points...
                # use linear interpolation...                                                
                for k in range(1, innerPoints):
                    new_points.append(self.lerp(sharp_points[i][1], sharp_points[i + 1][1], tStep * k))

            elif i < len(sharp_points) - 2:                
                    #in the middle of four sharp points... use Catmull-Rom                
                    for k in range(1, innerPoints):
                        new_points.append(self.catmullRom(sharp_points[i - 1][1], sharp_points[i][1], sharp_points[i + 1][1], sharp_points[i + 2][1], tStep * k))
              
        #now replace
        self.points = new_points
        
    #relocate points in the trace based on two boxes, 
    #one to define and clamp current values
    #and the second to relocate values inside of it
    #box in format [ minX, maxX, minY, maxY ] 
    def relocatePoints(self, inputBox, outputBox):
        inputWidth = inputBox[1] - inputBox[0]
        inputHeight = inputBox[3] - inputBox[2]
        outputWidth = outputBox[1] - outputBox[0]
        outputHeight = outputBox[3] - outputBox[2]
        
        if inputWidth == 0:
            inputBox[1] = inputBox[1] + 0.01
            inputBox[0] = inputBox[0] - 0.01
            inputWidth = 0.02
            
        if inputHeight == 0:
            inputBox[3] = inputBox[3] + 0.01
            inputBox[2] = inputBox[2] - 0.01
            inputHeight = 0.02  
        
        #for all points ... 
        for i in range(len(self.points)):
            #clamp (just to keep the function as general as possible)
            #minX
            if self.points[i][0] < inputBox[0]:
                self.points[i] = (inputBox[0], self.points[i][1])
            #maxX
            if self.points[i][0] > inputBox[1]:                
                self.points[i] = (inputBox[1], self.points[i][1])
                
            #minY
            if self.points[i][1] < inputBox[2]:
                self.points[i] = (self.points[i][0], inputBox[2])
            #maxX
            if self.points[i][1] > inputBox[3]:                
                self.points[i] = (self.points[i][0], inputBox[3])
                        
            #new Coordinates
            x = ((self.points[i][0] - inputBox[0]) / (inputWidth)) * outputWidth + outputBox[0]
            y = ((self.points[i][1] - inputBox[2]) / (inputHeight)) * outputHeight + outputBox[2]
            
            #replace...
            self.points[i] = (x, y)
            
        #update sharp points...
        for i in range(len(self.sharp_points)):
            #clamp (just to keep the function as general as possible)
            #minX
            if self.sharp_points[i][1][0] < inputBox[0]:
                self.sharp_points[i] = (self.sharp_points[i][0], (inputBox[0], self.sharp_points[i][1][1]))
            #maxX
            if self.sharp_points[i][1][0] > inputBox[1]:
                self.sharp_points[i] = (self.sharp_points[i][0], (inputBox[1], self.sharp_points[i][1][1]))
            #minY
            if self.sharp_points[i][1][1] < inputBox[2]:
                self.sharp_points[i][1] = (self.sharp_points[i][0], (self.sharp_points[i][1][0], inputBox[2]))
            #maxY
            if self.sharp_points[i][1][1] > inputBox[3]:
                self.sharp_points[i] = (self.sharp_points[i][0], (self.sharp_points[i][1][0], inputBox[3]))
                
            #new Coordinates
            x = ((self.sharp_points[i][1][0] - inputBox[0]) / (inputWidth)) * outputWidth + outputBox[0]
            y = ((self.sharp_points[i][1][1] - inputBox[2]) / (inputHeight)) * outputHeight + outputBox[2]
            
            #replace...
            self.sharp_points[i] = (self.sharp_points[i][0], (x, y))
            
        #bounding box has changed...
        self.bounding_box = None
                

    #gets the count, average, and range of the points at which segments 
    #are intersect by the given line 
    def getInfoCrossings(self, line):
        #returns (count, (avg_x, avg_y), (first_x, first_y), (last_x, last_y))  
        crossings = self.getLineCrossings(line)        

        l_x1, l_y1 = line[0]
        l_x2, l_y2 = line[1]
        
        total_count = float(len(crossings))
        if total_count > 0:
            avg_x = 0.0
            avg_y = 0.0
            #assume 0 is first and last...
            first_i = 0
            last_i = 0
            
            dist = math.sqrt(math.pow( crossings[0][0] - l_x1, 2) + math.pow(crossings[0][1] - l_y1, 2) )
            first_dist = dist
            last_dist = dist
            for i, c in enumerate(crossings):
                c_x, c_y = c
                avg_x += c_x
                avg_y += c_y
                
                #compute distance...
                dist = math.sqrt(math.pow( c_x - l_x1, 2) + math.pow(c_y - l_y1, 2) )
                
                #check if new first... (closest to first point of line)
                if dist < first_dist:
                    first_dist = dist
                    first_i = i
                #check if new last... (farthest to first point of line)    
                if dist > last_dist:
                    last_dist = dist
                    last_i = i
            
            avg_x /= total_count
            avg_y /= total_count
            avg_point = ( avg_x, avg_y )
            first_point = crossings[first_i]
            last_point = crossings[last_i]
        else: 
            avg_point = ((l_x1 + l_x2) / 2.0, (l_y1 + l_y2) / 2.0 )
            first_point = (l_x2, l_y2)
            last_point = (l_x1, l_y1)
        
        return (total_count, avg_point, first_point, last_point)

    #Calculate the crossings feature for current trace
    def getLineCrossings(self, line):
        l_x1, l_y1 = line[0]
        l_x2, l_y2 = line[1]
        
        #check minimum
        #....x....
        if ( l_x1 < l_x2 ):
            l_xmin = l_x1
            l_xmax = l_x2
        else:
            l_xmin = l_x2
            l_xmax = l_x1
        #....Y....
        if (l_y1 < l_y2):
            l_ymin = l_y1
            l_ymax = l_y2
        else:
            l_ymin = l_y2
            l_ymax = l_y1

        minX, maxX, minY, maxY = self.getBoundaries()
        if not (minX < l_xmax and maxX > l_xmin and minY < l_ymax and maxY > l_ymin):
            #not even on the same bounding box ...
            return []
        
        if l_x1 == l_x2 and l_y1 == l_y2:
            #it's a point, no crossings with a single point...
            return []
        
        crossings = []
        if l_x1 != l_x2:
            #use the slope and intersect to compare...
            l_m = (l_y2 - l_y1) / (l_x2 - l_x1)
            l_b = l_y1 - l_m * l_x1
            
            #for every segment....
            for i in range(len(self.points) - 1):
                s_x1, s_y1 = self.points[i]
                s_x2, s_y2 = self.points[i + 1]
                
                if s_x2 == s_x1:
                    #the segment is a vertical line... 
                    if l_xmin <= s_x1 and s_x1 <= l_xmax:
                        #the vertical segment is inside the range of the current line...
                        y_int = s_x1  * l_m + l_b
                        #check if y_int in range of vertical line...
                        if min(s_y1, s_y2) <= y_int and y_int <= max(s_y1, s_y2):
                            #intersection found...
                            crossings.append( (s_x1, y_int) )
                else:
                    #the segment is not a vertical line
                    s_m = (s_y2 - s_y1) / (s_x2 - s_x1)
                    s_b = s_y1 - s_m * s_x1
                    
                    #check if parallel
                    if s_m == l_m:
                        #parallel lines, can only intersect if l_b == s_b 
                        #(meaning they are the same line), and have intersecting ranges 
                        if l_b == s_b:
                            if l_xmin <= max(s_x1, s_x2) and min(s_x1, s_x2) <= l_xmax:
                                 #intersection found
                                 crossings.append( ( (s_x1 + s_x2) / 2.0, (s_y1 + s_y2) / 2.0 ) )
                        
                    else:                        
                        #not parallel, they must have an intersection point 
                        x_int = (s_b - l_b) / (l_m - s_m)
                        y_int = x_int * l_m + l_b
                                                
                        
                        #the intersection point must be in both lines...                        
                        if l_xmin <= x_int and x_int <= l_xmax and \
                           min(s_x1, s_x2) <= x_int and x_int <= max(s_x1, s_x2):                            
                            crossings.append( (x_int, y_int) )
                                           
        else:
            #the given line is a vertical line...
            #can't use the slope, use a different method
            #for every segment....
            for i in range(len(self.points) - 1):
                s_x1, s_y1 = self.points[i]
                s_x2, s_y2 = self.points[i + 1]
                
                if s_x2 == s_x1:
                    #the segment is a vertical line (too)...
                    #only if they are on the same x position, and their range intersects
                    if s_x1 == l_x1 and min(s_y1, s_y2) < l_ymax and l_ymin < max(s_y1, s_y2):
                        crossings.append( ( (s_x1 + s_x2) / 2.0, (s_y1 + s_y2) / 2.0 ) )
                else:
                    #calculate intersection point
                    if min(s_x1, s_x2) <= l_x1 and l_x1 <= max(s_x1, s_x2):
                        #the vertical line is inside the range of the current segment...
                        s_m = (s_y2 - s_y1) / (s_x2 - s_x1)
                        s_b = s_y1 - s_m * s_x1
                    
                        y_int = l_x1  * s_m + s_b
                        #check if y_int in range of vertical line...
                        if l_ymin <= y_int and y_int <= l_ymax:
                            #intersection found...
                            crossings.append( ( l_x1 , y_int ) )


        return crossings
        
    #Finds the point in the trace that is closest to the given point
    #then return the distance between that point and the given point
    def closestDistanceToPoint(self, x, y):
        #gets the minimal distance between the trace and a point...
        closest = 0
        closest_dist = math.sqrt(math.pow( self.points[0][0] - x, 2) + math.pow(self.points[0][1] - y, 2) )
        
        for i in range(1, len(self.points)):
            dist = math.sqrt(math.pow( self.points[0][0] - x, 2) + math.pow(self.points[0][1] - y, 2) )
            if dist < closest_dist:
                closest_dist = dist 
        
        return closest_dist, self.points[closest]

    #Calculate the minimum distance betwen two given traces (For segmentation)
    def traceDistance(self, other_trace):
        
        closest_local = 0
        closest_other = 0
        
        diff_x = self.sharp_points[0][1][0] - other_trace.sharp_points[0][1][0]
        diff_y = self.sharp_points[0][1][1] - other_trace.sharp_points[0][1][1]
        #use square of distance...
        closest_dist = diff_x * diff_x + diff_y * diff_y
        
        for i in range(len(self.sharp_points) ):
            for j in range(len(other_trace.sharp_points) ):
                diff_x = self.sharp_points[i][1][0] - other_trace.sharp_points[j][1][0]
                diff_y = self.sharp_points[i][1][1] - other_trace.sharp_points[j][1][1]
                #use square of distance...
                dist = diff_x * diff_x + diff_y * diff_y
                
                if dist < closest_dist:
                    #new closest found...
                    closest_local = i
                    closest_other = j
                    
                    closest_dist = dist
        
        return math.sqrt( closest_dist )
                
    #Get general features of the trace
    def lineCumulativeFeatures(self):
        #1) total angular change
        #2) normalized line length
        #3) total sharp points
        totalAngularChange = 0.0
        lineLength = 0.0                               
        
        #get the sum of the angles of the smoothed curve...
        previousAngle = None         
        for i in range(len(self.points) - 1):
            currentAngle = self.slopeAngle(self.points[i], self.points[i + 1])
            lineLength += self.distance(i, i + 1)
            
            if previousAngle != None:
                totalAngularChange += self.angularDifference(currentAngle, previousAngle)
                 
            previousAngle = currentAngle
               
        
        return [ totalAngularChange, lineLength, float(len(self.sharp_points)) ]
        #return [ [totalAngularChange], [lineLength], float(len(self.sharp_points)) ]
        
    #Return the types of the general features 
    def lineCumulativeFeaturesTypes(self):
        # C = Continuous
        return ['C', 'C', 'C' ]

    #Get histograms of vertical and horizontal projections of the trace
    def getHistograms(self, bins):
        #create empty bins
        horizontal = [ 0 for x in range(bins)]
        vertical = [ 0 for x in range(bins)]
            
        bin_size = 2.0 / (bins - 1)
        #the bins are located in a way that the bins at the extremes will be half outside
        start = -1.0 - bin_size * 0.5
        #place the points in the bins...             
        for x, y in self.points:
            h_bin = int(math.floor((x - start) / bin_size ))
            v_bin = int(math.floor((y - start) / bin_size ))
            
            horizontal[h_bin] += 1
            vertical[v_bin] += 1
            
        return (horizontal, vertical)

    #Get the 2D histogram of points of the trace
    def get2DHistogram(self, rows, cols):
        #create the bins...
        distribution = [ [ 0 for x in range(cols) ] for y in range(rows) ]
        
        bin_size_x = 2.0 / (cols - 1)
        bin_size_y = 2.0 / (rows - 1)
        
        for x, y in self.points:
            h_div = (x + 1.0) / bin_size_x
            h_bin0 = int(math.floor(h_div))
            if h_bin0 == cols - 1:
                h_bin0 = cols - 2
                h_w1 = 1.0
            else:
                h_w1 = h_div - h_bin0
            h_bin1 = h_bin0 + 1

            v_div = (y + 1.0) / bin_size_y
            v_bin0 = int(math.floor(v_div))
            if v_bin0 == rows - 1:
                v_bin0 = rows - 2
                v_w1 = 1.0
            else:
                v_w1 = v_div - v_bin0
            v_bin1 = v_bin0 + 1            
            
            distribution[v_bin0][h_bin0] += (1.0 - h_w1) * (1.0 - v_w1)
            distribution[v_bin0][h_bin1] += h_w1 * (1.0 - v_w1)
            distribution[v_bin1][h_bin0] += (1.0 - h_w1) * v_w1
            distribution[v_bin1][h_bin1] += h_w1 * v_w1                            
           
        return distribution

    #Get the histogram of slope orientations
    def getGabor(self, rows, cols):
        #           ....  0    45  90   135
        distribution = [ 0.0, 0.0, 0.0, 0.0 ] * (rows * cols)

        # The total length is required for weighting...
        lineLength = 0.0
        distances = []
        mid_points = []
        for i in range(len(self.points) - 1):            
            mid_x = (self.points[i][0] + self.points[i + 1][0]) / 2
            mid_y = (self.points[i][1] + self.points[i + 1][1]) / 2
            
            mid_points.append( ( mid_x, mid_y ) )
            distances.append( self.distance(i, i + 1) )
            
            lineLength += distances[i]

        # Now calculate the angles...
        pi4 = math.pi / 4
        #off = (math.pi * 9) / 8
        off = math.pi
        
        for i in range(len(self.points) - 1):
            #relative weight according to length in relation to total...
            wl = distances[i] / lineLength

            #use midpoint of the line to distribute values on the corresponding cells of the grid
            mid_x, mid_y = mid_points[i]
            
            if rows >= 2:
                #....for Y....
                val_y = ((mid_y + 1.0) / 2.0) * (rows - 1)
                c0_y = int(val_y)
                if c0_y == rows - 1:
                    c0_y -= 1                
                c1_y = c0_y + 1
                w1_y = val_y - c0_y
                w0_y = 1 - w1_y
                
            if cols >= 2:
                #....for X....
                val_x = ((mid_x + 1.0) / 2.0) * (cols - 1)
                c0_x = int(val_x)
                if c0_x == cols - 1:
                    c0_x -= 1                
                c1_x = c0_x + 1
                w1_x = val_x - c0_x
                w0_x = 1 - w1_x
                
            #angle of current segment of line...
            currentAngle = self.slopeAngle(self.points[i], self.points[i + 1])

            #angle is between -pi and pi, add offset and divide between pi / 4
            p = (currentAngle + off) / pi4
            #the base orientation
            fp = math.floor( p )
            #the weight of the second orientation
            wp1 = p - fp
            #select bin for current orientation
            p0 = int(fp % 4)
            #select bin for next orientation (circular)
            p1 = int((p0 + 1) % 4)

            #the values....
            g0 = wl * (1.0 - wp1)
            g1 = wl * (wp1)

            #add to corresponding bins...
            if rows >= 2 and cols >= 2:
                #16 or more values ... (4 cells affected, 8 bins)
                #(0,0)
                distribution[(c0_y *  cols + c0_x) * 4 + p0] += g0 * w0_x * w0_y
                distribution[(c0_y *  cols + c0_x) * 4 + p1] += g1 * w0_x * w0_y
                #(0,1)
                distribution[(c1_y *  cols + c0_x) * 4 + p0] += g0 * w0_x * w1_y
                distribution[(c1_y *  cols + c0_x) * 4 + p1] += g1 * w0_x * w1_y
                #(1,0)
                distribution[(c0_y *  cols + c1_x) * 4 + p0] += g0 * w1_x * w0_y
                distribution[(c0_y *  cols + c1_x) * 4 + p1] += g1 * w1_x * w0_y
                #(1,1)
                distribution[(c1_y *  cols + c1_x) * 4 + p0] += g0 * w1_x * w1_y
                distribution[(c1_y *  cols + c1_x) * 4 + p1] += g1 * w1_x * w1_y
            elif rows >= 2:
                #8 or more values ... (2 cells affected, 4 bins)
                #(0,0)
                distribution[c0_y * 4 + p0] += g0 * w0_y
                distribution[c0_y * 4 + p1] += g1 * w0_y
                #(0,1)
                distribution[c1_y * 4 + p0] += g0 * w1_y
                distribution[c1_y * 4 + p1] += g1 * w1_y
            elif cols >= 2:
                #8 or more values ... (2 cells affected, bins)
                #(0,0)
                distribution[c0_x * 4 + p0] += g0 * w0_x
                distribution[c0_x * 4 + p1] += g1 * w0_x
                #(1,0)
                distribution[c1_x * 4 + p0] += g0 * w1_x
                distribution[c1_x * 4 + p1] += g1 * w1_x
            else:
                #only 4 values.... (1 cell affected, 2 bins)
                distribution[p0] += g0
                distribution[p1] += g1
        
        return distribution, lineLength

    #Calculates feature based on estimation fo the types of line segments
    #of the trace
    def getTypeSubsegmentsInfo(self):
        #Get the angular difference between each pair of neighbor lines...
        slopes = []
        angles = []
        lengths = []
        
        for i in range(len(self.points) - 1):
            slopes.append( self.slopeAngle(self.points[i], self.points[i + 1]) )
            lengths.append( self.distance(i, i + 1) )
            
            if len(slopes) > 1:                
                angles.append( self.signedAngularDifference(slopes[-1], slopes[-2]) )
        
        #initially, all segments are individual subsegments (and curve)
        self.segments = [ (x, x + 1, 0, 2, lengths[x]) for x in range(len(self.points) - 1) ]        
                 
        #then, use this angle difference to group possible straight lines...
        pos = 0
        threshold = math.pi * 0.0125
        
        for i in range(len(angles)):
            init, end, ang, stype, l1 = self.segments[pos]

            new_ang = ang + angles[i]
            if -threshold < new_ang and new_ang < threshold:
                init2, end2, ang2, stype2, l2 = self.segments[pos + 1]
                
                #segment added to the previous...
                #... also, set as straight line
                self.segments[pos] = (init, end2, new_ang, 1, l1 + l2)
                
                #erase current segment..
                del self.segments[pos + 1]
            else:
                #start a new segment
                pos += 1

        #now, check for contiguous "curve segments" and merge them
        pos = 0
        while pos < len(self.segments) - 1:
            init, end, ang, stype, l1 = self.segments[pos]

            if stype == 2:
                #curve, check next one ...
                init2, end2, ang2, stype2, l2 = self.segments[pos + 1]
                if stype2 == 2:
                     #merge...
                     self.segments[pos] = (init, end2, ang + abs(angles[init2-1]), 2, l1 + l2)
                     #delete
                     del self.segments[pos + 1]
                else:
                    pos += 1
            else:
                pos += 1

        #check for broken straight lines...
        pos = 0
        while pos < len(self.segments) - 1:
            init1, end1, ang1, stype1, l1 = self.segments[pos]
            init2, end2, ang2, stype2, l2 = self.segments[pos + 1]

            if stype1 == 1 and stype2 == 1 and angles[init2 - 1] < math.pi * 0.025:
                self.segments[pos] = (init1, end2, ang1 + ang2, 1, l1 + l2)

                del self.segments[pos + 1]
            else:
                pos += 1
        
        #additional check for small "Straight lines" between curves
        #should be a single curve segment
        pos = 0
        while pos < len(self.segments) - 2:
            init1, end1, ang1, stype1, l1 = self.segments[pos]
            init2, end2, ang2, stype2, l2 = self.segments[pos + 1]
            init3, end3, ang3, stype3, l3 = self.segments[pos + 2]

            if (stype1 == 2 and stype2 == 1 and stype3 == 2) \
                and (((l2 < l1 or l2 < l3) and (end2 - init2) <= 4) or (abs(ang2) > 0.1)):
                
                #small straight segment between two curve segments, join as a single curve segment

                inner_ang = 0.0
                for i in range(init2, init3 + 1):
                    inner_ang += abs(angles[i - 1])
                    
                self.segments[pos] = (init1, end3, ang1 + inner_ang + ang3, 2, l1 + l2 + l3)

                del self.segments[pos + 2]
                del self.segments[pos + 1]
            else:
                pos += 1

        #now, do anylisis of segments
        length_curves = 0.0
        length_straight = 0.0

        for init, end, ang, t, l in self.segments:
            if t == 1:
                #straight line
                length_straight += l
            else:
                #curve line
                length_curves += l

        total_length = length_straight + length_curves

        # Now calculate the angles...
        pi4 = math.pi / 4
        off = math.pi

        distribution_str = [[ 0.0 for x in range(4) ] for y in range(4)]
        distribution_crv = [ 0.0 ] * 4
        distribution_arc = [ 0.0 ] * 4
        
        for init in range(len(self.points) - 1):
            p = (slopes[init] + off) / pi4
            l = lengths[init]
            
            #the base orientation
            fp = math.floor( p )

            #the weight of the second orientation
            wp1 = p - fp

            #select bin for current orientation
            p0 = int(fp % 4)
            #select bin for next orientation (circular)
            p1 = int((p0 + 1) % 4)

            #the values....
            g0 = l * (1.0 - wp1)
            g1 = l * (wp1)

            x, y = self.points[init]

            w_right = (x + 1.0) / 2.0
            w_bottom = (y + 1.0) / 2.0

            #top-bottom
            #0 top, 1 bottom                
            distribution_str[0][p0] += g0 * (1.0 - w_bottom)
            distribution_str[1][p0] += g0 * (w_bottom)

            distribution_str[0][p1] += g1 * (1.0 - w_bottom)
            distribution_str[1][p1] += g1 * (w_bottom)

            #2 left, 3 right
            distribution_str[2][p0] += g0 * (1.0 - w_right)
            distribution_str[3][p0] += g0 * w_right

            distribution_str[2][p1] += g1 * (1.0 - w_right)
            distribution_str[3][p1] += g1 * w_right

            
        
        return (length_straight, length_curves, distribution_str, distribution_crv, distribution_arc)

    #Get the types og the subsegments feature
    def getSubsegmentsFeaturesTypes(self):
        # % of straight line
        # lenght straight
        # lenght curve
        # str slope distribution
        
        return ['c'] * 8

    #Calculate the circumcenter of the triangle formed by the given points
    def circumcenter(self, a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by ))

        if d == 0.0:
            return (0.0, 0.0, -1.0)

        px1 = (ax ** 2 + ay ** 2) * (by - cy)
        px2 = (bx ** 2 + by ** 2) * (cy - ay)
        px3 = (cx ** 2 + cy ** 2) * (ay - by)
        ux = (px1 + px2 + px3) / d

        py1 = (ax ** 2 + ay ** 2) * (cx - bx)
        py2 = (bx ** 2 + by ** 2) * (ax - cx)
        py3 = (cx ** 2 + cy ** 2) * (bx - ax)
        uy = (py1 + py2 + py3) / d

        return (ux, uy, 1.0)
    
