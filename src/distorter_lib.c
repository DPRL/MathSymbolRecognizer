/*
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
*/

//Compile using:
//	gcc -shared distorter.c -o distorter.so

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

double distorter_init(){
    //Use current time as a seed
    srand(time(0));
}

double distorter_rand(){
    return ((double)rand() / (double)RAND_MAX);
}

double distorter_mirror(double value){

    double i_value = floor(value);

    if ( ((int)i_value) % 2 == 0 ){
        //Not mirror
        value -= i_value;
    } else {
        //Mirror
        value = 1.0 - (value - i_value);
    }

    return value;
}

double distorter_bilinear_filtering(double* map, double x, double y, int n_rows, int n_cols){
    if ( x < 0.0){
        x = 0.0;
    }
    if ( y < 0.0){
        y = 0.0;
    }
    //...row (y) and its weights...
    double v_row = y * (n_rows - 1);
    int r0 = (int)floor(v_row);
    double r_w1;
    if ( r0 >= n_rows - 1 ){
        r_w1 = 1.0;
        r0 = n_rows - 2;
    } else {
        r_w1 = v_row - floor(v_row);
    }
    int r1 = r0 + 1;

    //...col (x) and its weights
    double v_col = x * (n_cols - 1);
    int c0 = (int)floor(v_col);
    double c_w1;
    if (c0 >= n_cols - 1){
        c_w1 = 1.0;
        c0 = n_cols - 2;
    }else{
            c_w1 = v_col - c0;
    }
    int c1 = c0 + 1;

    //Bilinear filtering...
    double final_val = map[r0 * n_cols + c0] * (1.0 - r_w1) * (1.0 - c_w1) +
                    map[r1 * n_cols + c0] * r_w1 * (1.0 - c_w1) +
                    map[r0 * n_cols + c1] * (1.0 - r_w1) * c_w1 +
                    map[r1 * n_cols + c1] * r_w1 * c_w1;

    return final_val;
}

void distorter_create_noise_map(double* map_buffer, int map_n, int noise_n, int max_d){
    int i, row, col;

    //Create random noise...
    int t_noise = noise_n * noise_n;
    double* c_map = (double *)calloc(t_noise, sizeof(double));

    //...Set a random value per element...
    double* p_map = c_map;
    for (i = 0; i < t_noise; i++ ){
        *p_map = distorter_rand() * 2.0 - 1.0;
        p_map++;
    }

    //..also random displacements...
    int t_disp = 2 * max_d;
    double* disp = (double *)calloc(2 * max_d, sizeof(double));
    double* p_disp = disp;
    for ( i = 0; i < t_disp; i++){
        *p_disp = distorter_rand();
        p_disp++;
    }

    //Combine the maps acording to perlin noise algorithm...
    for (row = 0; row < map_n; row++ ){
        for (col = 0; col < map_n; col++ ){
            map_buffer[row * map_n + col] = 0.0;
        }
    }

    //... for each size of the map...
    double w, p_r, p_c, c_pow;
    for ( i = 0; i < max_d; i++){
        double dr = disp[ i * 2 ];
        double dc = disp[ i * 2 + 1 ];

        if ( i < max_d - 1 ){
            w = 1.0 / pow(2, i + 1);
        } else {
            w = 1.0 / pow(2, i);
        }

        c_pow = pow(2, i);

        //...for each pixel....
        for ( row = 0; row < map_n; row++){
            p_r = ((double)row / (double)map_n) * c_pow;
            //...mirror mapping...
            p_r =  distorter_mirror(p_r + dr);

            for ( col = 0; col < map_n; col++){
                p_c = ((double)col /(double)map_n) * c_pow;
                //...mirror mapping...
                p_c = distorter_mirror(p_c + dc);

                map_buffer[row * map_n + col] += w * distorter_bilinear_filtering(c_map, p_c, p_r, noise_n, noise_n);
            }
        }
    }

    //...Release allocated memory....
    free( disp );
    free( c_map );

}

/*
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
        noise_x = self.getNoiseMap(self.noise_size, self.noise_map_size, self.noise_map_depth)
        #...smooth x....
        noise_x = ndimage.gaussian_filter(noise_x, sigma=(2, 2), order=0)
        #...y....
        noise_y = self.getNoiseMap(self.noise_size, self.noise_map_size, self.noise_map_depth)
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

            #normalize...
            norm = math.sqrt(dist_x * dist_x + dist_y * dist_y)
            if norm > 0.0:
                dist_x /= norm
                dist_y /= norm
            else:
                dist_x = 0.0
                dist_y = 0.0

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

*/
