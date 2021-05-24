#####################
#  Use the k-means clustering algorithm to decide which of 3 clusters given data belongs in
#
#####################


import pandas
import numpy
from array import *
from itertools import permutations
import time
import math

def read_data_file(input_file_name):

    inputfile = open(input_file_name, "r")
    inputlines = inputfile.readlines()

    ##############
    # parse through data lines, splitting into lists for each line
    ##############

    # initialize data_table list

    data_table_input = []

    # initialize counter variables
    count = 0

    for line in inputlines:
        datatable = line.replace("\n", "")  # filter out newlines
        datatable2 = datatable.split(",")  # split data on commas
        datatable2 = list(map(float, datatable2))  # convert to float, make list
        data_table_input.append(datatable2)
        count += 1

    return data_table_input, count
# open data file for reading

data_file = "places.txt"

data_table, count = read_data_file(data_file)

##### set initial values to first items in data_table
x_min = data_table[0][0]
x_max = data_table[0][0]
y_min = data_table[0][1]
y_max = data_table[0][1]
x_avg = data_table[0][0]
y_avg = data_table[0][1]

count = 0
for pair in data_table:
    count += 1
    if (pair[0] < x_min):
        x_min = pair[0]
    if (pair[0] > x_max):
        x_max = pair[0]
    if (pair[1] < y_min):
        y_min = pair[1]
    if (pair[1] > y_max):
        y_max = pair[1]
    x_avg = (x_avg * (count - 1) + pair[0]) / count
    y_avg = (y_avg * (count - 1) + pair[1]) / count


old_k0 = [0, 0]
old_k1 = [0, 0]
old_k2 = [0, 0]

k0 = [x_min, y_min]
k1 = [x_max, y_max]
k2 = [x_max, y_avg]

k_list = []

# build initial list
for pair in data_table:
    dist_k0 = math.sqrt((pair[0] - k0[0]) ** 2 + (pair[1] - k0[1]) ** 2)
    dist_k1 = math.sqrt((pair[0] - k1[0]) ** 2 + (pair[1] - k1[1]) ** 2)
    dist_k2 = math.sqrt((pair[0] - k2[0]) ** 2 + (pair[1] - k2[1]) ** 2)

    if(dist_k0<dist_k1 and dist_k0<dist_k2):  # assign to k0
        new_k = 0
    elif(dist_k1<dist_k0 and dist_k1<dist_k2):  # assign to k1
        new_k = 1
    else:  # assign to k2
        new_k = 2
    new_set = [new_k, pair[0], pair[1]]
    k_list.append(new_set)




k_new = [[0,0],[0,0],[0,0]]
k_counter = [0,0,0]


k_delta = 1

while (k_delta > .001):   # see if we are within .001 threshold, otherwise keep iterating
    k_counter = [0, 0, 0]
    for coords in k_list:
        k_counter[coords[0]] += 1
        k_new[coords[0]][0] += coords[1]
        k_new[coords[0]][1] += coords[2]
    k_new[0][0] = k_new[0][0] / k_counter[0]
    k_new[0][1] = k_new[0][1] / k_counter[0]
    k_new[1][0] = k_new[1][0] / k_counter[1]
    k_new[1][1] = k_new[1][1] / k_counter[1]
    k_new[2][0] = k_new[2][0] / k_counter[2]
    k_new[2][1] = k_new[2][1] / k_counter[2]

    #### calculate Euclidean distances
    k0_old_new_dist = math.sqrt(((k0[0] - k_new[0][0]) ** 2)+((k0[1] - k_new[0][1]) ** 2))
    k1_old_new_dist = math.sqrt(((k1[0] - k_new[1][0]) ** 2)+((k1[1] - k_new[1][1]) ** 2))
    k2_old_new_dist = math.sqrt(((k2[0] - k_new[2][0]) ** 2)+((k2[1] - k_new[2][1]) ** 2))
    k_delta = k0_old_new_dist+k1_old_new_dist+k2_old_new_dist
    k0 = k_new[0]
    k1 = k_new[1]
    k2 = k_new[2]

    #### now check if need to move any to new clusters
    for coords in k_list:
        dist_k0 = math.sqrt((coords[1] - k0[0]) ** 2 + (coords[2] - k0[1]) ** 2)
        dist_k1 = math.sqrt((coords[1] - k1[0]) ** 2 + (coords[2] - k1[1]) ** 2)
        dist_k2 = math.sqrt((coords[1] - k2[0]) ** 2 + (coords[2] - k2[1]) ** 2)
        if (dist_k0 < dist_k1 and dist_k0 < dist_k2):  # assign to k0
            new_k = 0
        elif (dist_k1 < dist_k0 and dist_k1 < dist_k2):  # assign to k1
            new_k = 1
        else:  # assign to k2
            new_k = 2
        if (coords[0] != new_k):
            coords[0] = new_k

counter = 0

outputfile = open("clusters.txt", "w")


for coords in k_list:
    print("Point number", counter, "belongs to cluster #",coords[0], sep = ' ', file = outputfile)
    counter += 1
outputfile.close()
print ("output file clusters.txt written")







