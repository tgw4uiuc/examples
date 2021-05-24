####
#   Thomas Wright   tgw4@illinois.edu
#   Decision Tree Classifier
#   Create a height = 2 (root node and 2 levels of decision nodes) decision tree.
#   Train with training data to find optimal decision attributes and decision values
#   Then run test data through the model to see if it classifies the data correctly
####

import time
import sys
import math

#######################
#   Splits off the training or test portion of the input data, as requested.
#   Split is based on the first element of each list.  -1 is test data, anything else is trainig data.
#
#   inputs: list_to_split: (data list to be split into atest or training sublist)
#           dataset_type: indicates whether to split into training or test sets
#   outputs: new_data_list: the test or training dataset
######################

def make_training_or_test_data(list_to_split, dataset_type):
    new_data_list = []
    for data_row in list_to_split:
        if data_row[0] >= 0 and dataset_type == "training":
            new_data_list.append(data_row)
        elif data_row[0] < 0 and dataset_type == "test":
            new_data_list.append(data_row)
    return(new_data_list)




#######################
#   Calculates the info (entropy) of the submitted list of class counts
#
#   inputs: class_count_list: list of class counts
#   outputs: info_sum: entropy value of the list
######################

def calc_info(class_count_list):

    info_sum = 0.0

    for i in class_count_list:
        if (i != 0):
            info_calc = -(i/sum(class_count_list))*math.log(i/sum(class_count_list),2)
            info_sum += info_calc
    return(info_sum)




#######################
#   Extract the attributes from the dataset and create a list of just those attributes
#
#   inputs: data_list: (dataset to extract attributes from)
#   outputs: attributes: list of attributes
######################

def create_attrib_list(data_list):
    attributes = []
    for i in range(1, len(data_list[0])):
        attributes.append(data_list[0][i][0])
    return attributes




#######################
#   Find the split points halfway between the existing attribute values
#
#   inputs: input_list: set of attributes values to find splits between
#   outputs: split_list: list of best split points
######################

def make_split_list(input_list):
    split_list = []
    input_list = list(set(input_list))    # drop out duplicate values, if any
    input_list.sort()   # sort list in ascending order
    for i in range(len(input_list)-1):

        split_list.append((input_list[i]+input_list[i+1])/2)   # calculate spot halfway between points

    return(split_list)



#######################
#   Use the split points calculated previously and then find the info(entropy) value for each split.
#   Choose the one with the lowest entropy value for the attribute.
#
#   inputs:     input_list: data set to split
#               classes: the classes for the given data set
#               class_list_uniques:  list of just the unique values of the classes
#   outputs:    lowest_info:  the lowest info value found
#               split:  the best attribute split point
#               split_left_counts:  count of classes for data to the left of the split point
#               split_right_counts: count of classes for data to the right of the split point
######################

def process_splits_for_info(input_list, classes, class_list_uniques):

    split_list = make_split_list(input_list)

    best_split = []
    lowest_info = 9999.99    # pick an initial point far above any actual info value we'd encounter


    for split in split_list:
        split_left_counts = []
        split_right_counts = []
        for m in range(len(class_list_uniques)):    # set initial counts to zero
            split_left_counts.append(0)
            split_right_counts.append(0)
        for i in range(len(input_list)):
            data_point = input_list[i]

            if data_point <= split:
                for k in range(len(class_list_uniques)):
                    if class_list_uniques[k] == classes[i]:
                        split_left_counts[k] += 1
            else:
                for k in range(len(class_list_uniques)):
                    if class_list_uniques[k] == classes[i]:
                        split_right_counts[k] += 1

        ##### calculate the info(entropy) score for data the falls to the left and right sides of the split point
        info_split_left = calc_info(split_left_counts)*(sum(split_left_counts)/(sum(split_left_counts)+sum(split_right_counts)))
        info_split_right = calc_info(split_right_counts)*(sum(split_right_counts)/(sum(split_left_counts)+sum(split_right_counts)))


        info_split = info_split_left + info_split_right    # total info score is sum of left and right sides

        if (info_split < lowest_info):
            lowest_info = info_split
            best_split_list = [lowest_info, split, split_left_counts, split_right_counts]

            best_split = best_split_list.copy()
    return(best_split)




#######################
#   split the current training data based on the best split attribute and split point previously found
#   inputs:
#           split_attribute:     details of the split point attribute and values to use for splitting the data
#           data_to_split:      the data list to split into two parts
#   outputs:
#           data_return:    list of 2 lists that contain the left and right portions of the split data
######################

def split_data_to_right_left(split_attribute, data_to_split):
    data_left = []
    data_right = []
    data_return = []


    for item in data_to_split:
        for i in range(1, len(item)):

            if item[i][0] == split_attribute[0]:
                if item[i][1] <= split_attribute[2]:
                    data_left.append(item)
                else:
                    data_right.append(item)
    data_return.append(data_left)
    data_return.append(data_right)

    return(data_return)



#######################
#   count the occurences of each class in the provided data list
#   inputs:
#           data_list:     list of data to count the frequency of classes in
#   outputs:
#           class_list_counted:    list of classes and frequency counts of each class in the data
######################

def class_counter(data_list):
    class_list = []
    class_list_counted = []


    for item in data_list:
        if item[0] not in class_list:
            class_list.append(item[0])

    for item in class_list:
        append_item = [item, 0]
        class_list_counted.append(append_item)

    for item in data_list:
        for i in range(len(class_list)):
            if item[0] == class_list[i]:
                class_list_counted[i][1] += 1

    return(class_list_counted)



#######################
#   in the provided list, find which element (element name at index 0, element count at index 1) has the highest count
#   inputs:
#           data_list:     list of data to look for the max count in
#   outputs:
#           max_item:    returns the item label and count of the one in the list with the greatest count
######################

def maximum_count(data_list):

    max_count = 0
    max_item = []


    for item in data_list:
        if item[1] > max_count:
            max_count = item[1]
            max_item = item
        elif item[1] == max_count:
            if item[0] < max_item[0]:
                max_item = item

    return(max_item)



#######################
#   take the training data, find the best attribute and attribute value for each attribute, then find best
#   of all attributes to use for the split
#   inputs:
#           data_to_split:     list of the training data
#   outputs:
#           split_to_use:    the best split point , and its value
######################

def split_data(data_to_split):
    class_list = []

    for data_row in data_to_split:
        class_list.append(data_row[0])

    class_list_uniques = list(set(class_list))

    class_instance_counts = []

    for item in class_list_uniques:
        class_instance_counts.append(class_list.count(item))

    attrib_list = create_attrib_list(data_table)

    all_best_splits = []

    for i in range(len(attrib_list)):
        curr_attribs = []
        best_split_for_attrib = []
        for k in range(len(data_to_split)):  # find split points     # was training_data instead of data_to_split
            curr_attribs.append(data_to_split[k][i + 1][1])     # was training_data instead of data_to_split
        best_split_for_attrib.append(attrib_list[i])

        split_result = process_splits_for_info(curr_attribs, class_list, class_list_uniques)

        for item in split_result:
            best_split_for_attrib.append(item)
        all_best_splits.append(best_split_for_attrib)

    split_to_use = []
    split_to_use_value = 9999.9         # pick a value far above any that would actually happen for initial value

    for item in all_best_splits:    # loop through all the best points of each attribute, finding the best overall across all attributes
        if item[1] == split_to_use_value:    # resolve ties by using smaller label, as per instructions
            if item[0] < split_to_use[0]:
                split_to_use = item
        elif item[1] < split_to_use_value:
            split_to_use_value = item[1]
            split_to_use = item


    return (split_to_use)


##################
#  main program
#
##################

if __name__ == "__main__":
    start_time = time.time()

    input_type = "file"  # set whether to use file input for testing, or stdin for autograder
    timing = False
    if timing:
        start_time = time.time()


    if (input_type == "file"):   # read from file for testing, stdin for autograder

        ####
        # read from file
        ####
        inputfile = open("testinput.txt", "r")
        inputlines = inputfile.readlines()

        count = 0
        data_table = []
        line_counter = 0

        for line in inputlines:
            datatable = line.replace("\n","")    # filter out newlines
            datatable2 = datatable.split(" ")    # split text data on space characters
            datatable3 = []
            for element in datatable2:
                datatable5 = []
                if element.find(':') > 0:       # further split on colon character
                    datatable4 = (element.split(":"))
                    datatable5.append(int(datatable4[0]))
                    datatable5.append(float(datatable4[1]))
                    datatable3.append(datatable5)
                else:
                    datatable3.append(int(element))
            data_table.append(datatable3)
            line_counter += 1


    else:
        ######
        # Read from stdin for autograder
        ######

        data_table = []
        data_table_input = []
        data_rows = 0
        for line in sys.stdin:
            dataline = line.replace("\n", "")
            datalinesplit = dataline.split(" ")
            data_table_input.append(datalinesplit)
            data_rows += 1

        for row in data_table_input:
            datatable3 = []
            for element in row:
                datatable5 = []
                if element.find(':') > 0:
                    datatable4 = (element.split(":"))
                    datatable5.append(int(datatable4[0]))
                    datatable5.append(float(datatable4[1]))
                    datatable3.append(datatable5)
                else:
                    datatable3.append(int(element))

            data_table.append(datatable3)



    training_data = make_training_or_test_data(data_table, "training")
    test_data = make_training_or_test_data(data_table, "test")

########
#   Set the first and second level of tree nodes to be decision type nodes (as opposed to leaf nodes)
#   by default.  Will be changed to leaf nodes later if needed
########

    root_node = "Decision"
    left_node = "Decision"
    right_node = "Decision"


#########
#   root node data processing
#########
    split_to_use = split_data(training_data)  # find the best split
    root_attribute = split_to_use[0]
    root_value = split_to_use[2]

    left_right_data = split_data_to_right_left(split_to_use, training_data)   # split data into left and right parts
    root_left_data = left_right_data[0]
    root_right_data = left_right_data[1]


#########
# left node data processing
#########
    class_count_left = class_counter(root_left_data)

    if len(class_count_left) == 1:  # if only one class in the data, then this node becomes a leaf instead of decision node, and use that class for any data that takes this branch
        left_node = "Leaf"
        left_node_leaf_class = class_count_left[0][0]

    else:
        left_split_to_use = split_data(root_left_data)   # find best split for left data
        left_attribute = left_split_to_use[0]
        left_value = left_split_to_use[2]

        left_left_right_data = split_data_to_right_left(left_split_to_use, root_left_data)
        left_left_data = left_left_right_data[0]
        left_right_data = left_left_right_data[1]

        ll_class = maximum_count(class_counter(left_left_data))     # determine which classes to assign to the left branch below this node
        lr_class = maximum_count(class_counter(left_right_data))     # determine which classes to assign to the right branch below this node

    #########
    # right node data processing
    #########
    class_count_right = class_counter(root_right_data)

    if len(class_count_right) == 1:
        right_node = "Leaf"
        right_node_leaf_class = class_count_right[0][0]

    else:
        right_split_to_use = split_data(root_right_data)

        right_attribute = right_split_to_use[0]
        right_value = right_split_to_use[2]
        right_left_right_data = split_data_to_right_left(right_split_to_use, root_right_data)
        right_left_data = right_left_right_data[0]
        right_right_data = right_left_right_data[1]

        rl_class = maximum_count(class_counter(right_left_data))    # determine which classes to assign to the left branch below this node
        rr_class = maximum_count(class_counter(right_right_data))    # determine which classes to assign to the right branch below this node

    ######################
    #  building/training of the decision tree is done, next we'll travere the tree and classify the test data
    ######################


    ###############
    #  traverse tree with test data
    ###############

    next_step = "X"

    testing_counter = 0   # keep track of which training elements we've done so far

    for item in test_data:
        for i in range(1,len(item)):
            if (item[i][0] == root_attribute):
                if (item[i][1] <= root_value):
                    next_step = "L"
                else:
                    next_step = "R"

        if next_step == "L":   # go down left branch
            if left_node == "Leaf":
                print("Testing point ", testing_counter, " belongs to class: ",left_node_leaf_class)
                testing_counter += 1
            else:
                for j in range(1, len(item)):
                    if (item[j][0] == left_attribute):
                        if (item[j][1] <= left_value):    # choose whether to take left or right sub-branch
                            print("Testing point ", testing_counter, " belongs to class: ", ll_class[0])
                            testing_counter += 1
                        else:
                            print("Testing point ", testing_counter, " belongs to class: ", lr_class[0])
                            testing_counter += 1
        else:  # go down right branch
            if right_node == "Leaf":
                print("Testing point ", testing_counter, " belongs to class: ",right_node_leaf_class)
                testing_counter += 1
            else:
                for j in range(1, len(item)):
                    if (item[j][0] == right_attribute):
                        if (item[j][1] <= right_value):    # choose whether to take left or right sub-branch
                            print("Testing point ", testing_counter, " belongs to class: ", rl_class[0])
                            testing_counter += 1
                        else:
                            print("Testing point ", testing_counter, " belongs to class: ", rr_class[0])
                            testing_counter += 1

exit(0)