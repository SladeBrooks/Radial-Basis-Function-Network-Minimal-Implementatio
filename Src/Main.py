import os
import numpy as np
import matplotlib.pyplot as plt
import csv


"""
Prepare the data

"""
data_dir = "Data/"
train_file = "learningdata.csv"
test_file = "testdata.csv"

train_path = os.path.join(data_dir,train_file)
test_path = os.path.join(data_dir,test_file)

train_data = np.loadtxt(open(train_path, "rb"), delimiter=",", skiprows=0)
test_data = np.loadtxt(open(test_path, "rb"), delimiter=",", skiprows=0)

train_output = train_data[:,5]
train_data = np.delete(train_data,5,axis = 1)

"""
Create methods
"""
#Returns Distance between a data point vector and a center vector
#Kept as a seperate method so it can be swapped out if need be
def _distance(center, data_point):
    return np.linalg.norm(data_point-center)


#Creates and returns phi using an interpolation matrix
#initialises phi by size input amounts x center amounts
def _create_phi(inputs, centers):
    #initialise phi as an empty matrix
    phi = np.zeros((len(inputs), len(centers)))
    #iterate through inputs
    for i, input in enumerate(inputs):
        #iterate through centers
        for m, center in enumerate(centers):
            #calculates distance for each center per input
            phi[i, m] = _distance(center, input)
    return phi

"""
model data
"""
centers = train_data

#phi is created using all the training data as the centers in a full interpolation way
#due to the fact there is only 1000 training examples this is deemed acceptable
phi = _create_phi(train_data,centers)
#Weights are created from phi inverse times training outputs
weights = np.dot( np.linalg.inv(phi), train_output)



"""
produce predictions on "testdata.csv"
predictions are stored in "Output/predictions.csv"
each time predictions are calculated, the old predictions are overwritten
"""
phi = _create_phi(test_data,centers)
predictions = []
val = 0
for i,p in enumerate(phi):
    val = 0
    for n,w in enumerate(weights):
        val += (w * p[n])
    predictions.append(val)
    #Prints the predictions to the terminal for extra analysis ease
    print("prediction number {}:{}".format(i+1,val))

#Writes the predictions into a csv file in the folder "Output"
with open('Output/predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for o in predictions:
        writer.writerow([o])
