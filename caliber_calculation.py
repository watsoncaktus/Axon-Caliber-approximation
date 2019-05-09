## I am using python

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import numpy as np
import numpy.random as nprand
import pandas as pd
import scipy.stats as stats
from scipy.stats import binom

###########################################################################################
#  Calculate the caliber of motor neuron from serial electron microscopy image
###########################################################################################

# The sample was P0 Interscutularis muscle Motor Axon in the bundle, before the axon branches.
# The resolution is 4nm/pixel and the z-section is 60nm thick

# Segmentation was done manually 
# The segmentation volume has been pre-processed to generate a list of x, y, and z coordinates of the shell,
# though this shell is not very good (there are random lines in the middle of the hollow spaces 
# (as will be visualized below).

#############
# 1. Use panda to read the csv file containing the x, y, and z coordinates of all the points making up the volume 

## The z-cordinates correspond to the Electron microscopy z-section of the sample.
df = pd.read_csv("mask1.csv", names=["x","y", "z"])

## Visualize this volume with mathplotlib 3D library
plt.figure(figsize = (8,6))
ax = plt.gca(projection="3d")
ax.scatter(df['x'], df['y'], df['z'],
           s = 0.5, marker = '.', color ="royalblue")
ax.set_xlabel("x (pixel/4nm)", fontsize= 14)
ax.set_ylabel("y (pixel/4nm)", fontsize= 14)
ax.set_zlabel("z (section/60nm)", fontsize= 14)
plt.show()

## Let's example a z-section to see what we are dealing with:
plt.rc("font", size=14)
plt.figure(figsize = (8, 6))
plt.scatter(slice0['x'], slice0['y'], s = 0.5)
plt.xlabel("X Coordinate (pixel/4nm)")
plt.ylabel("Y Coordinate (pixel/4nm)")
plt.tight_layout()
plt.show()

#############
# 2. Now I want to thoroughly simplify the whole volume to a single point deep hull with Convex Hull analysis from the scipy library

whole_hull = ConvexHull(df.values)
whole_volume = df.values

## Visualize the Convex Hull
plt.rcdefaults()
plt.figure(figsize = (8,6))
ax = plt.gca(projection="3d")

for simplex in whole_hull.simplices:
    ax.plot(whole_volume[simplex, 0], whole_volume[simplex, 1], whole_volume[simplex, 2], alpha = 0.1, color = "royalblue")
for vertice in whole_hull.vertices:
    ax.scatter(whole_volume[vertice, 0], whole_volume[vertice, 1], whole_volume[vertice, 2], color = "black", s = 3)

ax.set_xlabel("X Coordinate (pixel/4nm)")
ax.set_ylabel("Y Coordinate (pixel/4nm)")
ax.set_zlabel("Z-section (60nm)")
plt.show()

## Conclusion: Simplification with 3D Convex Hull doesn’t work well

#############
# 3. I now try to use 2D Convex Hull on individual z-section

z_unique = df['z'].unique()
slices = []
slices_hull = []
for i in z_unique:
    # Index all rows with z = 0 ==> ALl points in the same plane of z = i
    current_slice = df.loc[df['z'] == i]
    
    # Drop z column, convert to a numpy array to input into scipy hull
    current_slice_drop = current_slice.drop(['z'], axis=1) 
    current_slice_values = current_slice_drop.values
    
    # Append to a list of slices, indices = z position
    slices.append(current_slice_values)
    
    # Calculate convex Hull using scipy
    current_hull = ConvexHull(current_slice_values)
    
    # Append to a list of hulls for each slice, indices = z position
    slices_hull.append(current_hull)
    
## Plot all the hulls

plt.rcdefaults()
plt.figure(figsize = (8,6))
ax = plt.gca(projection="3d")
for z_value in z_unique:
    this_slice = slices[z_value]
    this_hull = slices_hull[z_value]
    
    for simplex in this_hull.simplices:
        point_1 = this_slice[simplex, 0]
        point_2 = this_slice[simplex, 1]
        ax.plot(point_1, point_2, [z_value,z_value], "k-", alpha = 0.1)

ax.set_xlabel("X Coordinate (pixel/4nm)")
ax.set_ylabel("Y Coordinate (pixel/4nm)")
ax.set_zlabel("Z-section (60nm)")
plt.show()

#############
# 4. Skeletonization by average of all the points in a slice (centroid):

slices_avg_points_x = []
slices_avg_points_y = []
for z_value in z_unique:
    current_slice = slices[z_value]
    current_hull = slices_hull[z_value]
    current_x_avg = (sum(current_slice[current_hull.vertices, 0])) / len(current_hull.vertices)
    current_y_avg = (sum(current_slice[current_hull.vertices, 1])) / len(current_hull.vertices)
    slices_avg_points_x.append(current_x_avg)
    slices_avg_points_y.append(current_y_avg)
slices_avg_points_x = np.array(slices_avg_points_x)
slices_avg_points_y = np.array(slices_avg_points_y)

## Let's visualize all the hulls centered on their respective centroids

translated_vertices_x = []
translated_vertices_y = []
for z_value in z_unique:
    current_slice = slices[z_value]
    current_hull = slices_hull[z_value]
    current_avg_x = slices_avg_points_x[z_value]
    current_avg_y = slices_avg_points_y[z_value]
    translated_x = current_slice[current_hull.vertices, 0] - current_avg_x
    translated_y = current_slice[current_hull.vertices, 1] - current_avg_y
    translated_vertices_x.append(translated_x)
    translated_vertices_y.append(translated_y)
plt.rc("font", size=16)
plt.figure(figsize = (6, 6))
for z_value in z_unique:
    plt.plot(translated_vertices_x[z_value] *4/1000, translated_vertices_y[z_value] *4/1000, 'k-', alpha = 0.1)
plt.scatter([0],[0], color = "coral")
plt.xlabel(r"x ($\mu$m)", fontsize = 14)
plt.ylabel(r"y ($\mu$m)", fontsize = 14)
plt.axis('equal')
plt.show()

## Visualize the centroid line in the volume to see if it provides a good trajectory of the axon:

plt.figure(figsize = (8,6))
ax = plt.gca(projection="3d")
ax.plot(slices_avg_points_x* 4/1000, slices_avg_points_y* 4/1000, z_unique* 60/1000, alpha = 1, color = "coral")

for z_value in z_unique:
    this_slice = slices[z_value]
    this_hull = slices_hull[z_value]
    
    for simplex in this_hull.simplices:
        point_1 = this_slice[simplex, 0]
        point_2 = this_slice[simplex, 1]
        ax.plot(point_1 * 4/1000, point_2 * 4/1000, [z_value * 60/1000,z_value * 60/1000], "k-", alpha = 0.05)
# X = [10,  20]
# Y = [20, 40]
# Z = [0, 20]

# max_range = np.max(np.array([np.max(X)-np.min(X), np.max(Y)- np.min(Y), np.max(Z)- np.min(Z)]))
# Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(np.max(X)+np.min(X))
# Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(np.max(Y)+np.min(Y))
# Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(np.max(Z)+np.min(Z))
# for xb, yb, zb in zip(Xb, Yb, Zb):
#     ax.plot([xb], [yb], [zb], 'w')
ax.set_xlabel(r"X Coordinate ($\mu$m) ", fontsize= 14)
ax.set_ylabel(r"Y Coordinate ($\mu$m) ", fontsize= 14)
ax.set_zlabel(r"Z Coordinate ($\mu$m) ", fontsize= 14)
plt.show()

## A close up fot he centroid within volume
plt.figure(figsize = (8,6))
ax = plt.gca(projection="3d")
ax.plot(slices_avg_points_x[130:140], slices_avg_points_y[130:140], z_unique[130:140], alpha = 0.5, color = "coral")

for z_value in z_unique[130:140]:
    this_slice = slices[z_value]
    this_hull = slices_hull[z_value]
    
    for simplex in this_hull.simplices:
        point_1 = this_slice[simplex, 0]
        point_2 = this_slice[simplex, 1]
        ax.plot(point_1, point_2, [z_value,z_value], "k-", alpha = 0.2)


ax.set_xlabel("x", fontsize= 14)
ax.set_ylabel("y", fontsize= 14)
ax.set_zlabel("z", fontsize= 14)
plt.show()

#############
# 5. Calculate the centroid segment length as an approximation for the incremental height of the unit cylinder 
# that I'm fitting on the axon segment

# Calculate the length of the centroid line (average lines)
real_avg_points_x = slices_avg_points_x * 4/1000
real_avg_points_y = slices_avg_points_y * 4/1000
real_z_unique = z_unique * 60/1000

total_length = 0
segment_lengths = []
## Cartesian distance
for z_value in z_unique[1:]:
    x_diff = real_avg_points_x[z_value] - real_avg_points_x[z_value - 1]
    y_diff = real_avg_points_y[z_value] - real_avg_points_y[z_value - 1]
    z_diff = 60/1000
    length = np.sqrt(x_diff ** 2 + y_diff **2 + z_diff **2)
    segment_lengths.append(length)
    total_length = total_length + length
segment_lengths = np.array(segment_lengths)
plt.rc("font", size=14)
plt.figure(figsize = (10, 3))
plt.plot(z_unique[1:], segment_lengths)
plt.axhline(60/1000, color = "red")
plt.show()

## I'm also calculating the segmental tortuosity too 
segment_angle = (np.arccos(60/1000 / segment_lengths)) / np.pi * 180
plt.figure(figsize = (10, 3))
plt.plot(z_unique[1:], segment_angle)
plt.axhline(0, color = "red")
plt.show()

#############
# 6. Calculate incremental volumes (by adding the area of 2 z-sections * 60nm thick and divide by 2. 
# Than I will use this to calculate orthogonal cross-section = volume/segment length as an approx of volume/height

segment_volumes = []

for z_value in z_unique[1:]:
    this_hull = slices_hull[z_value]
    previous_hull = slices_hull[z_value - 1]
    this_volume = (this_hull.volume + previous_hull.volume) * 4/1000 * 4/1000 * 60/1000 * 1/2
    segment_volumes.append(this_volume)
segment_volumes = np.array(segment_volumes)
segment_cross_sections = segment_volumes / segment_lengths

#############
# 7. Some statistics on the segment cross sections :

## 95% CI interval from bottstrapping
def bootstrap_for_mean(sample, bootstrap_times, confidence):
    sample_size = len(sample)
    ### Mean
    mean = (sum(sample))/sample_size
    ### Bootstrap
    
    all_bootstrap_results =[]
    all_bootstrap_mean = []
    all_bootstrap_deviations = []
    for i in range(bootstrap_times):
        bootstrap_result_onetime = []
        for j in range(sample_size):
            x = nprand.randint(0, sample_size)
            result = sample[x]
            # Build the bootstrap sample
            bootstrap_result_onetime.append(result)
        all_bootstrap_results.append(bootstrap_result_onetime)
    
        # Calculate the mean of this bootstrap sample
        bootstrap_mean = (sum(bootstrap_result_onetime))/sample_size
        all_bootstrap_mean.append(bootstrap_mean)

        # Calculate the deviation of this bootstrap from the empirical sample
        bootstrap_deviation = bootstrap_mean - mean
        all_bootstrap_deviations.append(bootstrap_deviation)

    # Sort the deviation from smallest to biggest value
    sort_deviations = all_bootstrap_deviations
    sort_deviations.sort()
    
    # Approximate the low and high percentile of mean (to make the confidence_interval) 
    # by choosing the high and low, respectively index of the deviation list 
    # and find the difference between the empirical mean and the deviation
    
    low, high = (100 - confidence)/2, confidence + (100-confidence)/2
    index_for_low = int(round((high/100) * bootstrap_times -1))
    index_for_high = int(round((low/100) * bootstrap_times -1))
    
    percentile_low = mean - sort_deviations[index_for_low]
    percentile_high = mean - sort_deviations[index_for_high]
    return percentile_low, percentile_high, all_bootstrap_mean

edge_1, edge_2, means_m = bootstrap_for_mean(segment_cross_sections, 20, 95)


## Visualize the cross section plots 2 ways
plt.style.use('dark_background')
plt.figure(figsize = (9, 4))
plt.fill_between(z_unique[1:], segment_cross_sections, color = "gray")
plt.axhline(segment_cross_sections.mean(), color = "crimson", label = "Mean")
plt.axhline(edge_1, color = "yellow", linestyle= '--', label = "95% CI of the Mean")
plt.axhline(edge_2, color = "yellow", linestyle= '--')
plt.legend(loc="best")
plt.xlabel(r"Z-section  ")
plt.ylabel(r"Cross-section ($\mu$$m^{2}$) ")
plt.tight_layout()
plt.show()

## Histogram shows Gaussian distribution

plt.figure(figsize = (9, 4))
plt.hist(segment_cross_sections, color = "gray", density=True)

### I fit a Gaussian plot on this histogram

from scipy.stats import norm
meanie,std=norm.fit(segment_cross_sections)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, meanie, std)
plt.plot(x, y, label = "Normal Fit")
plt.axvline(segment_cross_sections.mean(), color = "crimson", label = "Mean")
plt.axvline(edge_1, color = "yellow", linestyle= '--', label = "95% CI of the Mean")
plt.axvline(edge_2, color = "yellow", linestyle= '--')
plt.xlabel(r"Cross-section ($\mu$$m^{2}$) ")
plt.ylabel("Frequency")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


