import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if len(sys.argv) < 7:
  print("Usage: python interpolation.py <original.csv> <extracted.csv> leftX rightX bottomY topY")
  sys.exit(1)
# Load the data from both files
# data1 is reference
data1 = pd.read_csv(sys.argv[2], header=None, skiprows=1, names=['x', 'y'], encoding='latin1')
data2 = pd.read_csv(sys.argv[1], header=None, skiprows=1, names=['x', 'y'], encoding='latin1')
h_data1 = pd.read_csv(sys.argv[2], nrows=0, encoding='latin1').columns
h_data2 = pd.read_csv(sys.argv[1], nrows=0, encoding='latin1').columns

data1 = data1.sort_values(by='x').reset_index(drop=True)
data2 = data2.sort_values(by='x').reset_index(drop=True)

# Finding the overlapping x range
overlap_x_min = max(data1['x'].min(), data2['x'].min())
overlap_x_max = min(data1['x'].max(), data2['x'].max())

# Determine the non-overlapping ranges
left_miss = data2['x'].min() - data1['x'].min()
right_miss = data1['x'].max() - data2['x'].max()

# Creating a common x range for the overlapping area
common_x_overlap = np.linspace(overlap_x_min, overlap_x_max, num=1000)

# Interpolating y values for both datasets onto the common x range for the overlapping area
interpolated_y1_overlap = np.interp(common_x_overlap, data1['x'], data1['y'])
interpolated_y2_overlap = np.interp(common_x_overlap, data2['x'], data2['y'])

# Calculating the absolute differences for the visualization
differences_overlap = interpolated_y2_overlap - interpolated_y1_overlap

# Calculate the average difference in y values for the overlapping range
average_difference_overlap = np.mean(np.abs(differences_overlap))

yrange = data1["y"].max()-data1["y"].min()
xrange = data1["x"].max()-data1["x"].min()
xrange = float(sys.argv[4])-float(sys.argv[3])
yrange = float(sys.argv[6])-float(sys.argv[5])
f1 = os.path.basename(sys.argv[1])
f2 = os.path.basename(sys.argv[2])
with open(f"interpolated_{f1}_VS_{f2}.stats", 'w') as file:
  file.write(f'MAE {average_difference_overlap/yrange} LeftMissed {left_miss/xrange} RightMissed {right_miss/xrange}\n')

plt.figure(figsize=(10, 6))

plt.plot(data1['x'],data1['y'], label='original', linestyle='-')
plt.plot(data2['x'],data2['y'], label='llm', linestyle='-')

bottom_y = min([data1['y'].min(),data2['y'].min()])
plt.fill_between(common_x_overlap, differences_overlap + bottom_y, bottom_y, color='gray', alpha=0.5, label='relative MAE ='+str(average_difference_overlap/yrange))
plt.scatter([None],[None], label='Miss '+str(left_miss)+" "+str(right_miss))

plt.title('Absolute Differences of Overlapping Interpolated Data Sets')
plt.xlabel(h_data2[0])
plt.ylabel(h_data2[1])
plt.legend()
plt.grid(True)

leftX   = min(float(sys.argv[3]), plt.xlim()[0])
rightX  = max(float(sys.argv[4]), plt.xlim()[1])
bottomY = min(float(sys.argv[5]), plt.ylim()[0])
topY    = max(float(sys.argv[6]), plt.ylim()[1])

plt.xlim(leftX,rightX)
plt.ylim(bottomY,topY)

plt.savefig(f"interpolated_{f1}_VS_{f2}.png")
plt.close()

