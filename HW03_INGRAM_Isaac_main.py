import math

import pandas
from pathlib import Path
from matplotlib import pyplot as plt

def read_traffic_data():
    # Store path where all traffic station data exists.
    traffic_stations_dir = Path('./Traffic_Stations_for_HW_03')
    # Recursively find every CSV in the traffic stations directory, and create a list of data from all CSVs.
    all_data_frames = list()
    for csv_path in traffic_stations_dir.rglob('*.csv'):
        # Read the CSV.
        data_frame = pandas.read_csv(csv_path)
        # Round the speed to the nearest integer.
        data_frame['SPEED'] = data_frame['SPEED'].round(0).astype(int)
        # Append this CSVs data to the list of all CSV data.
        all_data_frames.append(data_frame)
    # Combine all CSVs into one pile of data.
    final_df = pandas.concat(all_data_frames, ignore_index=True, sort=False)
    return final_df




def main():

    # Read in traffic data
    final_df = read_traffic_data()

    # Get minimum and maximum speeds
    min_speed = final_df['SPEED'].min()
    max_speed = final_df['SPEED'].max()

    # Store lists of all false alarm and true positive rates (x,y points) for graphing
    false_alarm_rates = list()
    true_positive_rates = list()

    # Initialize tracker for ideal point, setting it to the furthest point possible (1,0), and
    # the distance of that point from the true ideal point (sqrt(2)).
    ideal_point = (1,0)
    ideal_point_distance = math.sqrt(2)
    ideal_point_speed = 0

    # Iterate through all possible threshold speeds
    for threshold_speed in range(min_speed, max_speed + 1):

        # Divide data into two sets, one above the threshold speed, one below (and equal to) threshold speed
        below_threshold_df = final_df[final_df['SPEED'] <= threshold_speed]
        above_threshold_df = final_df[final_df['SPEED'] > threshold_speed]

        # Calculate number of true positives
        num_true_positives = above_threshold_df.loc[
            (above_threshold_df['INTENT'] == 2), 'SPEED'
        ].count()
        # Calculate number of false positives
        num_false_positives = above_threshold_df.loc[
            (above_threshold_df['INTENT'] == 0) | (above_threshold_df['INTENT'] == 1), 'SPEED'
        ].count()
        # Calculate number of false negatives
        num_false_negatives = below_threshold_df.loc[
            (below_threshold_df['INTENT'] == 2), 'SPEED'
        ].count()
        # Calculate number of true negatives
        num_true_negatives = below_threshold_df.loc[
            (below_threshold_df['INTENT'] == 0) | (below_threshold_df['INTENT'] == 1), 'SPEED'
        ].count()

        # Calculate true positive rate and add it to the list
        tpr = num_true_positives / (num_true_positives + num_false_negatives)
        true_positive_rates.append(tpr)
        # Calculate false negative rate and add it to the list
        fnr = num_false_positives / (num_false_positives + num_true_negatives)
        false_alarm_rates.append(fnr)

        # Calculate Euclidean distance of this point to (0,1)
        distance_from_ideal = math.sqrt((fnr ** 2) + ((tpr - 1) ** 2))
        # Store this point as the closest to the ideal point if it is better than the previous candidate
        if distance_from_ideal < ideal_point_distance:
            ideal_point_distance = distance_from_ideal
            ideal_point = (fnr, tpr)
            ideal_point_speed = threshold_speed

    # Plot the graph of false alarm rate and true positive rate points
    plt.plot(false_alarm_rates, true_positive_rates, marker='o', linestyle='--')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Alarm Rate")
    plt.title("Receiver Operating Characteristic Curve for Threshold Speeds")
    # Add label and surrounding box for point closest to ideal point
    plt.text(ideal_point[0] + 0.02, ideal_point[1] - 0.45, f'Point Closet to Ideal ({ideal_point_speed} mph)', color='purple', rotation=-45, ha='left')
    plt.plot(ideal_point[0], ideal_point[1], marker='s', mfc='none', mec='purple', markersize=10)
    plt.show()

if __name__ == '__main__':
    main()
