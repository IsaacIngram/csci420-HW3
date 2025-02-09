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
    # Initialize tracker for the first zero points for both false alarm rate and true positive rate
    first_zero_far_point_speed = None
    last_tpr_one_point_speed = None

    least_total_error_point = (1, 0, 1, 0)

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
        # Calculate total rate of mistakes for this threshold
        percent_mistakes = (num_false_negatives + num_false_positives) / final_df.size

        if percent_mistakes < least_total_error_point[2]:
            least_total_error_point = (fnr, tpr, percent_mistakes, threshold_speed)

        # Calculate Euclidean distance of this point to (0,1)
        distance_from_ideal = math.sqrt((fnr ** 2) + ((tpr - 1) ** 2))
        # Store this point as the closest to the ideal point if it is better than the previous candidate
        if distance_from_ideal < ideal_point_distance:
            ideal_point_distance = distance_from_ideal
            ideal_point = (fnr, tpr)
            ideal_point_speed = threshold_speed

        # Set as the first false alarm rate zero point if it hasn't already been set, and this point has a FAR of 0
        if fnr == 0 and first_zero_far_point_speed is None:
            first_zero_far_point_speed = (fnr, tpr, threshold_speed)

        # Set as the first true positive rate zero point if it hasn't already been set, and this point has a TPR of 0
        if tpr == 1:
            last_tpr_one_point_speed = (fnr, tpr, threshold_speed)

    # Plot the graph of false alarm rate and true positive rate points
    plt.plot(false_alarm_rates, true_positive_rates, marker='o', linestyle='--')
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Alarm Rate")
    plt.title("Receiver Operating Characteristic Curve for Threshold Speeds")
    # Add label and surrounding box for point closest to ideal point
    plt.text(ideal_point[0] + 0.03, ideal_point[1] - 0.49, f'Point Closet to Ideal ({ideal_point_speed} mph)', color='purple', rotation=-45, ha='left')
    plt.plot(ideal_point[0], ideal_point[1], marker='s', mfc='none', mec='purple', markersize=10)
    # Add label and surrounding box for first false alarm rate of zero point
    plt.text(first_zero_far_point_speed[0] + 0.04, first_zero_far_point_speed[1] - 0.02, f'First False Alarm Rate of Zero ({first_zero_far_point_speed[2]} mph)', color='blue')
    plt.plot(first_zero_far_point_speed[0], first_zero_far_point_speed[1], marker='s', mfc='none', mec='blue', markersize=10)
    # Add label and surrounding box for first true positive rate of one point
    plt.text(last_tpr_one_point_speed[0] - 0.02, last_tpr_one_point_speed[1] - 0.85, f'Last True Positive Rate of One ({last_tpr_one_point_speed[2]} mph)', rotation=-90, color='green')
    plt.plot(last_tpr_one_point_speed[0], last_tpr_one_point_speed[1], marker='s', mfc='none', mec='green', markersize=10)
    # Add label and surrounding box for point with the least number of mistakes
    plt.text(least_total_error_point[0] + 0.01, least_total_error_point[1] - 0.57, f'Best threshold with lowest number\nof mistakes ({least_total_error_point[3]} mph)', rotation = -45, color='red')
    plt.plot(least_total_error_point[0], least_total_error_point[1], marker='s', mfc='none', mec='red', markersize=10)

    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()
