import csv
import os

"""
ToCSV Class:

This class is responsible for handling writing data to a CSV file.

Methods:
    - fill_csv(list_of_images): Static method to fill a CSV file with the results of image predictions.
"""


class ToCSV:
    """
    Static method for populating a CSV file with image prediction results.

    Parameters:
        - list_of_images (list): It is a list containing the results of image predictions, where each element is
        a list [file_name, expected_category].
    """
    @staticmethod
    def fill_csv(list_of_images):
        # Checks if the output CSV file already exists, if it does not, it creates a new file.
        if not os.path.exists('classification_images.csv'):
            with open('classification_images.csv', 'w', newline='\n') as file:
                csv.writer(file, delimiter=',')
        # Opens the output CSV file to add prediction results.
        with open('classification_images.csv', 'a', newline='') as file:
            csv_file = csv.writer(file)
            # Writes each result line (file name and predicted category) to the CSV file.
            for i, j in list_of_images:
                csv_file.writerow([i, j])
