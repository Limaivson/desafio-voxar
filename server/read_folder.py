import os
from .prediction import Prediction
from .to_csv import ToCSV

prediction = Prediction()
to_csv = ToCSV()

""" 
ReadFolder Class:

This class is responsible for reading images in a folder and predicting their categories, writing the results to a CSV file.

Attributes:
    - list (list): An empty list to store prediction results.

Methods:
    - __init__(): Initialization method that initializes the list to store prediction results.
    - read_image_in_folder(path_folder): Method to read images in a folder and predict their categories.
"""


class ReadFolder:
    def __init__(self):
        self.list = []

    """
        Method for reading images in a folder and predicting their categories.

        Parameters:
            - path_folder (str): The path of the folder containing the images to be read.

        Returns:
            An error message if the folder is not found.
    """

    def read_image_in_folder(self, path_folder):
        try:
            # Iterates over the files in the specified folder.
            for arquivo in os.listdir(path_folder):
                # Gets the full path of the file.
                file_path = os.path.join(path_folder, arquivo)
                # Performs category prediction for the current file and stores the result in the list.
                line = prediction.predict(file_path)
                # Adds the file name and expected category to the list.
                self.list.append([line[0], line[1]])
            # Writes prediction results to the classification_images.csv file.
            to_csv.fill_csv(self.list)
        except FileNotFoundError:
            # Returns an error message if the folder is not found.
            return 'Error reading folder'
