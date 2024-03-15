import os
from PIL import Image
from .test1 import Prediction

prediction = Prediction()


class ReadFolder:
    @staticmethod
    def read_image_in_folder(path_folder):
        for arquivo in os.listdir(path_folder):
            file_path = os.path.join(path_folder, arquivo)
            prediction.predict(file_path)

