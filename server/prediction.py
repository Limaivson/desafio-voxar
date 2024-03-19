import os.path
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

"""
Prediction Class:
    This class provides functionality to make predictions about images using a pre-trained ResNet50 model.
Methods:
    - predict(file_path): Static method that takes the path of an image file as input and returns the name of the
    file and the expected category for the image.
    EXAMPLE:
    image.jpg, "cat"
"""


class Prediction:
    """
        Static method for making predictions about an image.

        Parameters:
            - file_path (str): The path of the image file to predict.

        Returns:
            A tuple containing the file name and expected category for the image.
    """

    @staticmethod
    def predict(file_path):
        img = Image.open(file_path)
        # Checks if the image is in RGB format, if not, converts it
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Load the weights from the pre-trained ResNet50 neural network
        weights = ResNet50_Weights.DEFAULT
        # Load the ResNet50 model with the weights
        model = resnet50(weights=weights)
        # Puts the model in evaluation mode (not training)
        model.eval()

        # Loads the preprocessing transformations
        preprocess = weights.transforms()

        # Apply preprocessing transformations
        batch = preprocess(img).unsqueeze(0)

        # Predict image category using loaded model
        prediction = model(batch).squeeze(0).softmax(0)
        # Returns the index of the category with the highest probability
        class_id = prediction.argmax().item()

        # Checks whether the predicted category is an animal or not, if not, returns "None"
        if class_id > 389:
            category_name = 'Nenhum'
        # But if so, returns the name of the predicted category
        else:
            # Returns the name of the predicted category
            category_name = weights.meta["categories"][class_id]
        # Returns the file name and expected category
        return os.path.basename(file_path), category_name
