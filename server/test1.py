from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights


class Prediction:

    @staticmethod
    def predict(file_path):
        img = read_image(file_path)

        # Step 1: Initialize model with the best available weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()

        # Step 2: Initialize the inference transforms
        preprocess = weights.transforms()

        # Step 3: Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)

        # Step 4: Use the model and print the predicted category
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        # score = prediction[class_id].item()

        if class_id > 289:
            category_name = 'Nenhum'
        else:
            category_name = weights.meta["categories"][class_id]

        print(f'{file_path} {category_name}')
        return category_name
