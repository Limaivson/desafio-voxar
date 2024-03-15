from torchvision import transforms, models
import torch


def robust_image_classification(image_path, confidence_threshold=0.8):
    """
    Classifica uma imagem de forma robusta usando um modelo ResNet50 pré-treinado.

    Argumentos:
        image_path (str): Caminho para o arquivo de imagem.
        confidence_threshold (float, opcional): Pontuação de confiança mínima
            necessária para a previsão. Padrão: 0,8.

    Retorno:
        str: Categoria prevista com confiança se a confiança exceder o limite,
            caso contrário, a mensagem "Incerto".
    """

    try:
        # Leitura da imagem e conversão para tensor
        img = transforms.pil_to_tensor(transforms.Resize(224)(transforms.ToTensor()(
            transforms.Image.open(image_path).convert('RGB'))))

        # Carregamento do modelo e pré-processamento
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights).to(device)
        model.eval()
        preprocess = weights.transforms()
        batch = preprocess(img).unsqueeze(0).to(device)

        # Previsão e cálculo da confiança
        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()

        # Retorno da categoria com base na confiança
        if score > confidence_threshold:
            category_name = weights.meta["categories"][class_id]
            return category_name
        else:
            return 'Incerto'

    except (FileNotFoundError, ValueError) as e:
        print(f"Erro: {e}")
        return 'Erro'


# Exemplo de uso
image_path = "imgs/gato.jpg"
category = robust_image_classification(image_path)
print(f"Categoria prevista: {category}")
