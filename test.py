import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

# Carregar a ResNet50 pré-treinada
model = resnet50(pretrained=True)
model.eval()


# Pré-processamento da imagem
def preprocess_image(img_path):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# Lista de classes de animais
classes_animais = ['tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray', 'stingray',
                   'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting',
                   'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 'kite', 'bald_eagle', 'vulture',
                   'great_grey_owl', 'European_fire_salamander', 'common_newt', 'eft', 'spotted_salamander', 'axolotl',
                   'bullfrog', 'tree_frog', 'tailed_frog', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin',
                   'box_turtle', 'banded_gecko', 'common_iguana', 'American_chameleon', 'whiptail', 'agama',
                   'frilled_lizard',
                   'alligator_lizard', 'Gila_monster', 'green_lizard', 'African_chameleon', 'Komodo_dragon',
                   'African_crocodile',
                   'American_alligator', 'triceratops', 'worm_snake', 'ringneck_snake', 'eastern_hognose_snake',
                   'smooth_green_snake',
                   'kingsnake', 'garter_snake', 'water_snake', 'vine_snake', 'night_snake', 'boa_constrictor',
                   'rock_python',
                   'Indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 'diamondback', 'sidewinder', 'trilobite',
                   'harvestman', 'scorpion', 'yellow_and_black_garden_spider', 'barn_spider', 'garden_spider',
                   'black_widow',
                   'tarantula', 'wolf_spider', 'tick', 'centipede', 'black_grouse', 'ptarmigan', 'ruffed_grouse',
                   'prairie_chicken',
                   'peacock', 'quail', 'partridge', 'African_grey', 'macaw', 'sulphur-crested_cockatoo', 'lorikeet',
                   'coucal',
                   'bee_eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted_merganser',
                   'goose',
                   'black_swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish',
                   'sea_anemone',
                   'brain_coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea_slug', 'chiton',
                   'chambered_nautilus',
                   'Dungeness_crab', 'rock_crab', 'fiddler_crab', 'king_crab', 'American_lobster', 'spiny_lobster',
                   'crayfish',
                   'hermit_crab', 'isopod', 'white_stork', 'black_stork', 'spoonbill', 'flamingo', 'little_blue_heron',
                   'American_egret',
                   'bittern', 'crane', 'limpkin', 'European_gallinule', 'American_coot', 'bustard', 'ruddy_turnstone',
                   'red-backed_sandpiper',
                   'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king_penguin', 'albatross', 'grey_whale',
                   'killer_whale', 'dugong',
                   'sea_lion', 'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu',
                   'Blenheim_spaniel', 'papillon',
                   'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
                   'black-and-tan_coonhound',
                   'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound',
                   'whippet', 'Ibizan_hound',
                   'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
                   'Staffordshire_bullterrier', 'American_Staffordshire_terrier',
                   'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
                   'Norwich_terrier', 'Yorkshire_terrier',
                   'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn',
                   'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull',
                   'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier',
                   'silky_terrier', 'soft-coated_wheaten_terrier',
                   'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever',
                   'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever',
                   'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter',
                   'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel',
                   'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael',
                   'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog',
                   'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler',
                   'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog',
                   'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff',
                   'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog',
                   'malamute', 'Siberian_husky', 'dalmatian', 'affenpinscher', 'basenji', 'pug', 'Leonberg',
                   'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow',
                   'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle',
                   'standard_poodle', 'Mexican_hairless', 'timber_wolf', 'white_wolf',
                   'red_wolf', 'coyote', 'dingo', 'dhole', 'African_hunting_dog']

# Passagem da imagem pela ResNet50
img_path = 'imgs/gato.jpg'
img_tensor = preprocess_image(img_path)
with torch.no_grad():
    outputs = model(img_tensor)

# Decodificação da classe predita
_, predicted_idx = torch.max(outputs, 1)
predicted_label = predicted_idx.item()

# Verificar se o rótulo predito corresponde a um animal
animal_presente = any(predicted_label == class_idx for class_idx in classes_animais)
print(f"Rótulo predito: {predicted_label}")

if animal_presente:
    print("Um animal está presente na imagem.")
else:
    print("Nenhum animal foi detectado na imagem.")
