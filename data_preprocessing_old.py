from imagenetv2_pytorch import ImageNetV2Dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
import urllib.request
from PIL import Image

# Download and load ImageNet class index mapping
IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with urllib.request.urlopen(IMAGENET_CLASS_INDEX_URL) as url:
    imagenet_class_index = json.loads(url.read().decode())

# Map label indices to class descriptions, replacing spaces with underscores
idx_to_class_desc = {}
for key, value in imagenet_class_index.items():
    label = int(key)
    class_desc = value[1]
    if " " in class_desc:
        class_desc = class_desc.replace(" ", "_")
    idx_to_class_desc[label] = class_desc

# Define the 9 superclasses and include 'Other'
superclass_names = [
    "Bird",
    "Boat",
    "Car",
    "Cat",
    "Dog",
    "Fruit",
    "Fungus",
    "Insect",
    "Monkey",
    "Other",
]
superclass_to_idx = {name: idx for idx, name in enumerate(superclass_names)}

# Map each label to one of the superclasses
label_to_superclass = {}

# Mapping labels to superclasses using idx_to_class_desc
for label, description in idx_to_class_desc.items():
    description_lower = description.lower()
    if "bird" in description_lower or any(
        bird in description_lower
        for bird in [
            "cock",
            "hen",
            "duck",
            "goose",
            "owl",
            "parrot",
            "swan",
            "flamingo",
            "penguin",
            "eagle",
            "vulture",
            "peacock",
            "crane",
            "heron",
            "kingfisher",
            "woodpecker",
            "hummingbird",
            "sparrow",
            "finch",
            "swallow",
            "gull",
            "kite",
            "robin",
            "magpie",
            "chickadee",
            "jay",
            "turkey",
            "pigeon",
            "ostrich",
            "quail",
            "ptarmigan",
        ]
    ):
        assigned_superclass = "Bird"
    elif "boat" in description_lower or any(
        boat in description_lower
        for boat in [
            "boat",
            "ship",
            "canoe",
            "gondola",
            "yawl",
            "liner",
            "schooner",
            "trimaran",
            "barge",
            "lifeboat",
            "submarine",
            "raft",
            "kayak",
            "pirate",
            "aircraft_carrier",
            "speedboat",
            "bobsled",
            "catamaran",
            "sailboat",
            "dinghy",
            "paddlewheel",
            "dock",
        ]
    ):
        assigned_superclass = "Boat"
    elif "car" in description_lower or any(
        car in description_lower
        for car in [
            "car",
            "vehicle",
            "taxi",
            "cab",
            "limousine",
            "jeep",
            "minivan",
            "convertible",
            "wagon",
            "bus",
            "truck",
            "ambulance",
            "pickup",
            "trailer",
            "van",
            "moped",
            "motor_scooter",
            "snowmobile",
            "trolleybus",
            "fire_engine",
            "school_bus",
            "garbage_truck",
            "police_van",
            "racing_car",
            "sports_car",
            "go-kart",
            "golfcart",
            "forklift",
            "bicycle",
            "motorcycle",
            "bicycle-built-for-two",
            "mountain_bike",
            "streetcar",
        ]
    ):
        assigned_superclass = "Car"
    elif "cat" in description_lower or any(
        cat in description_lower
        for cat in [
            "cat",
            "kitten",
            "lion",
            "tiger",
            "leopard",
            "jaguar",
            "cheetah",
            "cougar",
            "panther",
            "lynx",
            "bobcat",
            "ocelot",
            "caracal",
            "wildcat",
            "tiger_cat",
            "persian_cat",
            "siamese_cat",
            "egyptian_cat",
            "tabby",
        ]
    ):
        assigned_superclass = "Cat"
    elif "dog" in description_lower or any(
        dog in description_lower
        for dog in [
            "dog",
            "puppy",
            "wolf",
            "fox",
            "coyote",
            "hound",
            "dingo",
            "dhole",
            "jackal",
            "hyena",
            "poodle",
            "terrier",
            "retriever",
            "bulldog",
            "beagle",
            "spaniel",
            "sheepdog",
            "collie",
            "pinscher",
            "dalmatian",
            "husky",
            "greyhound",
            "chihuahua",
            "labrador",
            "boxer",
            "great_dane",
            "newfoundland",
            "samoyed",
            "pomeranian",
            "keeshond",
            "malamute",
            "shih-tzu",
            "papillon",
            "basenji",
            "pug",
            "leonberg",
            "eskimo_dog",
            "rottweiler",
            "doberman",
            "bloodhound",
            "schipperke",
            "griffon",
        ]
    ):
        assigned_superclass = "Dog"
    elif "fruit" in description_lower or any(
        fruit in description_lower
        for fruit in [
            "fruit",
            "apple",
            "banana",
            "orange",
            "lemon",
            "pineapple",
            "strawberry",
            "mango",
            "melon",
            "grape",
            "pear",
            "peach",
            "plum",
            "cherry",
            "fig",
            "pomegranate",
            "custard_apple",
            "jackfruit",
            "papaya",
            "guava",
            "kiwi",
            "apricot",
            "berry",
            "raspberry",
            "blackberry",
            "blueberry",
            "pineapple",
            "bell_pepper",
            "cucumber",
            "zucchini",
            "artichoke",
            "cauliflower",
            "broccoli",
            "mushroom",
            "potato",
            "squash",
            "corn",
            "pumpkin",
            "eggplant",
            "tomato",
            "olive",
            "avocado",
        ]
    ):
        assigned_superclass = "Fruit"
    elif "fungus" in description_lower or any(
        fungus in description_lower
        for fungus in [
            "fungus",
            "mushroom",
            "morel",
            "truffle",
            "bolete",
            "hen-of-the-woods",
            "earthstar",
            "gyromitra",
            "stinkhorn",
            "agaric",
            "polypore",
            "coral_fungus",
            "edible_mushroom",
            "toadstool",
            "bracket_fungus",
        ]
    ):
        assigned_superclass = "Fungus"
    elif "insect" in description_lower or any(
        insect in description_lower
        for insect in [
            "insect",
            "bug",
            "bee",
            "ant",
            "fly",
            "beetle",
            "butterfly",
            "grasshopper",
            "cockroach",
            "mosquito",
            "dragonfly",
            "mantis",
            "wasp",
            "cricket",
            "ladybug",
            "cicada",
            "locust",
            "termite",
            "firefly",
            "stick_insect",
            "lacewing",
            "damselfly",
            "weevil",
            "centipede",
            "spider",
            "scorpion",
            "tick",
            "tarantula",
            "silkworm",
            "flea",
            "mite",
            "aphid",
            "leafhopper",
            "praying_mantis",
            "leaf_beetle",
            "earwig",
            "cockchafer",
            "mantid",
            "stonefly",
            "dung_beetle",
            "black_widow",
        ]
    ):
        assigned_superclass = "Insect"
    elif "monkey" in description_lower or any(
        monkey in description_lower
        for monkey in [
            "monkey",
            "ape",
            "chimpanzee",
            "baboon",
            "gorilla",
            "orangutan",
            "gibbon",
            "lemur",
            "macaque",
            "mandrill",
            "capuchin",
            "howler",
            "titi",
            "squirrel_monkey",
            "colobus",
            "guenon",
            "proboscis_monkey",
            "langur",
            "spider_monkey",
            "siamang",
            "indri",
            "patas",
        ]
    ):
        assigned_superclass = "Monkey"
    else:
        assigned_superclass = "Other"
    label_to_superclass[label] = superclass_to_idx[assigned_superclass]


class CustomImageNetV2Dataset(ImageNetV2Dataset):
    def __init__(self, variant="matched-frequency", transform=None, location="."):
        super().__init__(variant=variant, transform=transform, location=location)
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.labels = []
        self.superclass_labels = []
        for fname in self.fnames:
            label = int(fname.parent.name)
            self.labels.append(label)
            superclass_idx = label_to_superclass.get(label, superclass_to_idx["Other"])
            self.superclass_labels.append(superclass_idx)

    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        superclass_idx = self.superclass_labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, superclass_idx


data_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

dataset = CustomImageNetV2Dataset("matched-frequency", transform=data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Print some example mappings
print("Example mappings from label to superclass:")
for idx in range(10):
    label = dataset.labels[idx]
    superclass_idx = dataset.superclass_labels[idx]
    class_desc = idx_to_class_desc[label]
    superclass_name = superclass_names[superclass_idx]
    print(f"Label {label}: {class_desc} --> Superclass: {superclass_name}")

# Example batch
for images, labels, superclasses in dataloader:
    print(f"\nBatch of images shape: {images.shape}")
    print(f"Batch of labels: {labels}")
    print(f"Batch of superclasses: {superclasses}")
    break

print(f"\nTotal number of superclasses: {len(superclass_names)}")