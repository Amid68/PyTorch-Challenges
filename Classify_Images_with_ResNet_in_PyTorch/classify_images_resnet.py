import warnings
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchinfo import summary
import torch.nn.functional as F

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess images
image_batch = [
    Image.open('bear.jpg').convert('RGB'),
    Image.open('condor.jpg').convert('RGB')
]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = torch.stack([preprocess(image) for image in image_batch])
input_tensor = input_tensor.to(device)

# Load the pretrained ResNet-18 model and compute probabilities
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model = model.to(device)
model.eval()

output = model(input_tensor)
probabilities = F.softmax(output, dim=1)
probabilities_bear = probabilities[0]
probabilities_condor = probabilities[1]

# Load class names and display top 5 predictions for each image
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

print('\nBear\n')
top5_prob, top5_catid = torch.topk(probabilities_bear, 5)
for i in range(top5_prob.size(0)):
    print(f'For the category {categories[top5_catid[i]]} the probability is {top5_prob[i].item()}')

print('\nCondor\n')
top5_prob, top5_catid = torch.topk(probabilities_condor, 5)
for i in range(top5_prob.size(0)):
    print(f'For the category {categories[top5_catid[i]]} the probability is {top5_prob[i].item()}')

