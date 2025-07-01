from PIL import Image
import torch
from torchvision import transforms
from petclassifier_pytorch import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model_weights.pth', weights_only=True, map_location=torch.device(device)))

img = Image.open(input('Enter image name: '))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(img_tensor)
    print("Raw logits:", logits)
    print("Probabilities:", torch.softmax(logits, dim=1))
    print("Prediction:", torch.argmax(logits, dim=1).item())
    if torch.argmax(logits, dim=1).item() == 1:
        print('Dog')
    else:
        print('Cat')
