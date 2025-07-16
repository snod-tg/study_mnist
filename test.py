import torch
from PIL import Image
from torchvision.transforms import transforms
from Two_Network import *

image_path = 'test_image/img_2.png'
img = Image.open(image_path)
# print(img)

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((28, 28)),
                                transforms.ToTensor()])

img = transform(img)
# print(img.shape)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model_Adam/mnist_49.pth', map_location=torch.device('cpu'), weights_only=False)
print(model)
# model = model.to(device)


img = torch.reshape(img, (1, 1, 28, 28))

model = model.eval()
with torch.no_grad():
    output = model(img)

print(output)
print(output.argmax(1).item())

