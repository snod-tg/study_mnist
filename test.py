import torch
from PIL import Image, ImageOps
from torchvision.transforms import transforms
from Two_Network import *

image_path = 'test_image/img_1_5.png'
img = Image.open(image_path)


# 判断背景是否接近黑色背景
def is_not_black_background(img, threshold=30):
    width, height = img.size
    # 获取图像边缘的像素
    edge_pixels = []
    for x in range(width):
        edge_pixels.append(img.getpixel((x, 0)))  # 上边
        edge_pixels.append(img.getpixel((x, height - 1)))  # 下边
    for y in range(height):
        edge_pixels.append(img.getpixel((0, y)))  # 左边
        if y != 0 and y != height - 1:  # 排除已添加的四个角
            edge_pixels.append(img.getpixel((width - 1, y)))  # 右边

    # 计算所有边缘像素的平均RGB值
    avg_r = sum(pixel[0] for pixel in edge_pixels) / len(edge_pixels)
    avg_g = sum(pixel[1] for pixel in edge_pixels) / len(edge_pixels)
    avg_b = sum(pixel[2] for pixel in edge_pixels) / len(edge_pixels)

    # 计算平均灰度值，使用加权公式：0.299R + 0.587G + 0.114B
    avg_gray = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b

    # 如果平均灰度值小于阈值，则认为是黑色背景
    return avg_gray >= threshold


# 对图像进行转换，确保测试图片与训练图片一样黑底白字
def img_invert(origin_img):
    if origin_img.mode != 'RGB':
        origin_img = origin_img.convert('RGB')

    origin_img.show()
    if is_not_black_background(origin_img):
        print("need invert")
        origin_img = ImageOps.invert(origin_img)

    return origin_img


# 图像反色
img = img_invert(img)

# print(img)
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((28, 28)),
                                transforms.ToTensor()])
# 图像变换
img = transform(img)
# print(img.shape)

#展示变换后的图像
img_PIL = transforms.ToPILImage()(img)
img_PIL.show()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model_Adagrad/mnist_49.pth', map_location=torch.device('cpu'), weights_only=False)
print(model)
# model = model.to(device)


img = torch.reshape(img, (1, 1, 28, 28))
print(type(img))

model = model.eval()
with torch.no_grad():
    output = model(img)

print(output)
print(output.argmax(1).item())
