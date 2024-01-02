from PIL import Image
import torchvision.transforms as transforms
from CNN import  CNN
import torch
import matplotlib.pyplot as plt
from PIL import ImageOps
def load_and_preprocess_image(image_path, image_size=28):
    # 加载图像
    image = Image.open(image_path).convert('L')  # 转换为灰度图像
    # image = ImageOps.invert(image)
    # 定义转换过程
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化 (根据训练时的设置调整)
    ])

    # 应用转换
    return transform(image).unsqueeze(0)  # 添加批次维度
def load_model(model_path):
    model = CNN()  # 使用您的模型架构
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == '__main__':
    # 加载模型

    model = load_model('cnn2.pkl')  # 替换为模型权重文件的路径

    # 加载和预处理图像
    image_tensor = load_and_preprocess_image('7.png')  # 替换为图像文件的路径
    plt.imshow(image_tensor.squeeze(), cmap='gray')  # 使用squeeze去掉批次维度
    plt.title("Processed Image")
    plt.show()
    # 进行预测
    prediction = predict_image(model, image_tensor)
    print(f'Predicted Class: {prediction}')
