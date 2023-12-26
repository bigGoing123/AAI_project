import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_data, write_image, load_image, set_seed

BATCH_SIZE = 64
EPOCH = 30
ONLY_TEST = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(6666)
tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 导入数据集
test_data_list, test_name_list = load_data("test", is_overwrite=True, is_visualize=True)

train_dataset = torchvision.datasets.MNIST(root='./dataset', train=True, transform=tf, download=False)
valid_dataset = torchvision.datasets.MNIST(root='./dataset', train=False, transform=tf, download=False)
test_dataset = load_image("./visual_data")

train_dataloader = DataLoader(batch_size=BATCH_SIZE, dataset=train_dataset, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(batch_size=BATCH_SIZE, dataset=valid_dataset, shuffle=False, num_workers=0)
test_dataloader = DataLoader(batch_size=BATCH_SIZE, dataset=test_dataset, shuffle=False, num_workers=0)

if ONLY_TEST:
    resnet50 = torch.load("./Resnet_best_model.bin")
else:
    resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 10)

# 将原resnet50网络中的最后一个全连接层改成10分类的输出

resnet50 = resnet50.to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optim = torch.optim.Adam(resnet50.fc.parameters(), lr=0.001)
final_labels, best_accuracy = [], 0

if not ONLY_TEST:
    for epoch in range(EPOCH):
        resnet50.train()
        for data in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            output = resnet50(images)
            loss = loss_fn(output, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

        resnet50.eval()
        with torch.no_grad():
            accuracy = 0
            for data in tqdm(valid_dataloader, total=len(valid_dataloader), desc="Validating"):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                output = resnet50(images)
                accuracy += ((output.argmax(1) == labels).sum())
        print("第{}轮中，测试集上的准确率为：{}".format(epoch + 1, accuracy / len(valid_dataset)))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            print("找到了最好的模型！准确率为：{}".format(best_accuracy))
            torch.save(resnet50, "Resnet_best_model.bin")

with torch.no_grad():
    resnet50.eval()
    for data in tqdm(test_dataloader, total=len(test_dataloader), desc="Testing"):
        data = data.to(device)
        output = resnet50(data)
        labels = torch.argmax(torch.softmax(output, dim=1), dim=1)
        final_labels.extend(labels)
write_image(image_list=test_data_list, label_list=final_labels, name_list=test_name_list)
