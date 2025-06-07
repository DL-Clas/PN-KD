import torch
import sys
import os
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights


# Cut conv1 (input)
def prune_conv1(conv_layer, prune_ratio):
    weight = conv_layer.weight.data
    num_pruned_channels = int(conv_layer.out_channels * prune_ratio)  # Calculate the number of crop channels based on the ratio
    # print(num_pruned_channels, conv_layer.out_channels)
    norms = weight.view(weight.size(0), -1).norm(2, dim=1)  # Calculate the number of L2 paradigms for each channel
    _, indices = torch.topk(norms, k=num_pruned_channels, largest=True)  # Getting the least important access
    pruned_weight = weight[indices]

    new_conv_layer = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=pruned_weight.size(0),
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )

    new_conv_layer.weight.data = pruned_weight.clone()

    if conv_layer.bias is not None:
        new_conv_layer.bias.data = conv_layer.bias.data[indices].clone()

    return new_conv_layer, indices


# Cut conv2 (output)
def prune_conv2(conv_layer, remaining_indices):
    weight = conv_layer.weight.data
    pruned_weight = weight[:, remaining_indices, :, :]  # Only the residual portion of the channel is retained

    new_conv_layer = nn.Conv2d(
        in_channels=pruned_weight.size(1),
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )

    new_conv_layer.weight.data = pruned_weight.clone()

    if conv_layer.bias is not None:
        new_conv_layer.bias.data = conv_layer.bias.data.clone()

    return new_conv_layer, remaining_indices


# Cut BatchNorm
def prune_bn_layer(bn_layer, remaining_indices):
    new_bn_layer = nn.BatchNorm2d(len(remaining_indices))
    new_bn_layer.weight.data = bn_layer.weight.data[remaining_indices].clone()
    new_bn_layer.bias.data = bn_layer.bias.data[remaining_indices].clone()
    new_bn_layer.running_mean = bn_layer.running_mean[remaining_indices].clone()
    new_bn_layer.running_var = bn_layer.running_var[remaining_indices].clone()
    return new_bn_layer


# Cut DenseLayer
def prune_bottleneck_block(layer, prune_ratio):
    # Make sure the layer is a DenseLayer object
    if hasattr(layer, 'conv1') and hasattr(layer, 'conv2'):
        # Update the norm1 layer, first update norm1 then crop conv1
        indices = torch.arange(layer.conv1.in_channels)  # Number of input channels using conv1
        layer.norm1 = prune_bn_layer(layer.norm1, indices)  # Update norm1

        # Cut conv1
        new_conv1, indices = prune_conv1(layer.conv1, prune_ratio)
        layer.conv1 = new_conv1

        # Update the norm2 layer, the update needs to use the number of output channels of conv1
        layer.norm2 = prune_bn_layer(layer.norm2, torch.arange(layer.conv1.out_channels))

        # Cut conv2
        new_conv2, _ = prune_conv2(layer.conv2, indices)
        layer.conv2 = new_conv2

    return layer


# 裁剪DenseBlock
def prune_denseblock(dense_block, prune_ratio):
    for name, layer in dense_block.named_children():
        # 如果是DenseLayer对象，则进行裁剪
        if name.startswith("denselayer"):
            # print(name)
            dense_block[name] = prune_bottleneck_block(layer, prune_ratio)
    return dense_block


# 剪枝整个DenseNet模型
def prune_densenet(model, prune_ratios):
    model_layers = list(model.features.children())

    # 跳过最初的conv0, norm0, relu0, pool0层
    conv0 = model_layers.pop(0)  # 这是最初的conv0层
    norm0 = model_layers.pop(0)  # 这是最初的norm0层
    relu0 = model_layers.pop(0)  # 这是最初的relu0层
    pool0 = model_layers.pop(0)  # 这是最初的pool0层
    norm5 = model_layers.pop()  # 跳过最后的norm5层

    # 裁剪每个DenseBlock
    for i, dense_block in enumerate(model_layers):
        prune_ratio = prune_ratios[i // 2]  # 根据需要裁剪的比例取值
        model_layers[i] = prune_denseblock(dense_block, prune_ratio)  # 直接在原有位置上替换

    # 跳过最后的分类层
    classifier = model.classifier

    # 重新将剪枝后的层组合在一起，更新features部分
    model.features = nn.Sequential(
        conv0, norm0, relu0, pool0, *model_layers, norm5
    )

    # 保留原始的分类器层（classifier）
    model.classifier = classifier

    return model


# 模型训练与评估
def train_eval(compressed_model, epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")
    print('裁剪后的网络结构')
    print(compressed_model)
    compressed_model.to(device)
    # weights_init(compressed_model)

    loss_function = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(compressed_model.parameters(), lr = 0.001)

    # 训练生成的候选模型
    for epoch in range(epochs):
        # 训练和验证
        compressed_model.train()
        running_loss = 0.0
        best_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            outputs = compressed_model(images)
            loss = loss_function(outputs, labels.to(device))
            loss = loss.requires_grad_(True)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            running_loss += loss
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 评估模型精度
        compressed_model.eval()
        correct = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = compressed_model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels.to(device)).sum()
        val_acc = correct / len(val_loader.dataset)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {val_acc:.4f}")
        if best_acc <= val_acc:
            best_acc = val_acc
    model_name = f"./Weight/CI_D169_{best_acc:.3f}_model.pth"
    torch.save(compressed_model, model_name)
    return best_acc


# 已有的目标模型stu_model（输入为超参数，输出为准确率）
def target_model(state):
    print(state)
    # 保存到 txt 文件
    output_file = "D169_state.txt"
    with open(output_file, "a") as f:
        f.write(f"{state}\n")  # 每个值占一行

    tea_model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
    # 修改最后的全连接层以适应新数据集
    num_classes = 3  # 将此处改为类别数
    tea_model.classifier = torch.nn.Linear(tea_model.classifier.in_features, num_classes)
    stu_model = prune_densenet(tea_model, state)
    # 数据预处理，包括数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    # # 加载训练集与验证集
    # train_data = torchvision.datasets.CIFAR100(
    #     root = './data', train = True, transform = transform)
    # val_data = torchvision.datasets.CIFAR100(
    #     root = './data', train = False, transform = transform)

    # # 加载STL-10数据集
    # train_data = datasets.STL10(root = './data', split = 'train',
    #                             transform = transform)
    # val_data = datasets.STL10(root = './data', split = 'test',
    #                           transform = transform)
    # # 数据加载器
    # batch_size = 32
    # train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    # val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = 2)

    # 加载 Brain数据集
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data", "Brain_3")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root = os.path.join(image_path, "train"),
                                         transform = transform)
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent = 3)
    with open("calss_indices.json", 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数计算

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = nw)
    val_dataset = datasets.ImageFolder(root = os.path.join(image_path, "val"),
                                       transform = transform)
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = nw
                                             )
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    reward = train_eval(stu_model, 15, train_loader, val_loader)
    # 保存到 txt 文件
    output_file = "D169_val.txt"
    with open(output_file, "a") as f:
        f.write(f"{reward}\n")  # 每个值占一行

    return reward


if __name__ == '__main__':

    # 训练环境测试
    print(target_model([0.3, 0.5, 0.5, 0.5]))