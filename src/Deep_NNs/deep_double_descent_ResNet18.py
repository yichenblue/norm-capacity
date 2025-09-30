import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_channels=64):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(3, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def make_resnet18k(k=64, num_classes=10) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=k)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
## 20% label noise to train set
## Comment out below to turn off label noise
num_samples = len(trainset.targets)
rands = np.random.choice(num_samples, num_samples//5, replace=False)
for rand in rands:
  tmp = trainset.targets[rand]
  trainset.targets[rand] = np.random.choice( list(range(0,tmp)) + list(range(tmp+1,10)) )
## Comment out above to turn off label noise
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Training
def train(net, criterion, optimizer, epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/(batch_idx+1), 1-correct/total

def test(net, criterion, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return test_loss/(batch_idx+1), 1-correct/total


def calculate_path_norm(model):
    # Compatible with DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = copy.deepcopy(model)
    with torch.no_grad():  # Disable gradient computation to avoid affecting optimization
        for param in model.parameters():
            param.data = param.data ** 2
    # CIFAR-10 input dimensions: (batch_size, channels, height, width) = (1, 3, 32, 32)
    path_input = torch.ones(1, 3, 32, 32).to(device)
    path_output = model(path_input)
    path_norm = torch.sum(path_output)
    return path_norm.item()


def calculate_sparse_approx_path_norm(model: nn.Module, input_shape=(3, 32, 32), samples=10) -> float:
    """
    Memory-efficient sparse approximation of Definition 5 path norm.
    Uses random sparse inputs to estimate ||θ||_P = α^T ∏(I + 1/L |U||W|) e
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    device = next(model.parameters()).device

    def get_blocks(model):
        blocks = []
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                blocks.append(block)
        return blocks

    def apply_conv_pair(W_conv, U_conv, x):
        x = F.conv2d(x, W_conv.weight, stride=W_conv.stride, padding=W_conv.padding)
        x = F.relu(x)
        x = F.conv2d(x, U_conv.weight, stride=U_conv.stride, padding=U_conv.padding)
        return x

    # === Step 1: Sample sparse vectors ===
    C, H, W = input_shape
    e_sum = 0.0
    blocks = get_blocks(model)
    L = len(blocks)

    for _ in range(samples):
        # Create a sparse binary input: 1 at k random locations
        x = torch.zeros(1, C, H, W, device=device)
        num_nonzero = max(1, (H * W * C) // 100)  # ~1% sparsity
        idx = torch.randint(0, C * H * W, (num_nonzero,))
        x.view(-1)[idx] = 1.0

        for i, block in enumerate(blocks):
            # Adjust channel mismatch using shortcut or 1x1 convolution
            if x.shape[1] != block.conv1.in_channels:
                if hasattr(block, 'shortcut'):
                    x = block.shortcut(x)
                else:
                    conv1x1 = nn.Conv2d(x.shape[1], block.conv1.in_channels, kernel_size=1).to(x.device)
                    x = conv1x1(x)
            gain = apply_conv_pair(block.conv1, block.conv2, x)  # [1, C, H, W]
            gain = gain.abs()
            # If spatial dimensions don't match, adjust x's size
            if x.shape[2:] != gain.shape[2:]:
                x = F.interpolate(x, size=gain.shape[2:], mode='nearest')
            # If channel count doesn't match, adjust x's channel count
            if x.shape[1] != gain.shape[1]:
                conv1x1 = nn.Conv2d(x.shape[1], gain.shape[1], kernel_size=1).to(x.device)
                x = conv1x1(x)
            x = x + (1.0 / L) * gain  # I + (1/L)·|U||W|

        # Flatten and sum to approximate e^T M_prod e
        e_sum += x.view(-1).sum().item()

    e_avg = e_sum / samples  # ≈ e^T M_prod e

    # === Step 2: Output layer
    linear_weight = model.linear.weight.detach().abs()
    alpha = linear_weight[0].abs()  # shape: [D], force non-negative
    alpha_norm = alpha.sum().item()

    return alpha_norm * e_avg


# Model

def main():
    print('==> Building model..')
    # Define the list of model widths to iterate through
    width_list = [12, 20, 24, 28, 64]
    epochs = 4000

    for model_width in width_list:
        print(f'\n==== Training model, width={model_width} ====')
        # Build model
        net = make_resnet18k(k=model_width, num_classes=10)
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        # Set log file path, including model width information
        log_path = f'log_{model_width}.txt'
        with open(log_path, 'w') as f:
            f.write('Epoch,Train Loss,Train Error,Test Loss,Test Error,Path Norm\n')

        for epoch in range(start_epoch+1, start_epoch+epochs):
            train_loss, train_error = train(net, criterion, optimizer, epoch)
            test_loss, test_error = test(copy.deepcopy(net), criterion, epoch)
            # Calculate path norm
            path_norm = calculate_path_norm(copy.deepcopy(net))
            # Remove resnet_path_norm related content
            # exact_path_norm = calculate_sparse_approx_path_norm(copy.deepcopy(net))
            print(f'Epoch: {epoch:03} | Train Loss: {train_loss:.04} | '
                  f'Train Error: {train_error:.04} | Test Loss: {test_loss:.04} | '
                  f'Test Error: {test_error:.04} | Path Norm: {path_norm:.06}')
            with open(log_path, 'a') as f:
                f.write(f'{epoch},{train_loss:.09},{train_error:.09},{test_loss:.09},{test_error:.09},{path_norm:.09}\n')
        
        # Save the trained model
        model_save_path = f'model_width_{model_width}.pth'
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epochs,
            'model_width': model_width,
            'final_train_loss': train_loss,
            'final_train_error': train_error,
            'final_test_loss': test_loss,
            'final_test_error': test_error,
            'final_path_norm': path_norm
        }, model_save_path)
        print(f'Model saved to: {model_save_path}')

if __name__ == '__main__':
    main()