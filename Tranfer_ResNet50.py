'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt

import torchsummary

#from models import *
#from utils import progress_bar



os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epoch_num = 25#300
print(device)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ColorJitter(0.1,0.1,0.1,0.1),
    #전이학습
    transforms.Resize(256),
    transforms.RandomCrop(224, padding=0),
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    
    #과적합 방지 효과가 있다고 함
    transforms.RandomRotation(10), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),#전이학습
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    #batch_size=32,
    shuffle=True,
    num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False, 
    download=True,
    transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=32, 
    #batch_size=32, 
    shuffle=False, 
    num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')

# net = ResNet50()
# 전이학습
net = torchvision.models.resnet50(pretrained=True)
num_fltrs = net.fc.in_features
#net.fc = nn.Linear(num_fltrs , 10)

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
# 전이학습 Frozon 추가
class MyNewNet(nn.Module):
  def __init__(self):
    super(MyNewNet, self).__init__()
    self.resnet50 = torchvision.models.resnet50(pretrained=True)
    self.linear1 = nn.Linear(512*4, 512)
    self.bn1 = nn.BatchNorm1d(512)
    self.drop1 = nn.Dropout(0.3)
    self.linear2 = nn.Linear(512, 10)

    # Forward Pass 정의 부분
    
    def forward(self, x):
      x = self.resnet50(x)
      #x = self.linear1(x)
      x = F.gelu(self.drop1(self.bn1(self.linear1(x))))
      return self.linear2(x)

my_model = MyNewNet()
for param in my_model.parameters():
  param.requires_grad = True
for param in my_model.linear1.parameters():
  param.requires_grad = True
for param in my_model.linear2.parameters():
  param.requires_grad = True
  
from adabelief_pytorch import AdaBelief

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr)
optimizer=AdaBelief(net.parameters(), lr=0.0001, eps =1e-16, betas=(0.9, 0.999),
                    weight_decay = 5e-4, weight_decouple=False, rectify=False,
                    fixed_decay=False)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, last_epoch=-1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult=1, last_epoch=-1)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: 0.95 ** epoch)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
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
        if batch_idx % 50 == 0:
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'.format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'.format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return (100. * correct / total)

def test(epoch):
    global best_acc
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

        print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'.format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        #  print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return best_acc


train_error = []
test_error = []

lerningrate_temp = []

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+epoch_num): 
        train_error.append(train(epoch))    
        test_error.append(test(epoch))          
        scheduler.step()
        print(optimizer.param_groups[0]['lr'])
        lerningrate_temp.append(optimizer.param_groups[0]['lr'])
        
        #print(train_error)
        #print(test_error)
    
    plt.plot(train_error,label='train_acc')
    plt.plot(test_error,label='test_acc')
    plt.legend()
    plt.show()
    
    plt.plot(lerningrate_temp,label='lr')
    plt.legend()
    plt.show()