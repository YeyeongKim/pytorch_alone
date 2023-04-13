# 패키지 임포트 
import torch 
import torch.nn as nn 
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import Normalize

import argparse #파서 사용하기 위해

# parser 정의하기 : 하이퍼파라미터
def parse_args():
    parser = argparse.ArgumentParser() #그릇 만들기

    parser.add_argument("--batch_size", type=int, default=100, help='size of batch') #배치 사이즈라는 인자가 파서에 들어갈 거라고 말해줌
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device")
    parser.add_argument("--img_sizes", type=int, default=32)
    parser.add_argument("--model_type", default='lenet', choices=['mlp', 'lenet', 'linear', 'multi_conv', 'incep'])
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    #print(args.__dict__) #딕셔너리 형태로 바꾸기
    #print(vars(args)) #딕셔너리로 바꾸는 다른 방법

    # 이미지 통계치 확인 (mean, std)
    tmp_dataset = CIFAR10(root='../data', train=True, transform=None, download=True)
    mean = list(tmp_dataset.data.mean(axis=(0, 1, 2))/255)
    std = list(tmp_dataset.data.std(axis=(0, 1, 2))/255)

    # dataset 만들기 
    # 이미지 크기 변경, 텐서 만들기 
    # parser 에서 args에서 정의된 하이퍼 파라미터들 앞에 args. 붙여주기
    trans = Compose([Resize((args.img_size, args.img_size)), 
                    ToTensor(),
                    Normalize(mean=mean, std=std)])

    train_dataset = CIFAR10(root='../data', train=True, transform=trans, download=True)
    test_dataset = CIFAR10(root='../data', train=False, transform=trans, download=True)

    # dataloader 만들기 
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

# 모델 class 만들기 
class MyMLP(nn.Module):
    def __init__(self, hidden_size, num_class):
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size) 
        self.fc4 = nn.Linear(hidden_size, num_class) 

    def forward(self, x): 
        b, c, w, h = x.shape # [100, 1, 28, 28]
        # x = x.reshape(-1, w*h) # [100, 28*28]
        x = x.reshape(b, -1) # [100, 28*28]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x 

class myLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # batch norm 
        self.bn1 = nn.BatchNorm2d(num_features=6)
        # activation function 
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # batch norm 
        self.bn2 = nn.BatchNorm2d(num_features=16)
        # activation function 
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): 
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.reshape(b, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x 

class myLeNet_seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), 
            nn.BatchNorm2d(num_features=6), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2), 

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), 
            nn.BatchNorm2d(num_features=16), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2), 
        )
        
        self.seq2 = nn.Sequential(
            nn.Linear(16*5*5, 120), 
            nn.Linear(120, 84), 
            nn.Linear(84, 10), 
        )

    def forward(self, x): 
        b, c, h, w = x.shape
        x = self.seq1(x)
        x = x.reshape(b, -1)
        x = self.seq2(x)
        return x 

class myLeNet_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Linear 
        self.conv_fc1 = nn.Linear(1176, 2048)
        self.conv_fc2 = nn.Linear(2048, 1176)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): 
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        b, tmp_c, tmp_h, tmp_w = x.shape
        x = x.reshape(b, -1)
        x = self.conv_fc1(x) 
        x = self.conv_fc2(x) 
        x = x.reshape(b, tmp_c, tmp_h, tmp_w)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.reshape(b, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x 

class myLeNet_multi_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ = nn.ModuleList([nn.Conv2d(3, 6, 5, 1, 2)] + 
                                    [nn.Conv2d(6, 6, 5, 1, 2) for _ in range(2)])
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): 
        b, c, h, w = x.shape
        for module in self.conv1_: 
            x = module(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.reshape(b, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x 

class myLeNet_incep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.merge = nn.Conv2d(in_channels=18, out_channels=6, kernel_size=5)
        
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): 
        b, c, h, w = x.shape

        x1 = self.conv1_1(x)
        x2 = self.conv1_3(x)
        x3 = self.conv1_5(x)
        x_concat = torch.cat((x1, x2, x3), 1)
        x = self.merge(x_concat)

        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.reshape(b, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x 


# 모델 객체 만들기, loss 만들기, optim 만들기

# 위에서 제한해준 모델 이름 - 실제 우리의 모델과 연결지어주기
if args.model_type == 'mlp': 
    model = MyMLP(args.hidden_size, args.num_class).to(args.device) 
elif args.model_type == 'lenet' : 
    model = myLeNet().to(args.device)
elif args.model_type == 'linear': 
    model = myLeNet_linear().to(args.device)
elif args.model_type == 'multi_conv': 
    model = myLeNet_multi_conv().to(args.device)
elif args.model_type == 'incep': 
    model = myLeNet_incep().to(args.device)
else : 
    raise ValueError('뭔가 잘못됨')
    
model = myLeNet_incep().to(device)
loss_fc = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=lr)

def evaluate(model, dataloader):
    with torch.no_grad(): 
        model.eval() 
        correct = 0 
        total = 0 
        for (image, target) in dataloader: 
            image = image.to(device)
            target = target.to(device) 

            output = model(image)
            value, index = torch.max(output, dim=1)
            correct += (index == target).sum().item()
            total += index.shape[0]

        acc = correct / total
    model.train() 
    return acc 

def evaluate_class(model, dataloader):
    with torch.no_grad(): 
        model.eval() 
        correct = torch.zeros(num_class) 
        total = torch.zeros(num_class) 
        for (image, target) in dataloader: 
            image = image.to(device)
            target = target.to(device) 

            output = model(image)
            value, index = torch.max(output, dim=1)
            for i in range(num_class): 
                total[i] += (target == i).sum().item() 
                correct[i] += ((target == i) * (index == i)).sum().item()

        
    model.train() 
    return total, correct

# 학습 loop 돌리기 
for epoch in range(epochs) : 
    for idx, (image, target) in enumerate(train_loader): 
        image = image.to(device)
        target = target.to(device)

        out = model(image)
        loss = loss_fc(out, target)
        
        optim.zero_grad() 
        loss.backward()
        optim.step()

        if idx % 100 == 0: 
            print('Loss : ', loss.item())
            acc = evaluate(model, test_loader)
            print('accuracy : ', acc)
            total, correct = evaluate_class(model, test_loader)
            pass 


if __name__ == '__main__':
    main()