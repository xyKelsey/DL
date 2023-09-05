import os
import random
import matplotlib.pyplot as plt
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms, datasets
from torch import nn, optim
from dataset import CatDog_Dataset
from torch.utils.data import DataLoader
from resnet import ResNet18

random.seed(20)
torch.manual_seed(20)

batch_size = 20
epochs = 20
device = torch.device("mps")
learning_rate = 0.001
root_path = 'mydata3'
normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])

train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ToTensor(),
                                      normalize])
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     normalize])

train_dir = os.path.join(root_path, 'train')
test_dir = os.path.join(root_path, 'test')

train_imgs = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
test_imgs = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]

train_data = CatDog_Dataset(train_imgs, train_transform)
test_data = CatDog_Dataset(test_imgs, test_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def main():
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # model.load_state_dict(torch.load("ResNet18_model2.pth", map_location=device), strict=False)
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        test_correct = 0
        batch_num = 0
        train_loss = 0
        test_loss = 0
        for i,(input, label) in enumerate(train_loader):
            batch_num += 1
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.max(output, 1)[1].cpu().numpy()
            label = label.cpu().numpy()
            correct = (pred == label).sum()
            train_correct += correct
            train_loss += loss.item()
        train_loss_list.append(train_loss/batch_size)
            # if num % 40 == 0:
            #     print("Epoch:", epoch+1, "Batch:", num, 'loss:{:.5f}, acc:{:.2f}%'.format(loss.item(), 100*correct/batch_size))

        model.eval()
        with torch.no_grad():
            for ii, (input, label) in enumerate(test_loader):
                input, label = input.to(device), label.to(device)
                output = model(input)
                loss = criterion(output, label)

                pred = torch.max(output, 1)[1].cpu().numpy()
                label = label.cpu().numpy()
                correct = (pred == label).sum()
                test_correct += correct
                test_loss += loss.item()
            test_loss_list.append(test_loss/batch_size)
            train_acc = train_correct / len(train_data)
            test_acc = test_correct / len(test_data)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("Epoch:", epoch+1, "train_loss:{:.5f}, test_loss:{:.5f}, train_acc:{:.2f}%, test_acc:{:.2f}%".format(train_loss/(i+1), test_loss/(ii+1), train_acc*100, test_acc*100))
    matplot_acc(train_acc_list, test_acc_list)
    matplot_loss(train_loss_list, test_loss_list)
    torch.save(model.state_dict(), 'ResNet18_model.pth')

def matplot_loss(train_loss, test_loss):
    plt.plot(train_loss, label="train_loss")
    plt.plot(test_loss, label="val_loss")
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel("epoch")
    plt.title("loss")
    plt.show()

def matplot_acc(train_acc, test_acc):
    plt.plot(train_acc, label="train_acc")
    plt.plot(test_acc, label="val_acc")
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel("epoch")
    plt.title("acc")
    plt.show()

if __name__ == '__main__':
    main()

