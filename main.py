from dataset import myDataSet
from VGG16 import my_vgg16

from torch.utils.data import Dataset
from torchvision import transforms

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchsize = 1
epoch = 40
# device = torch.device("cpu")

data_transform = transforms.Compose([
    transforms.Resize(size=(244, 244)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
net = my_vgg16(numClass=2).to(device)
CatAndDog = myDataSet("./train/", data_transform)
cadSet = torch.utils.data.DataLoader(CatAndDog, batch_size=batchsize, shuffle=True, num_workers=2)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
loss = torch.nn.CrossEntropyLoss()

"""print(len(cadSet))
for batch_idx, (img, label) in enumerate(cadSet):
    print(batch_idx,"his label is ", label)"""

for epoch in range(40):
    ally = 0
    for batch_idx, (img, label) in enumerate(cadSet):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        testlable = net(img)
        lossData = loss(testlable, label)
        lossData.backward()
        optimizer.step()
        print("Epoch:%d [%d|%d] loss:%f" % (epoch, batch_idx, len(cadSet), lossData))
        checkA = testlable.data.cpu().numpy()
        checkB = label.data.cpu().numpy()
        for idxc in range(len(checkB)):
            checkC = 1
            if checkA[idxc, 0] > checkA[idxc, 1]:
                checkC = 0
            if checkC == checkB[idxc]:
                ally += 1
                print("y", end=" ")
            else:
                print("n", end=" ")
        print(" ")
        '''
    
        '''
torch.save(net.state_dict(), 'ckp/model.pth')

'''
batch_size=5 batch_idx=4999

'''
