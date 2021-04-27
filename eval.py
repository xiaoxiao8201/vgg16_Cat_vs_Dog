from dataset import myDataSet
from VGG16 import my_vgg16
from torchvision import transforms
import torch

batch_size = 1

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
preNet = my_vgg16(numClass=2)
preNet.load_state_dict(torch.load("model99_0.97192.pth", map_location=device))
CatAndDog = myDataSet("./test_set/all/", data_transform)
cadSet = torch.utils.data.DataLoader(CatAndDog, batch_size=batch_size, shuffle=True, num_workers=2)
allYes = 0
preNet.cuda()


for batch_idx, (img, label) in enumerate(cadSet):
    img = img.to(device)
    label = label.to(device)
    testlable = preNet(img)
    print(batch_idx, "/", len(cadSet), end=" ")
    checkA = testlable.data.cpu().numpy()
    checkB = label.data.cpu().numpy()
    for idxc in range(len(checkB)):
        checkC = 1
        if checkA[idxc, 0] > checkA[idxc, 1]:
            checkC = 0
        if checkC == checkB[idxc]:
            allYes += 1
            print("yes", end=" ")
        else:
            print("no", end=" ")
    print(" ")
print("all yes:", allYes)
print("acc:", allYes / CatAndDog.__len__())
