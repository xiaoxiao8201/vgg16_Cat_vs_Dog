# 使用VGG16实现猫狗分类，全过程概述（网络，数据集，训练）

本文旨在记录自己学习的过程，vgg是一个非常经典的大型网络，我将用它来实现猫狗分类，使用的数据集是kaggle的猫狗数据集。要点有：

- vgg16网络定义

- 如何定义dataset，怎么加载数据集

- 训练

- cuda，以及多卡训练

## 网络定义
按照论文里给出的图，可以看到vgg的简单结构：代码在vgg16.py。

![c2fdfc039245d6888f2a8c35134ecc18d31b248d](D:\xiao_xiao\free DOWNLOAD MANNGER\c2fdfc039245d6888f2a8c35134ecc18d31b248d.jpeg)

vgg16即是C这一栏，有16层。我们需要注意的一点是在FC层要设定最后一层的输出等于要分类的类别数。没有需要特别注意的。


## 数据集的加载，dataset的定义

这个问题困扰了很久，我下载了kaggle的数据集,代码在dataset.py里

![](D:%5CZZZhuomian%5Cvgg16%5Cdataset.png)

我们需要定义一个类，它继承torch.utils.data.Dataset，我们要实现`__init__()`,`__getitem__`,`__len__`,这几个方法。一般我们在init内先找出需要图片的路径：

```python
def __init__(self, root, transform):
    self.image_files = np.array([x.path for x in os.scandir(root)
                                 if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
    self.transform = transform
```





利用init中图片路径，来打开每一张图片，transform是对图片所需要做的处理：

```python
    def __getitem__(self, item):
        x = Image.open(self.image_files[item])
        x = self.transform(x)
        thisLabel =0
        if "dog" in self.image_files[item]:
            thisLabel = 1
        return x, thisLabel
```

在kaggle的猫狗数据集内，它的标签被写在文件名里面，我们在加载这一个图片的时候，将它的标签一起取出。



`__len__`比较简单，直接返回数据集的长度即可：

```python
def __len__(self):
    return len(self.image_files)
```

如此我们就定义好了这个类，我们现在所需要的就是加载训练

## 训练

`device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`
使用这个语句判断cuda是否可用，如不能使用则使用cpu计算
### 实例化网络

```python
net = my_vgg16(numClass=2).to(device)
```

### 实例化数据集类，并且加载

- 预处理，vgg16在最后使用了FC层来分类，我们的输入的图片大小就需要一致，在这里直接使用torchvision的transform来对图片进行预处理

  ```
  data_transform = transforms.Compose([
      transforms.Resize(size=(244, 244)),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])
  ```

- 实例化数据集`CatAndDog = myDataSet("./train/", data_transform)`

- 使用DataLoader加载

  ```
  cadSet = torch.utils.data.DataLoader(CatAndDog, batch_size=batchsize, shuffle=True, num_workers=2)
  ```

  第一个参数需要数据集对象，第二个batch_size表示每次取出多少张图片训练，小了会让训练时间变长，大了一方面是内存显存不足，另一方面会降低训练效果（亲身经历512时，无法收敛），一般取64；shuffle表示打乱数据集，num_workers表示取出数据集时的线程。

### 实例化优化器，与损失函数

```
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
loss = torch.nn.CrossEntropyLoss()
```

### 确定训练次数，开始训练

```python
for epoch in range(40):
	# 训练40次
    ally = 0 # 在这里我用ally来累加我在本次训练过程中分类正确的数量
    # 在我每一个batchsize内
    for batch_idx, (img, label) in enumerate(cadSet):
        img = img.to(device) # 使用显卡
        label = label.to(device)
        optimizer.zero_grad() # 梯度置零
        testlable = net(img) # 向前推理
        lossData = loss(testlable, label) # 计算loss
        lossData.backward() # 计算梯度
        optimizer.step() # 更新梯度
        print("Epoch:%d [%d|%d] loss:%f" % (epoch, batch_idx, len(cadSet), lossData))
        # 查看当前这个batchsize内，正确多少
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
```

保存模型：`torch.save(net.state_dict(), 'ckp/model.pth')`

## 推理，并评价模型

与训练过程类似，不多赘述，代码在eval.py里。

## 使用多张显卡

在代码里会用到`.to(device)`,表示使用将数据等移动到device上，如果使用多卡训练时我们则需要先

`os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'`,后面0,1表示使用显卡的编号，然后使用`.cuda()`数据移动到cuda上即可。

## ps：小白一个，欢迎交流