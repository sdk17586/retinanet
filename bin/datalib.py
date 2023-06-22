import os
import cv2
import json
import torch
import random
import numpy as np

import skimage
import skimage.io
import skimage.transform
import skimage.color

from torchvision import transforms
from PIL import Image
from torch.utils.data.sampler import Sampler


from torch.utils.data import DataLoader, random_split
# Get file list each className
def getFileList(classInfo, className):

    for _class in classInfo:
        if className == _class["className"]:
            return className
        elif "sourceClassName" in _class and className == _class["sourceClassName"]:
            return _class["className"]
    
    return None


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, imagePathList, classInfo, classNameList, imageSize, batchSize, grayScale, transform):
        self.imagePathList = imagePathList
        self.classNameList = classNameList
        self.classInfo = classInfo
        self.grayScale = grayScale
        self.imageSize = imageSize
        self.batchSize = batchSize
        self.transform = transform

    def __len__(self):
        return len(self.imagePathList)

    def __getitem__(self, idx):
        annotations = np.zeros((0, 5))
        imagePath = self.imagePathList[idx]

        rootPath, file = os.path.split(imagePath)
        fileName, _ = os.path.splitext(file)

        image = cv2.imread(imagePath)
        height, width = image.shape[:2]

        if self.grayScale == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        datData = os.path.join(rootPath, fileName + ".dat")
        with open(datData, "r") as f:
            datData = json.load(f)

        polygonData = datData["polygonData"]
        for polygon in polygonData:
            x1 = float(polygon["position"][0]["x"])
            y1 = float(polygon["position"][0]["y"])
            x2 = float(polygon["position"][1]["x"])
            y2 = float(polygon["position"][1]["y"])

            # draw.rectangle((int(x1), int(y1), int(x2), int(y2)), outline="green", width=3)
            newX1 = max(min(x1, width), 0)
            newY1 = max(min(y1, height), 0)
            newX2 = max(min(x2, width), 0)
            newY2 = max(min(y2, height), 0)

            className = getFileList(self.classInfo, polygon["className"])
            if className is None:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = [newX1, newY1, newX2, newY2]
            annotation[0, 4] = float(self.classNameList.index(className))
            annotations = np.append(annotations, annotation, axis=0)

        # if int(self.batchSize) == 1:
        #     image = image.squeeze(0)

        #여기서 0이 되어버린다. 계산속도를 빠르게 하기위해 255로 나누어 노멀라이즈한다.
        image = image.astype(np.float32)/255.0
        sample = {'img': image, 'annot': annotations}

        sample = self.transform(sample)

        return sample

    def collate_fn(self, batch):

        imgs = [s['img'] for s in batch]
        annots = [s['annot'] for s in batch]
        scales = [s['scale'] for s in batch]

        widths = [int(s.shape[0]) for s in imgs]
        heights = [int(s.shape[1]) for s in imgs]
        batch_size = len(imgs)

        max_width = np.array(widths).max()
        max_height = np.array(heights).max()

        # 모든 원소는 0으로 초기화됩니다.
        padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
        
        for i in range(batch_size):
            img = imgs[i]
            padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:
            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

            if max_num_annots > 0:
                for idx, annot in enumerate(annots):
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1

        padded_imgs = padded_imgs.permute(0, 3, 1, 2)

        return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


# class AspectRatioBasedSampler(Sampler):

#     def __init__(self, data_source, batch_size, drop_last):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.groups = self.group_images()

#     def __iter__(self):
#         random.shuffle(self.groups)
#         for group in self.groups:
#             yield group

#     def __len__(self):
#         if self.drop_last:
#             return len(self.data_source) // self.batch_size
#         else:
#             return (len(self.data_source) + self.batch_size - 1) // self.batch_size

#     def group_images(self):
#         # determine the order of the images
#         order = list(range(len(self.data_source)))
#         order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

#         # divide into groups, one group = one batch
#         return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


if __name__ == "__main__":
    from torchvision import transforms
    from utils import Normalizer, Resizer

    imagePathList = ["/data/sungmin/retinanet/sample/img/cat_0.jpg"]

    originImage = cv2.imread(imagePathList[0])
    # originImage = cv2.resize(originImage, (608, 608))
    classInfo = [
        {
            "classId": "e99f5",
            "className": "cat",
            "color": "#ef8783",
            "desc": "",
            "dsId": "e3375",
            "dsName": "[CLF] LG\uc5d4\uc194_Train_dataset",
            "isClick": False,
            "showTf": True,
        },
        {
            "classId": "76e2e",
            "className": "dog",
            "color": "#0000ff",
            "desc": "",
            "dsId": "e337e",
            "dsName": "[CLF] LG\uc5d4\uc194_Train_dataset",
            "isClick": False,
            "showTf": True
        }
    ]
    
    classNameList = ["cat", "dog"]
    imageSize = 608
    batchSize = 1
    grayScale = 3
    b = CustomImageDataset(imagePathList, classInfo, classNameList, imageSize, batchSize, grayScale, transform=transforms.Compose([Normalizer(), Resizer()]))
    trainDataLoader = DataLoader(b, batch_size=batchSize, shuffle=True, drop_last=True, pin_memory=True, collate_fn=b.collate_fn)
    
    for data in b:
        image, annot, scale = data["img"], data["annot"], data["scale"]
        image = np.array(image)

        for _annot in annot:
            # print(_annot)
            cv2.rectangle(originImage, (int(_annot[0]), int(_annot[1])), (int(_annot[2]), int(_annot[3])), (255, 0, 0), 2)

    cv2.imwrite("originImage.jpg", originImage)











































# 해당 코드는 데이터셋으로부터 미니배치를 생성하는 collate_fn 함수를 정의하는 부분입니다. 이 함수는 주로 객체 검출(object detection)과 같은 작업에서 사용되는 데이터를 처리하고 배치 단위로 정리하는 역할을 합니다.

# 해당 collate_fn 함수는 다음과 같은 작업을 수행합니다:

# batch 리스트에서 이미지(img), 어노테이션(annot), 스케일(scale) 정보를 추출합니다.
# imgs 리스트에서 각 이미지의 너비(widths)와 높이(heights)를 추출합니다.
# batch_size 변수를 통해 배치의 크기를 확인합니다.
# np.array를 사용하여 widths와 heights 리스트의 최대값을 찾아 max_width와 max_height 변수에 저장합니다.
# torch.zeros 함수를 사용하여 크기가 (batch_size, max_width, max_height, 3)인 4차원 텐서인 padded_imgs를 생성합니다. 이때, 모든 원소는 0으로 초기화됩니다.
# 반복문을 통해 각 이미지에 대해 패딩 작업을 수행합니다. img의 크기에 맞게 padded_imgs에 이미지를 할당합니다.
# annots 리스트에서 어노테이션의 최대 개수(max_num_annots)를 찾습니다.
# max_num_annots가 0보다 큰 경우, annot_padded 텐서를 생성하여 모든 원소를 -1로 초기화합니다. 이 텐서의 크기는 (len(annots), max_num_annots, 5)입니다.
# 반복문을 통해 각 어노테이션에 대해 패딩 작업을 수행합니다. annot의 크기에 맞게 annot_padded에 어노테이션을 할당합니다.
# padded_imgs의 차원을 변경하여 (batch_size, 3, max_width, max_height)로 변경합니다.
# 딕셔너리 형태로 데이터를 반환합니다. 키는 'img', 'annot', 'scale'이고, 값은 각각 패딩된 이미지, 패딩된 어노테이션, 스케일입니다.
# 이렇게 생성된 미니배치는 주로 PyTorch의 데이터로더(DataLoader)에서 활용되어 모델의 학습 또는 추론에 사용될 수 있습니다.