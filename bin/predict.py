import os
import sys
import cv2
import json
import time
import torch
import traceback
import numpy as np
import skimage

from loguru import logger
from torchvision import transforms


basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.extend([basePath, os.path.join(basePath, "model")])

from model import createModel
from logger import WedaLogger
from checker import gpuChecker, initGpuChecker

gpuNo = initGpuChecker()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuNo


class Predictor():
    def __init__(self, pathInfo):

        self.pathInfo = pathInfo
        self.modelPath = self.pathInfo["modelPath"] if "modelPath" in self.pathInfo else '/app'
        self.weightPath = self.pathInfo["weightPath"] if "weightPath" in self.pathInfo else "/app/weight"
        self.log = WedaLogger(logPath=os.path.join(self.modelPath, "log/predict.log"), logLevel="info")

        # set cpu/gpu
        self.setGpu()

        if os.path.isfile(os.path.join(self.weightPath, "weight.pth")):
            with open(os.path.join(self.weightPath, "classes.json"), "r") as jsonFile:
                self.classesJson = json.load(jsonFile)

            self.classNameList = [classInfo["className"] for classInfo in self.classesJson["classInfo"]]
            self.imageSize = self.classesJson["imageInfo"]["imageSize"] if "imageSize" in self.classesJson["imageInfo"] else 300
            self.grayScale = int(self.classesJson["imageInfo"]["imageChannel"])

            # if self.grayScale == 1:
            #     self.transform = transforms.Compose([
            #         transforms.Resize((self.imageSize, self.imageSize)),
            #         transforms.Grayscale(num_output_channels=self.grayScale),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=0.5, std=0.5)
            #     ])

            # else:
            #     self.transform = transforms.Compose([
            #         transforms.Resize((self.imageSize, self.imageSize)),
            #         transforms.ToTensor(),
            #     ])

            # model load
            logger.info("Model Loading ...")

            modelLoadStartTime = time.time()

            self.model = createModel(
                pretrained=False,
                channel=self.grayScale,
                numClasses=len(self.classNameList),
                device=self.device
            )
            
            self.model.load_state_dict(torch.load(os.path.join(self.weightPath, "weight.pth"), map_location=self.device))
            self.model.training = False
            self.model.eval()
            self.model.to(self.device)

            modelLoadTime = time.time() - modelLoadStartTime
            logger.debug(f"Model Load Success, Duration : {round(modelLoadTime, 4)} sec")

        else:
            raise Exception("This Model is not Trained Model, Not Found Model's Weight File")

    def setGpu(self):
        self.device, self.deviceType = gpuChecker(log=self.log, gpuIdx="auto")

    def runPredict(self, predImage):

        try:
            logger.info("Starting Model Predict...")
            logger.info("-"*100)
            logger.info("  Device:             {}  ".format(self.device.type))
            logger.info("  Image Scaling:      {}  ".format((self.imageSize, self.imageSize, self.grayScale)))
            logger.info("  Labels:             {}  ".format(self.classNameList))

            totalStartTime = time.time()

            # 이미지 예측을 위한 전처리
            logger.info("Input Data Preprocessing for Model...")
            preProStartTime = time.time()

            result = []
            heatMapImage = None
            originImage = predImage.copy()
            height, width, channel = originImage.shape

            smallest_side = min(height, width)

            min_side = 608
            max_side = 1024
            scale = min_side / smallest_side

            largest_side = max(height, width)

            if largest_side * scale > max_side:
                scale = max_side / largest_side

            predImage = cv2.resize(predImage, (int(round(width * scale)), int(round((height * scale)))))

            rows, cols, cns = predImage.shape

            pad_w = 32 - rows % 32
            pad_h = 32 - cols % 32

            new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
            new_image[:rows, :cols, :] = predImage.astype(np.float32)

            predImage = new_image.astype(np.float32)

            predImage = new_image.astype(np.float32)
            predImage /= 255
            predImage -= [0.485, 0.456, 0.406]
            predImage /= [0.229, 0.224, 0.225]
            predImage = np.expand_dims(predImage, 0)
            predImage = np.transpose(predImage, (0, 3, 1, 2))
            predImage = torch.from_numpy(predImage).to(self.device).float()

            preProTime = time.time() - preProStartTime
            logger.debug(f"Input Data Preprocessing Success, Duration : {round(preProTime, 4)} sec")

            # 이미지 예측시작
            logger.info("Predict Start...")
            
            predStartTime = time.time()
            with torch.no_grad():
                scores, classification, transformed_anchors = self.model(predImage)

            predTime = time.time() - predStartTime
            logger.debug(f"Predict Success, Duration : {round(predTime, 4)} sec")

            # 예측 결과 형태 변환
            transferOutputStartTime = time.time()
            logger.info("Output Format Transfer...")

            idxs = np.where(scores.cpu() > 0.5)

            for i in range(idxs[0].shape[0]):
                accuracy = round(float(scores[idxs[0][i]]), 4)
                className = self.classNameList[int(classification[idxs[0][i]])]

                bbox = transformed_anchors[idxs[0][i], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)

                newX1 = max(min(x1, width), 0)
                newY1 = max(min(y1, height), 0)
                newX2 = max(min(x2, width), 0)
                newY2 = max(min(y2, height), 0)

                cv2.rectangle(originImage, (int(newX1), int(newY1)), (int(newX2), int(newY2)), (255, 0, 0), 1)

                tmpResult = {
                    "className": className,
                    "accuracy": accuracy,
                    "cursor": 'isRect',
                    "needCount": 2,
                    "position": [
                        {"x": newX1, "y": newY1},
                        {"x": newX2, "y": newY2},
                    ]
                }

                result.append(tmpResult)

            cv2.imwrite("./originImage.png", originImage)

            logger.info(result)
            trasferTime = time.time() - transferOutputStartTime
            logger.debug(f"Output Format Transfer Success, Duration : {round(trasferTime, 4)} sec")

            totalTime = time.time() - totalStartTime
            logger.info(f"Finish Model Predict, Duration : {round(totalTime, 4)} sec")
            logger.info("-"*100)

        except Exception as e:
            logger.error(f"Error :{str(e)}")
            logger.error(f"Traceback : {str(traceback.format_exc())}")

        return result, heatMapImage


if __name__ == "__main__":
    pathInfo = {
        "modelPath": "/data/sungmin/retinanet",
        # "weightPath": "/data/mjkim/retinanet/originWeight",
        "weightPath": "/data/sungmin/retinanet/weight",
    }

    path = "/data/sungmin/retinanet/sample/img/cat_0.jpg"
    # path = "/data/mjkim/retinanet/sample/img/cat-7.jpg"
    # img = cv2.imread("/data/mjkim/deeplabv3/catdog.png")
    img = cv2.imread(path)
    p = Predictor(pathInfo)

    # while True:
    predResult, heatMapImage = p.runPredict(img)
    print(predResult)
