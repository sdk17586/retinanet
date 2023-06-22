import os
import sys
import json
import gc
import time
import inspect
import traceback
import pytorch_lightning as pl

from loguru import logger
from pydantic import validate_arguments
from pydantic.dataclasses import dataclass

basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.extend([basePath, os.path.join(basePath, "model")])

from sender import sendMsg
from logger import Logger
from callback import Callback
from creator import createDataLoader
from checker import gpuChecker, initGpuChecker
from maker import makeDir, makeClasses, makeClassInfo

gpuNo = initGpuChecker()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuNo


@validate_arguments
def paramToData(param: dict):
    return param["dataInfo"], param["servInfo"], param["pathInfo"], param["selectedHyperParam"], param["useClassBalancer"]


@validate_arguments
def saveDataToJson(dataInfo: dict, servInfo: dict, pathInfo: dict, selectedHyperParam: dict):
    savePath = pathInfo["modelPath"]
    paramList = list(inspect.signature(saveDataToJson).parameters.keys())

    try:
        for param in paramList:
            saveParamPath = os.path.join(savePath, f"{param}.json")
            with open(saveParamPath, "w") as f:
                json.dump(locals()[param], f)

    except Exception:
        logger.error(f"err : {traceback.format_exc()}")


@dataclass
class Trainer():
    dataInfo: dict
    servInfo: dict
    pathInfo: dict
    selectedHyperParam: dict
    useClassBalancer: bool

    def __post_init__(self):
        
        self.setLogger()
        self.setGpu()
        self.setHyperParam()
        self.setClassInfo()
        self.setDataset()

    # Setting Logger
    def setLogger(self):
        self.Log = Logger(logPath=self.pathInfo["logPath"], logLevel="info")

    # Setting CPU / GPU
    def setGpu(self):
        global gpuNo

        gpuIdx = str(self.selectedHyperParam.get("gpuIdx", "auto"))
        self.device, self.deviceType = gpuChecker(log=self.Log, gpuIdx=gpuIdx, gpuNo=gpuNo)

    def setHyperParam(self):
        # hyperParam
        self.epoch = int(self.selectedHyperParam.get("epoch", 100))
        self.batchSize = int(self.selectedHyperParam.get("batchSize", 1))
        self.imageSize = 608
        self.grayScale = 1 if self.selectedHyperParam.get("grayScale", "true") == "true" else 3
        self.optimizer = str(self.selectedHyperParam.get("optimizer", "adam"))
        self.learningRate = float(self.selectedHyperParam.get("learningRate", 0.01))
        self.trainPathList = self.pathInfo["trainPathList"]

        if len(self.trainPathList) == 0:
            raise Exception(f"TrainPath Length is {len(self.trainPathList)}, No Data!")

        self.trainRate = self.selectedHyperParam.get("trainRate", {"train": 70, "validation": 30})
        self.useClassBalancer = False

        # early stopping
        self.earlyStopping = True if self.selectedHyperParam.get("earlyStopping", "true") == "true" else False
        self.patience = int(self.selectedHyperParam.get("patience", 10))
        self.monitor = str(self.selectedHyperParam.get("monitor", "valAccuracy"))

        # path
        self.classInfo = self.dataInfo["classInfo"]
        self.weightPath = str(self.pathInfo.get("weightPath", "/app/weight"))
        self.modelWeightPath = os.path.join(self.weightPath, "weight.pth")

        self.originWeightPath = "/data/sungmin/retinanet/weight"
        self.backboneWeightPath = "/app/backboneWeight"

        # make weightPath
        makeDir(self.weightPath)

        # etc
        self.pretrained = True if os.path.isfile(os.path.join(self.originWeightPath, "weight.pth")) else False
        self.dataType = str(self.dataInfo.get("dataType", "image"))
        self.purposeType = "detection"

    # Setting classIdList, classNameList, colorList, TransferModel ClassCount checkmakeClassInfo
    def setClassInfo(self):
        self.classIdList, self.classNameList, self.colorList, self.preChannel, self.preNumClasses = makeClassInfo(self.pretrained, self.originWeightPath, self.classInfo)

    # Setting Files for train / DataLoader
    def setDataset(self):
        self.Log.info("Create Dataset...")
        datasetStartTime = time.time()

        # make classes.names / classes.json file
        makeClasses(self.weightPath, self.classNameList, self.classIdList, self.colorList, self.imageSize, self.grayScale, self.purposeType)

        # make trainLoader
        self.trainDataLoader, self.validDataLoader = createDataLoader(self.trainPathList, self.classInfo, self.classNameList, self.imageSize, self.batchSize, self.grayScale, self.trainRate)

        datasetTime = time.time() - datasetStartTime
        self.Log.info(f"Finish Create Dataset, Duration : {round(datasetTime, 4)} sec")

        gc.collect()

    def runTrain(self):

        trainStartTime = time.time()

        callback = Callback(param=self)

        if self.deviceType == "cpu":
            trainer = pl.Trainer(
                max_epochs=self.epoch,
                accelerator=self.deviceType,
                enable_progress_bar=False,
                enable_checkpointing=False,
                logger=None
            )
        else:
            trainer = pl.Trainer(
                max_epochs=self.epoch,
                accelerator=self.deviceType,
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
                logger=None
            )

        trainer.fit(callback)

        trainTime = time.time() - trainStartTime
        self.Log.info(f"Finish Model Training, Duration : {round(trainTime, 4)} sec")


if __name__ == "__main__":

    try:
        # img
        data = '{"runType":"train","dataInfo":{"purposeType":"detection","classInfo":[{"classId":"e99f5","className":"cat","color":"#ef8783","desc":"","dsId":"e3375","dsName":"[CLF] LG\uc5d4\uc194_Train_dataset","isClick":false,"showTf":true},{"classId":"76e2e","className":"dog","color":"#0000ff","desc":"","dsId":"e337e","dsName":"[CLF] LG\uc5d4\uc194_Train_dataset","isClick":false,"showTf":true}],"mlType":"vision","sourceType":"file"},"servInfo":{"servIp":"string","servPort":"string","sendResultUrl":"string","sendStatusUrl":"string"},"pathInfo":{"trainPathList":["/data/sungmin/retinanet/sample/img"],"modelPath":"/data/sungmin/retinanet","weightPath":"/data/sungmin/retinanet/weight","logPath":"/data/sungmin/retinanet/log/log.log","pvPath":"/data/sungmin/retinanet"},"selectedHyperParam":{"epoch":100,"batchSize":1,"trainRate":{"train":70,"validation":30},"grayScale":"false","optimizer":"adam","learningRate":0.001,"earlyStopping":"true","patience":10,"monitor":"loss","gpuIdx":"auto"},"useClassBalancer":false}'

        param = json.loads(data)

        # Setting info from data
        dataInfo, servInfo, pathInfo, selectedHyperParam, useClassBalancer = paramToData(param)
        sendResultUrl = f'{servInfo["servIp"]}:{servInfo["servPort"]}/{servInfo["sendResultUrl"]}'

        # Save dataInfo, servInfo, pathInfo, hyperParam to json
        saveDataToJson(dataInfo, servInfo, pathInfo, selectedHyperParam)

        # Setting Trainer
        trainer = Trainer(dataInfo=dataInfo, servInfo=servInfo, pathInfo=pathInfo, selectedHyperParam=selectedHyperParam, useClassBalancer=useClassBalancer)
        trainer.runTrain()

        sys.exit(0)

    except Exception:
        err = traceback.format_exc()
        logger.error(f"err : {err}")

        sys.exit(1)
