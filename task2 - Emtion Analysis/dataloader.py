from dataset import MyData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def generateLoadedData():
    bsz = 50
    train_dataset = MyData("data/2018-EI-reg-En-train/", True, bsz * 10)
    #test_dataset = MyData("test_for_later", False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=True)
    #test_loader = DataLoader(dataset=test_dataset,batch_size=bsz, shuffle=False)

    return train_loader, train_dataset

#generateLoadedData()
