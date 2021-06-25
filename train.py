import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import time
import os
import copy
# from model import MultiLabelModel
from torch.utils.data import Dataset, DataLoader
# from loss import FocalLoss




from model import *
from dataset import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


model_path='./model/'
best_path=model_path+'model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    




def train_model(model, dataloaders, criterion_clss,criterion_l1, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss=999999999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        b_no=1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            # running_loss3 = 0.0
            # running_corrects = 0
            # running_corrects1 = 0
            # running_corrects2 = 0
            # running_corrects3 = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if b_no%100==0:

                  print('%s batch'%str(b_no))
                b_no+=1
                inputs = inputs.to(device)
                true_bbox,true_clss = labels
                true_bbox=true_bbox.to(device)
                true_clss=true_clss.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        # outputs, aux_outputs = model(inputs)
                        bbox,clss = model(inputs)
                        clss_pred = clss.squeeze(2)
                        clss_pred = clss_pred.swapaxes(1,2)
                        # print(clss_pred.shape,true_clss.shape)
                        classification_loss = criterion_clss(clss_pred,true_clss)

                        bbox_true=torch.flatten(true_bbox)
                        bbox_pred=bbox.squeeze(2)
                        bbox_pred = bbox_pred.swapaxes(1,2)
                        bbox_pred=torch.flatten(bbox_pred)
                        # print(bbox_pred.shape,bbox_true.shape)
                        detection_loss = criterion_l1(bbox_pred,bbox_true)
                        loss = loss1 + loss2
                    else:
                        bbox,clss = model(inputs)
                        clss_pred = clss.squeeze(2)
                        clss_pred = clss_pred.swapaxes(1,2)
                        # print(clss_pred.shape,true_clss.shape)
                        classification_loss = criterion_clss(clss_pred,true_clss)

                        bbox_true=torch.flatten(true_bbox)
                        bbox_pred=bbox.squeeze(2)
                        bbox_pred = bbox_pred.swapaxes(1,2)
                        bbox_pred=torch.flatten(bbox_pred)
                        # print(bbox_pred.shape,bbox_true.shape)
                        detection_loss = criterion_l1(bbox_pred,bbox_true)
                        loss = detection_loss + classification_loss
    #                         print(loss)
                            
                            
                        # _, preds1 = torch.max(output1, 1)
                        # _, preds2 = torch.max(output2, 1)
                        # _, preds3 = torch.max(output3, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss1 += classification_loss.item() * inputs.size(0)
                    running_loss2 += detection_loss.item() * inputs.size(0)
                    # running_loss3 += loss3.item() * inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    # running_corrects1 += torch.sum(preds1 == labels[:,0].data)
                    # running_corrects2 += torch.sum(preds2 == labels[:,1].data)
                    # running_corrects3 += torch.sum(preds3 == labels[:,2].data)
    #                 running_corrects += torch.sum(preds == labels.data)

                epoch_loss1 = running_loss1 / len(dataloaders[phase].dataset)
                epoch_loss2 = running_loss2 / len(dataloaders[phase].dataset)
            # epoch_loss3 = running_loss3 / len(dataloaders[phase].dataset)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc1 = running_corrects1.double() / len(dataloaders[phase].dataset)
            # epoch_acc2 = running_corrects2.double() / len(dataloaders[phase].dataset)
            # epoch_acc3 = running_corrects3.double() / len(dataloaders[phase].dataset)
            

            print('{} classification Loss: {:.4f}'.format(phase, epoch_loss1))
            print('{} detection Loss: {:.4f}'.format(phase, epoch_loss2))
            # print('{} Sleeve Length Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss3, epoch_acc3))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))



            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print('Saving at {} Epoch'.format(epoch))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), best_path)
            # if phase == 'val':
            #     val_acc_history.append((epoch_acc1,epoch_acc2,epoch_acc3))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




def main(m,criterion_clss,criterion_l1,num_epochs,lr_rate,optimizer,data_transforms,batch_size):

    

    
    
    
    train_df=pd.read_csv('train.csv')
    val_df=pd.read_csv('test.csv')
    train_dataset=MyDataset(train_df,data_transforms['train'])
    valid_dataset=MyDataset(val_df,data_transforms['val'])


    train_dl=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl=DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)




    data_loaders={'train':train_dl,'val':valid_dl}
    
    model=train_model(m,data_loaders, criterion_clss,criterion_l1, optimizer, num_epochs=num_epochs, is_inception=False)

    torch.save(model.state_dict(), model_path+'last_epoch.pth')

    # with open(model_path+'val_acc_hist.pkl', 'wb') as f:
    #     pickle.dump(val_acc_history, f)


if __name__ == '__main__':


    # criterion=FocalLoss()
    # criterion=nn.CrossEntropyLoss()
    # criterion = MultiBoxLoss()

    num_epochs=100
    lr_rate=0.001

    # Input SIze
    input_size=(32,64)
    batch_size=32
    m=Detector(ResidualBlock, [2, 2]).to(device) 

    # m.cuda();
    optimizer = torch.optim.Adam(m.parameters(), lr=lr_rate)

    pos_weight = torch.ones([11]).to(device)
    criterion_clss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    criterion_l1 = loss = nn.MSELoss()

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size[0],input_size[1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size[0],input_size[1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    

    
    main(m,criterion_clss,criterion_l1,num_epochs,lr_rate,optimizer,data_transforms,batch_size)
