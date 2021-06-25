import torch
from torchvision import  transforms
from PIL import Image
from model import *
from dataset import *


def predict(model,f):

    input_size= (64,32)
    data_transforms_pred = transforms.Compose([
        transforms.Resize((input_size[0],input_size[1])),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img=torch.unsqueeze(data_transforms_pred(Image.open(f).convert('RGB')),0)
    img = img.to(device)
    #pred=model(img)
    print(img.shape)

    bbox,clss = model(img)
    print(bbox.shape,clss.shape)
    clss_pred = clss.squeeze(3).squeeze(0)
    clss_pred = clss_pred.swapaxes(0,1)
    # bbox_true=torch.flatten(true_bbox)
    bbox_pred=bbox.squeeze(3).squeeze(0)
    bbox_pred = bbox_pred.swapaxes(0,1)
    # print(bbox_pred.shape,clss_pred.shape)
    # print(clss_pred)
    classes = torch.argmax(clss_pred, dim=1)
    print(classes)



m=Detector(ResidualBlock, [2, 2]).to(device) 
f='test/44.png'
best_model_wts = 'model/best.pth'
m.load_state_dict(torch.load(best_model_wts,map_location=torch.device('cpu')))
predict(m,f)s