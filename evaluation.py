import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from utils import utils
from domain_adaptation import transfor_net
from sklearn import metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report


BATCH_SIZE = 10

def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    print(y_true)
    print(y_pred)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()


def test(net,file,show,shuffle):
    print(net)
    # net.eval()
    net.set_train(False)
    test_data = torchvision.datasets.ImageFolder('C:/Users/lyyc/Desktop/BirdRecognition/video recognition_test',
    # test_data = torchvision.datasets.ImageFolder('D:/VALIDATION',
                                                 transform=transforms.Compose([
                                                     # utils.Padding(),
                                                     transforms.Resize(224),
                                                     # transforms.RandomCrop(224),
                                                     # transforms.RandomHorizontalFlip(),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
                                                )
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=shuffle)

    a = []
    print(test_data)
    true = []
    pred = []

    correct = 0
    all = 0
    for step, (b_x, b_y) in enumerate(data_loader):  # 分配 batch data, normalize x when iterate train_loader
        x = b_x.cuda()
        y = b_y.cuda()
        # print(x.shape)
        # print(x.data.shape)
        # output = net(x)  # cnn output
        output = net.predict(x)
        # print(output.data)
        # print(output)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        # print(pred_y=)
        all += BATCH_SIZE
        # print(pred_y)
        # print(y)
        for i in range(len(pred_y)):
            pred.append(int(pred_y[i]))
            true.append(int(y[i]))
            if pred_y[i] == y[i]:
                # print(int(pred_y[i]))
                correct += 1
            else:
                if show is True:
                    utils.show_from_tensor(b_x[i])
        print('{0}/{1}'.format(correct, all))
    print(correct/len(test_data))
    print(metrics.confusion_matrix(true, pred))
    return true, pred

if __name__ == '__main__':
    # BATCH_SIZE = 1
    # net = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
    # print(net.bottleneck_layer())
    c_net = torch.load('D:/model/DANN_accuracy0.8743386243386243_c_net_0.9853333333333333.pkl')
    model = transfor_net.DANN(base_net='ResNet101', use_bottleneck=True, bottleneck_dim=256, class_num=15,
                              hidden_dim=1024,
                              trade_off=1.0, use_gpu=True)
    model.load_model('D:/model/DANN_accuracy0.8703703703703703_c_net', 'D:/model/DANN_accuracy0.8743386243386243_d_net')
    # print(net)
    # test(net,'TEST0',True,False)
    # model = torchvision.models.densenet161(pretrained=False)
    # model.classifier = torch.nn.Linear(2208, 15)
    # model.load_state_dict(torch.load('densnet_0.94_dict.pth'))
    # model = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
    #
    # model.cuda()
    np.set_printoptions(precision=2)
    true, pred = test(model, 'video recognition_test', False, False)
    true.extend(true)
    pred.extend(pred)
    classname = ['Grey Butcherbird','Grey Fantail','House Sparrow ','Laughing Kookaburra','Little Wattlebird '
        ,'New Holland Honeyeater ','Noisy Miner','Pied Currawong','Rainbow Lorikeet','Red Wattlebire','Red-browed Finch'
        ,'Red-whiskered Bulbul','Silvereye','Spotted Pardalote','Spotted Turtle Dove']
    plot_confusion_matrix(true, pred, classes=classname,
                          title='Confusion matrix, without normalization')
    print(classification_report(true, pred, target_names=classname))
    plt.show()