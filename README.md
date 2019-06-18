![mahua](https://ichef.bbci.co.uk/news/660/cpsprodpb/146E7/production/_105178638_gettyimages-898368424.jpg)
##What's propose of this project?
This is a transfer learning implementation of classification and dectetion model for 15 Austrilia Birds on video source image.

Different from image detection task, to track the target object, a video detection task relies on a faster processing speed to handle more frames in each second.  For a mobile device, the computation speed is limited. Therefore, some lightweight detection models are proposed to increase the FPS on a mobile device. However, most of those models sacrifice the accuracy on classification.

To handle this problem, in our implementation, a detection model is only used to detect the position of birds and a classification model with higher accuracy is used to classify bird less frequently. Because the bird species don't frequently change on videos, less frequency on classification would have little inference.  However, a higher accuracy on video source images is needed to reduce the error on classification. 

## Datasets:
* Image source dataset: Contains about 15000 photographs scawled from webset, this dataset is the source domain of transferlearning
    * dataset for train https://drive.google.com/open?id=1C0W9ksD1TjNmaPkV04MMxL2dljk0CsDH
    * dataset for test https://drive.google.com/open?id=108nK0qQAqLqwIvgar1UEyZuQlj-tT_5k
    * dataset for validation: https://drive.google.com/open?id=1pRDwfVtARONF9W4NpqyY4tgg0Dx14hU8
* Video source dataset: This dataset contains 3000 croped images from birds videos 
    * dataset for train: https://drive.google.com/open?id=1idsmgggFkDbk6Ios-aOo6tU7mdu2n-Jd
    * dataset for test: https://drive.google.com/open?id=1nAuju4k0fyJBRdcYfoyrJneO1j-u-rF1
    * dataset for validation: https://drive.google.com/open?id=1uIivr32-OHa64FitjpzFjJNSgY2w1OJW
* Detection dataset: This task relies on VOC2012 which is a public datase to to recognize objects from a number of visual object classes in realistic scenes. The bird is one of the object that required to detect in this dataset.
    * See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    
##What functions this repo provides？

* A Resnet and Densenet on image source recognition task

* A transfer learning from image source datset to video source dataset based on DANN

* A Mobilenet-ssd V2 to detect the position of bird training on voc2012

* Some transfermation of those models from pytorch to ONNX and CoreML for IOS device

* A matlibplot is given to describe the traning loss and accuracy of training process

* Confusion matrix and classification_report is provide on evaluation


##How to use?
* You neet to install the Dependencies on requirements.txt using 'pip install requirements.txt'

* If you don't have CUDA environment, you need to delete .cuda() in the code to run on CPU

* To run bird recognition tasks on imageset, before you use command 'python train.py',you need to set following parameter:
    * img_address : Address of image training dataset
    * model_saving_adress: The default name for model
    * test_address: the address of test dataset

* To transfer learning model using DANN, before you use command 'python domain_adaptation/train_two_datasets.py',you need to set following parameter:
    * img_train_address : Address of image training dataset
    * image_test_address : Address of image test dataset
    * video_train_address : Address of video training dataset
    * video_test_address : Address of video test dataset
* Threre will be two sperate nets on DANN, c_net corresponds to the classification net and d_net represents the the discriminator in DANN.
* The default models_saving_address is this repo, this can be modified by:
                model_instance.save_model(c_net_path='DANN_IMAGE_accuracy{0}_c_net'.format(standard),d_net_path='DANN_IMAGE_accuracy{0}_d_net'.format(standard))
in train() method


* To evaluate the result of models, before you use command 'python domain_adaptation/train_two_datasets.py',you need to set following parameter:
    * base_net : The address of pre-trained model 
    * validation : Address of validation dataset
    * if you want to evaluate the models preformance on DANN 
        use following code to load the model:
            model = transfor_net.DANN(base_net='ResNet101',use_bottleneck=True, bottleneck_dim=256,
                                    class_num=15,
                                    hidden_dim=1024,
                                    trade_off=1.0, use_gpu=True)
            model.load_model('address of feature generator and Classifier:c_net',  'Address of d_net')
            model.set_train(False)
    * if you intent to evaluate hte models on resnet or densnet you can simply load the models by:
             model = torch.load('D:/model/resnet101_0.9606666666666667.pkl')
             model.eval()
* To run the mobilenet SSD v2 : you should read https://github.com/qfgaohao/pytorch-ssd and run 'python vison/train_SSD.py'




##Thanks to:
* [OswinGuai](https://github.com/OswinGuai)
* [Hao qfgaohao](https://github.com/qfgaohao/)
