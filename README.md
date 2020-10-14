# Sign-Hawk
Sign Language Translator

Please see the readme in the respective folders to know more about them.

Diagrams related to project are in the docs folder

![Output from InterHand Model](https://github.com/SuhelNaryal/Sign-Hawk/blob/main/index.jpg)

This is an example of the outputs from interhand model


#Ongoing Works:

1) Words to sentence prediction dataset creation. Current progress can be found in sentence forlder.

2) sign language translation dataset creation which contains videos of people performing various signs. All videos are dowloaded from internet. The dataset has not been uploaded to cloud yet.

3) Current hand detection solution uses a palm detector which sometimes recognises ears as palms when there is only one hand in image. To overcome this issue a new model is being trained.
    - Tested models: Centernet HourGlass104 512x512, CenterNet Resnet101 V1 FPN 512x512
    - current models under trainig: CenterNet Resnet50 V1 FPN 512x512
    - details regarding models can be found @ https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    - models are trained using Tensorflow object detection API.
    - Datset used: http://vision.cs.stonybrook.edu/~supreeth/TV-Hand.zip
    - Training stats for current model are as follows(Outliers excluded)(Actual values are fainted lines)(Darker lines are values smoothened on a scale of 0.6)(x-axis: no of steps, y-axis: loss values):
    - Object Center Loss 
    ![Object Center Loss](https://github.com/SuhelNaryal/Sign-Hawk/blob/main/Loss_object_center_current.svg)
    - Box Offset Loss 
    ![Box Offset Loss](https://github.com/SuhelNaryal/Sign-Hawk/blob/main/Loss_box_offset_current.svg)
    - Box Scale Loss 
    ![Box Scale Loss](https://github.com/SuhelNaryal/Sign-Hawk/blob/main/Loss_box_scale_current.svg)
    - Total Loss 
    ![Total Loss](https://github.com/SuhelNaryal/Sign-Hawk/blob/main/Loss_total_loss_current.svg)
