# Camera activity classification using Machine Learning

Project developed for the class [SEM5952 - Neural Networks and Machine Learning](https://uspdigital.usp.br/janus/componente/disciplinasOferecidasInicial.jsf?action=3&sgldis=SEM5952&idioma=en), from the Graduate Program in Mechanical Engineering at EESC-USP.

### Autors:
Carlos André Persiani Filho

[Leonardo de Souza Bornia](https://github.com/LeoBelmont)

[Leonardo Simião de Luna](https://github.com/leonardosimiao)

María José Burbano Guzmán

[Tarcísio Ladeia de Oliveira](https://github.com/TarcisioLOliveira)

[Valentin Mohl](https://github.com/Kartolon)

Vincent Edward Wong Díaz

## Problem Statement

The São Carlos campus of the University of São Paulo features a large array of cameras, but not enough security personnel to monitor it. 
Therefore, a system which could bring attention to cameras with possible anomalous activity could make the system more effective. 
We present a proof-of-concept implementation of such a system, based on Inception v2 neural network.

## How it works
With a given set of video or screen inputs (the cameras), the algorithm uses Inception v2 to detect classes of interest (a person, or selected objects). Each class has a weight associated with it. The code ranks the inputs according to an attention score, given by the weighted sum of all the detections. The cameras' streams are composed in a display that puts the highest-rated camera into focus, taking the biggest portion of the display.

On the video streams, the interactions between people and objects of interest are highlighted. Their bounding boxes appear in red to bring more attention to the action. As a way to demonstrate all the instances detected in the videos, the other detections appear in blue.

The code generates a log that lists the cameras' attention scores and all the interacted objects, to aid in posterior activity supervision.

## How it was made

The project uses a [Faster-RCNN Inception v2](https://github.com/tensorflow/models/blob/ac8d0651935ecf8a5e7c73ff4aaa97ca1b3623b5/research/object_detection/g3doc/tf1_detection_zoo.md) network, trained using the [COCO dataset](https://cocodataset.org/#home). The algorithm was tested and debugged using the [VIRAT Video Dataset](https://viratdata.org/) as inputs.

## How to use it

In the directory, execute command below to run the main code. All the camera inputs must be altered in code, inserted as the list ***cameras***.

    python main.py
    
File [settings.py](https://github.com/leonardosimiao/Neural-Network-Class/blob/master/settings.py) contains most of the configurations for the algorithm operation. Listed in [requirements.txt](https://github.com/leonardosimiao/Neural-Network-Class/blob/master/requirements.txt) are the libraries that require especific versions.
