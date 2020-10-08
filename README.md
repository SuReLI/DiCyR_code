# DiCyR: Disentangled cyclic reconstruction for domain adaptation
Official github repository for the paper Disentangled cyclic reconstruction for domain adaptation submited to ICLR 2021.
This repository contains the code and notebooks illustrating the experiments presented in the paper.

## Installation
First clone the repository:
```
git clone https://github.com/AnonymousDiCyR/DiCyR.git
cd DiCyR
```
Create a virtual environment:
```
conda env create -f environment.yaml
```
Create the data folder:
```
mkdir data
```

## Datasets
Download the 3D shapes data [here](https://console.cloud.google.com/storage/browser/3d-shapes) and copy it into the data folder.  
The German GTSRB dataset can be downloaded [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  
The Syn-signs dataset can be dowloaded [here](http://graphics.cs.msu.ru/en/node/1337/).

## Notebooks:
All notebooks are availables [here](https://github.com/AnonymousDiCyR/DiCyR/tree/main/DiCyR/notebooks).  
Each of them represents one experiment iteration.
