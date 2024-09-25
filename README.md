<!-- PROJECT LOGO
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
-->

<h3 align="center">Receipt Scanner Backend</h3>

  <p align="center">
    Backend and Machine Learning repository 
   <br />

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#installation">Installation</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li>
     <a href="#usage">Usage</a>
      <ul>
        <li><a href="#bart">BART</a></li>
        <li><a href="#yolov8">YOLOv8</a></li>
        <li><a href="#rcnn">RCNN</a></li>
        <li><a href="#flask-server">Flask Server</a></li>
      </ul></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This repository hosts the backend server code and utilities for training the various models used in the [Receipt Scanner](https://github.com/kcccr123/receipt-scanner) project.

### Built With

![Gunicorn](https://img.shields.io/badge/gunicorn-%298729.svg?style=for-the-badge&logo=gunicorn&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

<!-- INSTALLATION -->

## Installation

To run this project, please clone the repository.

### Prerequisites

Please ensure you have [Python](https://www.python.org/downloads/) 3.9 or higher installed.

A list of all required packages can be found in the requirements.txt file located in the requirements folder.
These packages can be quickly installed using the `install-packages.py` Python script located in the same folder.

To build an image of the server using the dockerfile, please install [Docker](https://docs.docker.com/engine/install/).

<!-- USAGE -->

## Usage

### BART


### YOLOv8

Training tools for our YOLOv8 model are located in the `RSYOLOV8` folder.

The main training script is `receiptScannerYolo.ipynb`   
To setup and run the training script:

**Data Folder**
1. Create a folder on your device.
2. Inside that folder, create another folder named `unsplit`.
3. Create two folders inside the `unsplit` named `images` and `labels`.
4. Place your images and labels in there respective folders.

**Set path in config.yaml**
1. Open the `config.yaml` file located in the `RSYOLOV8` folder.
2. Copy the absolute path of the data folder you created in the last section and replace the `path` field.

**Run the training script**
1. Open the `receiptScannerYolo.ipynb` training script
2. Read the instructions and run each cell up to and including the "Train the Model" cell.
  
For more information regarding training parameters, refer to the Ultralytics documentation for YOLOv8: [YOLOv8 Tuning Documentation](https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters)

### RCNN

### Flask Server



<!-- CONTACT -->

## Contact

Feel free to contact us at:

@Kevin Chen - kevinz.chen@mail.utoronto.ca\
@Gary Guo - garyz.guo@mail.utoronto.ca

<p align="right">(<a href="#readme-top">back to top</a>)</p>

 
