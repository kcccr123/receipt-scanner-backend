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

Training scripts for our BART model are located in the `BART` folder in the root directory.

BART is used for word and sentence correction. In our case, product names. 
The data required is a CSV containing words you expect BART will need to learn to correct. The script takes these words and generates improper versions of them for BART to learn for.

Open `trainBart.py`.

At the top of the script, replace each of the folder paths, `csv_path`, `model_dir`, and `save_dir`.
Hyper parameters can be tuned at line 87.

Run `python trainBart.py` to run the script.

Once the training concludes, the new model will be saved to the selected model directory.

### YOLOv8

Training tools for our YOLOv8 model are located in the `RSYOLOV8` folder.

The main training script is `receiptScannerYolo.ipynb`  
To setup and run the training script:

**Data Folder**

1. Create a folder on your device.
2. Inside that folder, create another folder named `unsplit`.
3. Create two folders inside the `unsplit` folder named `images` and `labels`.
4. Place your images and labels in there respective folders.

**Set path in config.yaml**

1. Open the `config.yaml` file located in the `RSYOLOV8` folder.
2. Copy the absolute path of the data folder you created in the last section and replace the `path` field.

**Run the training script**

1. Open the `receiptScannerYolo.ipynb` training script
2. Read the instructions and run each cell up to and including the "Train the Model" cell.

For more information regarding training parameters, refer to the Ultralytics documentation for YOLOv8:  
[YOLOv8 Tuning Documentation](https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters)

### RCNN

Training toolds for RCNN model are located in the `RCNN` folder.

To train, simply run the training script `TrainModelGREY.py` in the terminal.

To setup the training script:

**Data Folder**

1. Create a folder on your device.
2. Place traiing images inside that folder.
3. In the same folder, create a .jsonl file containing the labels in the format `{"file_name":"image.jpg","text":"label"}`.

**Set path in training script**

1. In `TrainModelGREY.py`, locate `data_path` and `model_path` fields at the begining of the script.
2. Copy the absolute path of the data folder created in the previous section and the dictionary to save the model, then replace the `data_path` and `model_path` fields respectively.

### Flask Server

The backend code for our application is located inside the `server` folder.

The server can be run in development mode using Flask or built as a production server with Gunicorn, containerized within a Docker image.

With the new chatGPT integration, an OpenAI API key should be added as an environment variable. 
Create a `.env` file inside the server folder. Create the variable:

`OPENAI_API_KEY=<Your API KEY>`

The utilities inside `gpt_utils.py` will load and use the given key.

**Run a development server**

1. Open a terminal and CD into the `server` folder.
2. CD into the `api` folder.
3. Run the command: `python flask_server.py`

The server is now running, and you can make local RESTful requests to the endpoints served by the server.  
The default URL used by Flask is: `http://127.0.0.1:5000`

**Export as Docker image**

1. Start your Docker daemon.
2. Open a terminal and CD into the `server` folder, where the dockerfile is located.
3. Run the command: `docker build -t <image_name>:<tag> .` -> Replace `<image_name>` and `<tag>` with a name and tag of your choice.

Your Docker image should now be building.  
You can run your new image in a container using: `docker run -d --name <container_name> -p 5000:5000 <image_name>:<tag>`  
Replace `<container_name>` with a container name of your choice, and `<image_name>`, `<tag>` with the image name and tag you chose in the previous step.

You should now be able to make requests to `http://127.0.0.1:5000`

In the docker run command, the host port 5000 is mapped to container port 5000.  
You can change the host port to any value you prefer. For example, to bind the host port 8080 to container port 5000, use:  
`<Host Port>:5000 -> 8080:5000`

Full example command: `docker run -d --name newContainer -p 8080:5000 flask_server:latest`

Local requests will now be served at the URL: `http://127.0.0.1:8080`

<!-- CONTACT -->

## Contact

Feel free to contact us at:

@Kevin Chen - kevinz.chen@mail.utoronto.ca\
@Gary Guo - garyz.guo@mail.utoronto.ca

<p align="right">(<a href="#readme-top">back to top</a>)</p>
