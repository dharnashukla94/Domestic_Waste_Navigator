# Domestic Waste Navigator
 
The docker image is uploaded on DockerHub. You can find the image [here](https://hub.docker.com/repository/docker/dharnashukla94/domestic_waste_navigator/)
or you can run the following cmd with docker installed:
```shell
$ docker pull dharnashukla94/domestic_waste_navigator
```

## Getting Started in 10 Minutes

- Clone this repo 
- Install requirements
- Run the script
- Go to http://localhost:8080
- Done!

## Run with Docker

With **[Docker](https://www.docker.com)**, you can quickly build and run the entire application in minutes :whale:

```shell
# 1. First, clone the repo
$ git clone https://github.com/dharnashukla94/Domestic_Waste_Navigator.git
$ cd Domestic_Waste_Navigator

# 2. Build Docker image
$ docker build -t domestic_waste_navigator .

# 3. Run!
$ docker run -p 8080:8080 domestic_waste_navigator
```

Open http://localhost:8080 and wait till the webpage is loaded.

## Local Installation

It's easy to install and run it on your computer.

Github does not allow files larger than 100 MB, You can download the model from [here](https://drive.google.com/file/d/1bxDDanKqu7toxu19dcnG7Nq99xpWy6OR/view?usp=sharing) just copy the downloaded model to models folder.

```shell
# 1. First, clone the repo
$ git clone https://github.com/dharnashukla94/Domestic_Waste_Navigator.git
$ cd Domestic_Waste_Navigator

# 2. Install Python packages
$ pip install -r requirements.txt

# 3. Run!
$ python main.py
```

Open http://localhost:8080 and wait till the webpage is loaded.
