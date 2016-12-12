# SDC-Term1-Docker-GPU
Docker instance for Udacity SDC ND Term1 - GPU version

The image is built using Ubuntu14.04/cuda8.0/cudnn5 as base. This is based on Ubuntu 14.04. Image based on Ubuntu 16.04 is still work in progress.
Refer: 	https://hub.docker.com/r/nvidia/cuda/
	      https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements


Nvidia-docker need not be seperately installed. Dockerfile builds an image for nvidia/cuda/cudnn as well.

    Ubuntu 		# 14.04 for now. 16.04 should be available shortly
    CUDA 8.0
    cuDNN v5
    python_version=3.5.2
    miniconda_version=4.1.11
    tensorflow_version=0.12.0rc0
    Keras_version=1.1.2
    iPython/Jupyter Notebook
    Numpy, SciPy, Pandas, Scikit Learn, Matplotlib
    opencv3, moviepy


# Prerequisites
Docker
  Install Docker following the installation guide for your platform.
  For details- https://docs.docker.com/engine/installation/

  1. docker installation
  sudo apt-get update
  sudo apt-get install apt-transport-https ca-certificates
  sudo apt-key adv \
  --keyserver hkp://ha.pool.sks-keyservers.net:80 \
  --recv-keys 58118E89F3A912897C070ADBF76221572C52609D

  echo "deb https://apt.dockerproject.org/repo ubuntu-xenial main" | sudo tee /etc/apt/sources.list.d/docker.list

  sudo apt-get update
  sudo apt-get install linux-image-extra-$(uname -r) linux-image-extra-virtual
  sudo apt-get install docker-engine
  sudo service docker start

  setup docker for another user
  sudo groupadd docker
  sudo gpasswd -a ${USER} docker
  sudo service docker restart

  check docker installation
  docker run hello-world
  

  2. Nvidia-drivers
  Install supported nvidia drivers - 367 and above


  3. CUDA-capable GPU
  Check if your system has CUDA-capable GPU
  Refer https://developer.nvidia.com/cuda-gpus


# Build the image
Place the Dockerfile-<version> (rename the file to "Dockerfile") in the current working directory and execute -
$ docker build -t <dockerhub userid>/udacitysdc-term1:gpu .

Once the image is built successfully, do a quick check with,
$ docker images

You should find two images listed. 
<dockerhub userid>/udacitysdc-term1:gpu
nvidia/cuda


# Run the docker image in a container
Once we've built the image, we have all the required softwares installed in it. We can now spin up one or more containers using this image.

$ nvidia-docker run -it -v $PWD:/src -p 8888:8888 <dockerhub userid>/udacitysdc-term1:gpu bash

Note the use of nvidia-docker rather than just docker

Parameters
-it
This creates an interactive terminal you can use to iteract with your container

-p 8888:8888 -p 6006:6006
This exposes the ports inside the container so they can be accessed from the host. The format is -p <host-port>:<container-port>. The default iPython Notebook runs on port 8888 and Tensorboard on 6006

-v $PWD:/src
This shares the folder $PWD on your host machine to /src inside your container. Any data written to this folder by the container will be persistent. You can modify this to anything of the format -v /local/shared/folder:/shared/folder/in/container/. See Docker container persistence

<dockerhub userid>/udacitysdc-term1:gpu
This the image that you want to run. The format is image:tag. In our case, we use the image udacitysdc-term1 and tag gpu or cpu to spin up the appropriate image

bash
This provides the default command when the container is started. Even if this was not provided, bash is the default command and just starts a Bash session. You can modify this to be whatever you'd like to be executed when your container starts. For example, you can execute 
nvidia-docker run -it -p 8888:8888 <dockerhub userid>/udacitysdc-term1:gpu jupyter notebook. 

This will execute the command jupyter notebook and starts your Jupyter Notebook for you when the container starts



# Sanity check

1.   Check opencv3. Run "Lane Finding Demo" as in the below link
	    https://medium.com/self-driving-cars/lane-finding-demo-bd834d7928a9#.i32ooenzi

      a) Clone "lane-demo" package to your host folder
      b) From the nvidia-docker terminal, run - 
	        $ jupyter notebook --port=8888 --ip=0.0.0.0
      c) From the browser, type-in "localhost:8888"
      d) Run "Lane Finding Demo" and check if the demo runs successfully without errors.

2.   Check tensorflow, gpu, keras
      From the nvidia-docker terminal, run -
	    $ curl -sSL https://github.com/fchollet/keras/raw/master/examples/mnist_mlp.py | python

      a) Verify that terminal prints message for tensorflow and esblishing connections to GPU
      b) check the output.

60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 0
5s - loss: 0.4356 - acc: 0.8716 - val_loss: 0.1863 - val_acc: 0.9421
.....
.....
.....
Epoch 19
5s - loss: 0.0383 - acc: 0.9874 - val_loss: 0.0705 - val_acc: 0.9820
Test score: 0.0704572771238
Test accuracy: 0.982


# References:
1. https://docs.docker.com/engine/installation/
2. https://hub.docker.com/r/nvidia/cuda/
3. https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements
4. https://github.com/saiprashanths/dl-docker/blob/master/README.md
5. https://github.com/fchollet/keras/blob/master/docker/Dockerfile
6. http://ermaker.github.io/blog/2015/09/08/get-started-with-keras-for-beginners.html
7. http://www.webupd8.org/2016/06/how-to-install-latest-nvidia-drivers-in.html
