# SDC-Term1-Docker-GPU
Docker instance for Udacity SDC ND Term1 - GPU version

The image is built using Ubuntu14.04/cuda8.0/cudnn5 as base. Image based on Ubuntu 16.04 is still work in progress.

Refer: 	https://hub.docker.com/r/nvidia/cuda/


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

1. docker installation
  	Install Docker following the installation guide for your platform.
  	
	For details- https://docs.docker.com/engine/installation/

2. Nvidia-drivers
	Install supported nvidia drivers - 367 and above


3. CUDA-capable GPU
	Check if your system has CUDA-capable GPU
	
	Refer https://developer.nvidia.com/cuda-gpus


# Build the image
	Place the Dockerfile-<version> (rename to “Dockerfile”) in the current working directory and execute -
		$ docker build -t <dockerhub userid>/udacitysdc-term1:gpu .

	Once the image is built successfully, do a quick check with,
		$ docker images

	You should find two images listed. 
		1. <dockerhub userid>/udacitysdc-term1:gpu
		2. nvidia/cuda


# Run the docker image in a container
We can now spin up one or more containers using this image.

	$ nvidia-docker run -it -v $PWD:/src -p 8888:8888 <dockerhub userid>/udacitysdc-term1:gpu bash

Note the use of nvidia-docker rather than just docker.



# Sanity check

1.   Check opencv3. Run "Lane Finding Demo" as in the following link
	
	https://medium.com/self-driving-cars/lane-finding-demo-bd834d7928a9#.i32ooenzi
	
	Clone "lane-demo" package to your host folder ($PWD on host that maps to /src on Docker).
	From the nvidia-docker terminal, run - 

		$ jupyter notebook --port=8888 --ip=0.0.0.0

	From the browser, type-in "localhost:8888"
	
	Run "Lane Finding Demo" and check if the demo runs successfully without errors.

2.   Check tensorflow, gpu, keras
      From the nvidia-docker terminal, run -
		
		$ curl -sSL https://github.com/fchollet/keras/raw/master/examples/mnist_mlp.py | python

	Verify that terminal prints messages for tensorflow and esblishing connections to GPU.
	
	Check the output.

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
4. https://github.com/fchollet/keras/blob/master/docker/Dockerfile
5. http://ermaker.github.io/blog/2015/09/08/get-started-with-keras-for-beginners.html
