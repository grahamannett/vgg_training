# Information

This image will train a VGG16 model for the [imagenette](https://github.com/fastai/imagenette) dataset.  Almost all the information out there is for using a pretrained model or training on a very simple image dataset.  This image attempts to create a recreate-able way to generate the weights for the full model in keras.

# How to build

The following command will build the image:

```sh
docker build -t vgg_training:16 -f Dockerfile .
```

This will get the .tgz data and put it into the image for later use.

# How to run

By default this trains on CPU.  To train with GPU, its advised to use the nvidia-docker runtime.  Once that is installed, you can run something such as: `sudo docker run --rm --gpus all vgg_training:gpu`
To use that follow the instructions at: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

To run simply use `docker run vgg_training:16`

# What is expected

A trained model is trained in the image.  To save the model, mount the output folder in the image to somewhere, for instance: `docker run -v $(pwd):/root/output vgg_training:16` or set the ENV OUTPUT on run.