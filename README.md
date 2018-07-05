# deeplens-magiccards

Generate a SageMaker object classification / detection model in SageMaker.

It's really confusing where to begin.

[Build your own object classification model in SageMaker and import it to DeepLens](https://aws.amazon.com/blogs/machine-learning/build-your-own-object-classification-model-in-sagemaker-and-import-it-to-deeplens/)

"To build you own model, you first need to identify a dataset. You can bring your own dataset or use an existing one. In this tutorial, we show you how to build an object detection model in Amazon SageMaker using Caltech-256 image classification dataset."

train and validation sets for Caltech-256:
http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec
http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec

For this tutorial, we will use ResNet network. ResNet is the default image classification model in Amazon SageMaker.

What is a .rec file and how do I make one?
Creating a .rec file involves .lst files
Here's some amazon info on it (under "Training with Image Format"):
https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html

From https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html:
```
Most Amazon SageMaker algorithms work best when you use the optimized protobuf recordIO format for the training data. Using this format allows you to take advantage of Pipe mode when training the algorithms that support it. File mode loads all of your data from Amazon Simple Storage Service (Amazon S3) to the training instance volumes. In Pipe mode, your training job streams data directly from Amazon S3. Streaming can provide faster start times for training jobs and better throughput. With Pipe mode, you also reduce the size of the Amazon Elastic Block Store volumes for your training instances. Pipe mode needs only enough disk space to store your final model artifacts.
```

This table of Image Classification algorithms seems useful:
https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html

We're doing Image Recognition or Object Detection, not Image Classification.

SSD resizes training images to 300x300?
https://github.com/zhreshold/mxnet-ssd/issues/81

Going through these steps: https://cosminsanda.com/posts/counting-object-with-mxnet-and-sagemaker/

# Doing stuff

```
$ mkvirtualenv deeplens-magiccards
(deeplens-magiccards) $ pip install ipython
(deeplens-magiccards) $ pip install jupyter
```
Because I am a fool who installed anaconda, I have to start jupyter like this:
```
(deeplens-magiccards) deeplens-magiccards $ /Users/agussman/.virtualenvs/deeplens-magiccards/bin/jupyter notebook
```

ImageNet Dining Room Tables:
http://image-net.org/synset?wnid=n03200906#
http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03201035




# Tips and Troubleshooting

If you need to change your WiFi network, you'll need to use a pin to hit the "Reset" button on the back. This will re-activate the DeepLens' WiFi network. You can then reconnect to it and add the new network (it will remember previous networks).


# References
1) [Build your own object classification model in SageMaker and import it to DeepLens](https://aws.amazon.com/blogs/machine-learning/build-your-own-object-classification-model-in-sagemaker-and-import-it-to-deeplens/)
2) [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
3) [AWS DeepLens Extensions: Build Your Own Project](https://aws.amazon.com/blogs/machine-learning/aws-deeplens-extensions-build-your-own-project/)
4) [ResNet for Traffic Sign Classification With PyTorch](https://towardsdatascience.com/resnet-for-traffic-sign-classification-with-pytorch-5883a97bbaa3)
5) [Using TensorFlow for ALPR/ANPR](https://matthewearl.github.io/2016/05/06/cnn-anpr/)
6) [Training Data Formats](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html)
7) [MXNet Iterators and im2rec.py](https://mxnet.incubator.apache.org/tutorials/basic/data.html?highlight=im2rec)
8) [Counting Objects with MXNet and SageMaker (good read, involves using your own data)](https://cosminsanda.com/posts/counting-object-with-mxnet-and-sagemaker/)
9) [Implementing Object Detection in Machine Learning for Flag Cards with MXNet](https://medium.com/ymedialabs-innovation/implementing-object-detection-in-machine-learning-for-flag-cards-with-mxnet-6bc276bb0b14)
10) [MXNET BYOM for SageMaker](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/mxnet_mnist_byom/mxnet_mnist.ipynb)
11) [Classifying traffics igns with mxnet](https://www.oreilly.com/ideas/classifying-traffic-signs-with-mxnet-an-introduction-to-computer-vision-with-neural-networks)
12) [Multi Object Detection with MXNet Gluon and SSD](https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html)
13) [imgaug- python library for augmenting images for machine learning projects](https://github.com/aleju/imgaug)
14) [Augmentor: image augmentation library for Python](https://github.com/mdbloice/Augmentor)
14) [Python library for grabbing images from the web and randomly modifying them](https://github.com/tomahim/py-image-dataset-generator)
15) [SynDataGeneration: cut paste and learn](https://github.com/debidatta/syndata-generation)
