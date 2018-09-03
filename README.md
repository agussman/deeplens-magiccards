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
(deeplens-magiccards) $ pip install numpy
(deeplens-magiccards) $ pip install matplotlib
$ pip install mxnet
(deeplens-magiccards) $ pip install Pillow
$ pip install opencv-python
```
Because I am a fool who installed anaconda, I have to start jupyter like this:
```
(deeplens-magiccards) deeplens-magiccards $ /Users/agussman/.virtualenvs/deeplens-magiccards/bin/jupyter notebook
```

ImageNet Dining Room Tables:
http://image-net.org/synset?wnid=n03200906#
http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03201035


In trying to get sagemaker to work, I created a Role via the sagemaker wizard. But then when I launched a notebook instance, to do the training, I had to create a Policy that allowed the notebook's execution role to iam:GetRole the sagemaker-wizard Role? This didn't seem right.

Confirmed that you can launch a SageMaker training job from a local notebook instance. However, you'll need to put in a request to raise your instance limit type (I used `ml.p2.xlarge`). Note that this is different from the `p2.xlarge` instance type, the `ml.` matters. Also, this instance type isn't one of the options in the dropdown. You may also need to specify that it's for Sagemaker training. Basically, it took me about 6 tries of back and forth with AWS support until they increased my limit and I stopped getting the error:
```
ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the CreateTrainingJob operation: The account-level service limit 'ml.p2.xlarge for training job usage' is 0 Instances, with current utilization of 0 Instances and a request delta of 1 Instances. Please contact AWS support to request an increase for this limit.
```

Another issue with training is that the training job will need to be able to access your S3 buckets. Ostensibly you can tag the bucket/object with `SageMaker` and a value of `true`. I think that works, but it got wonky when I tried to upload the results to `Validation`. I ended up creating a new Policy that gave pretty full S3 rights and added it to the Role created via the sagemaker wizard. It might have worked if I'd pre-created `validation/test.rec` and tagged the empty file with `SageMaker=true`.



Another error you may encounter is "An error occurred during deployment. Model download failed". I was able to fix this by turning it off and back on and trying again.


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
16) [Composing Images with Python for Synthetic Datasets](https://www.immersivelimit.com/tutorials/composing-images-with-python-for-synthetic-datasets)
17) [Instructions on creating a .rec file for Image Detection](https://mxnet.incubator.apache.org/versions/master/api/python/image/image.html)
18) [Reading/Writing RecordIO files](https://mxnet.incubator.apache.org/tutorials/basic/data.html)
19) [Preparing Data](https://mxnet.incubator.apache.org/faq/finetune.html)
20) [Saving and Loading Gluon Models](https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html)
21) [SageMaker Object Detection with RecordIO](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/object_detection_pascalvoc_coco/object_detection_recordio_format.ipynb)
22) [Import your custom SageMaker model to DeepLens](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-import-from-sagemaker.html)
23) [Create a custom project from your model](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-create-custom-project.html)
