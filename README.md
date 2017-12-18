# Super-Resolution
Final project for CIS 680. Deep learning for single image super resolution

## Introduction
This project is a tensorflow implementation of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf) and [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf), combined with ensemble method proposed in [Learning a mixture of Deep Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1701.00823.pdf).

The result is obtained following the same settings from v5 edition of [SRGAN](https://arxiv.org/pdf/1609.04802.pdf) and [EDSR](https://arxiv.org/pdf/1707.02921.pdf) winning the [NTIRE2017](http://www.vision.ee.ethz.ch/ntire17/) challenge. However, due to limited resources, the networks are trained on the [RAISE dataset](http://mmlab.science.unitn.it/RAISE/) which contains 8156 high resolution images. Tests on Set5, Set14 with different algorithms are shown below. The code is highly inspired by [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow).

## Sample Results

<table>
    <tr>
        <td><center>LR</center></td>
        <td><center>SRResNet</center></td>
        <td><center>SRGAN</center></td>
        <td><center>EDSR</center></td>
        <td><center>ensemble</center></td>
        <td><center>HR</center></td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/comic.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/comic_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/comic_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/comic_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/comic_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/comic.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/baboon.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/baboon_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/baboon_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/baboon_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/baboon_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/baboon.png" height="200"></center>
        </td>
    </tr>
</table>

## Dependency

* Python 2.7 or Python 3.5
* Tensorflow (tested on r1.4)
* Download the VGG19 weights from the [TF-slim models](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)
* The code is tested on 
    * Ubuntu 16.04 LTS with CPU architecture x86_64 + NVIDIA GeForce GTX 1060, and 1080 Ti

## Getting Started

* Train the model

To train the models, follow the steps below.

1. Clone the repository

```bash
# clone the repository from github
git clone https://github.com/owenqyzhang/Super-Resolution
cd Super-Resolution
```

2. Data and checkpiont preparation

```bash
# make the directory to put the vgg19 pre-trained model
mkdir vgg19/
cd vgg19/
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
tar xvf ./vgg19_2016_08_28.tar.gz
```

3. Train the model

run the training script
```bash
sh ./train_SRResNet.sh
```

## More results on public benchmark dataset

* Benchmark results compared to baseline algorithms

<table>
    <tr>
        <td><center>PSNR</center></td>
        <td><center>nearset</center></td>
        <td><center>bicubic</center></td>
        <td><center>SRCNN</center></td>
        <td><center>SelfExSR</center></td>
        <td><center>DRCN</center></td>
        <td><center>ESPCN</center></td>
        <td><center>SRResNet</center></td>
        <td><center>SRGAN</center></td>
        <td><center>EDSR</center></td>
        <td><center>ensemble</center></td>
    </tr>
    <tr>
        <td><center>Set5</center></td>
        <td><center>26.26</center></td>
        <td><center>28.43</center></td>
        <td><center>30.07</center></td>
        <td><center>30.33</center></td>
        <td><center>31.52</center></td>
        <td><center>30.76</center></td>
        <td><center>34.55</center></td>
        <td><center>28.90</center></td>
        <td><center>32.12</center></td>
        <td><center>31.14</center></td>
    </tr>
    <tr>
        <td><center>Set14</center></td>
        <td><center>24.64</center></td>
        <td><center>25.99</center></td>
        <td><center>27.18</center></td>
        <td><center>27.45</center></td>
        <td><center>28.02</center></td>
        <td><center>27.66</center></td>
        <td><center>27.99</center></td>
        <td><center>25.52</center></td>
        <td><center>28.54</center></td>
        <td><center>27.52</center></td>
    </tr>
</table>

* Set5

<table>
    <tr>
        <td><center>LR</center></td>
        <td><center>SRResNet</center></td>
        <td><center>SRGAN</center></td>
        <td><center>EDSR</center></td>
        <td><center>ensemble</center></td>
        <td><center>HR</center></td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/baby.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/baby_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/baby_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/baby_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/baby_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/baby.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/bird.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/bird_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/bird_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/bird_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/bird_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/bird.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/butterfly.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/butterfly_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/butterfly_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/butterfly_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/butterfly_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/butterfly.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/head.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/head_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/head_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/head_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/head_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/head.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/woman.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/woman_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/woman_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/woman_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/woman_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/woman.png" height="200"></center>
        </td>
    </tr>
</table>

* Set14

<table>
    <tr>
        <td><center>LR</center></td>
        <td><center>SRResNet</center></td>
        <td><center>SRGAN</center></td>
        <td><center>EDSR</center></td>
        <td><center>ensemble</center></td>
        <td><center>HR</center></td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/baboon.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/baboon_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/baboon_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/baboon_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/baboon_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/baboon.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/barbara.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/barbara_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/barbara_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/barbara_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/barbara_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/barbara.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/bridge.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/bridge_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/bridge_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/bridge_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/bridge_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/bridge.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/coastguard.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/coastguard_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/coastguard_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/coastguard_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/coastguard_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/coastguard.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/comic.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/comic_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/comic_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/comic_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/comic_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/comic.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/face.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/face_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/face_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/face_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/face_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/face.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/flowers.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/flowers_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/flowers_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/flowers_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/flowers_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/flowers.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/foreman.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/foreman_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/foreman_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/foreman_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/foreman_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/foreman.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/lenna.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/lenna_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/lenna_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/lenna_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/lenna_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/lenna.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/man.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/man_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/man_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/man_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/man_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/man.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/monarch.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/monarch_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/monarch_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/monarch_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/monarch_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/monarch.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/pepper.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/pepper_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/pepper_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/pepper_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/pepper_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/pepper.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/ppt3.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/ppt3_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/ppt3_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/ppt3_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/ppt3_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/ppt3.png" height="200"></center>
        </td>
    </tr>
    <tr>
        <td>
            <center><img src="./result/LR/zebra.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRResNet/images/zebra_SRResNet-MSE.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/SRGAN/images/zebra_SRGAN-VGG54.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/EDSR/images/zebra_EDSR.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/ensemble/images/zebra_ensemble.png" height="200"></center>
        </td>
         <td>
            <center><img src="./result/HR/zebra.png" height="200"></center>
        </td>
    </tr>
</table>