# BGANet: A Medical Image Segmentation Network via Boundary-guided and Anti-aliasing

📌 This is an official PyTorch implementation of **BGANet: A Medical Image Segmentation Network via Boundary-guided and Anti-aliasing**

To address the challenge of difficulty in accurate segmentation caused by soft boundaries in medical images, we specifically designed the Learnable Anti-aliasing Block, the Boundary Learner, and the Boundary Guidance Module. The entire network follows the U-KAN settings and is divided into three parts: encoder layers, decoder layers, and a bottleneck layer. During the encoding stage, we introduce the Learnable Anti-aliasing Block to perform learnable anti-aliasing processing on the encoded features before downsampling, thereby reducing boundary misalignment during segmentation and the Boundary Learner is employed to extract boundary information from both shallow and deep features. During the decoding phase, the Boundary Guidance Module utilizes this learned boundary information in a cross-scale manner to guide precise segmentation of the target regions.

<div align="center">
    <img width="100%" alt="BGANet overview" src="overview.png"/>
</div>




## Setup

```bash
git clone https://github.com/LingXin-Cui/BGANet.git
cd BGANet
conda create -n bganet python=3.10
conda activate bganet
pip install -r requirements.txt
```

**Tips A**: We test the framework using pytorch=1.13.0, and the CUDA compile version=11.6.



## Data Preparation
**BUSI**:  The dataset can be found [here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). 

**GLAS**:  The dataset can be found [here](https://websignon.warwick.ac.uk/origin/slogin?shire=https%3A%2F%2Fwarwick.ac.uk%2Fsitebuilder2%2Fshire-read&providerId=urn%3Awarwick.ac.uk%3Asitebuilder2%3Aread%3Aservice&target=https%3A%2F%2Fwarwick.ac.uk%2Ffac%2Fcross_fac%2Ftia%2Fdata%2Fglascontest&status=notloggedin).

**ISIC 2017**:  The dataset can be found [here](https://challenge.isic-archive.com/data/#2017). 





The resulted file structure is as follows.
```
data
├── ISIC2017
│   ├── train
│     ├── images
│           ├── ISIC_0000000.jpg
|           ├── ...
|     ├── masks
│           ├── ISIC_0000000_segmentation.png
|           ├── ...
│   ├── val
│     ├── images
│           ├── ISIC_0001769.jpg
|           ├── ...
|     ├── masks
│           ├── ISIC_0001769_segmentation.png
|           ├── ...
│   ├── test
│     ├── images
│           ├── ISIC_0012092.jpg
|           ├── ...
|     ├── masks
│           ├── ISIC_0012092_segmentation.png
|           ├── ...
```





## Training BGANet

You can simply train BGANet on a single GPU by specifing the dataset name ```--dataset``` and input size ```--input_size```.
```bash
python train.py --arch BGANet --dataset {dataset} --input_w {input_size} --input_h {input_size} --name {dataset}_BGANet  --data_dir [YOUR_DATA_DIR]
```
For example, train BGANet with the resolution of 256x256 with a single GPU on the BUSI dataset in the ```inputs``` dir:
```bash
python train.py --arch BGANet --dataset BUSI --input_w 256 --input_h 256 --name BUSI_BGANet  --data_dir ../data/BUSI
```
Please see scripts.sh for more details.
Note that the resolution of GlaS is 512x512, differing with other datasets (256x256).



## Testing BGANet
Run the following scripts to 
```bash
python test.py --name ${dataset}_BGANet --output_dir [YOUR_OUTPUT_DIR] 
```
