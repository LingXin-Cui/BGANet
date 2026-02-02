dataset=BUSI
input_size=256
python train.py --arch BGANet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python test.py --name ${dataset}_UKAN

dataset=GlaS
input_size=512
python train.py --arch BGANet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python test.py --name ${dataset}_UKAN

dataset=ISIC2017
input_size=256
python train.py --arch BGANet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python test.py --name ${dataset}_UKAN

dataset=CVC
input_size=256
python train.py --arch BGANet --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
python test.py --name ${dataset}_UKAN







