rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir t2m
cd t2m 
echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1dtKP2xBk-UjG9o16MVfBJDmGNSI56Dch/view?usp=sharing
unzip humanml3d_models.zip
rm humanml3d_models.zip
cd ../../