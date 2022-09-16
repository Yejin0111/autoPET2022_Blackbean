
zip -r weights_s5d2w32_crop192_minspc.zip 3d_fullres
sudo bash build_s5d2w32_crop192_minspc.sh
sudo docker save autopet_s5d2w32_crop192_minspc | gzip -c > autopet_s5d2w32_crop192_minspc.tar.gz
