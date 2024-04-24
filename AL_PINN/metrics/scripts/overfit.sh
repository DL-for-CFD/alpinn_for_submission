# overfit
for i in {0..99}
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/train_overfit.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/data/cylinder_$i.npy
done
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/train_overfit.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/data/cylinder_1.npy

for i in {1..5}
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/train_overfit.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/letters_dataset/complex$i.npy
done


# for k in 00 24 44 
for k in 00
do
for i in 06 07 08 09 10 11 12
do
for j in {0..4}
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/train_overfit.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/airfoil_dataset/airfoil_$k$i\_$j.npy
done
done
done



# 00 24 44
# 06 07 08
for k in 24 00
do
for i in 06 07 08 09 10 11 12
do
for j in {0..4}
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/train_overfit.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/problem_update/airfoil_added_mask/airfoil_$k$i\_$j.npy
done
done
done





# get inference result
for k in 44
do
for i in 06 07 08 09 10 11 12
do
for j in {0..4}
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/airfoil_dataset/airfoil_$k$i\_$j.npy --model_path='/csproject/t3_lzengaf/lzengaf/fyp/68.state' --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/baseline_airfoil
done
done
done


python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/airfoil_dataset/airfoil_0012_1.npy --model_path='/csproject/t3_lzengaf/lzengaf/fyp/520.state' --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/debug_bl


for j in {0..50}
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/data/cylinder_$j.npy --model_path=/csproject/t3_lzengaf/lzengaf/fyp/68.state --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/baseline_cylinder
done





# overfit
# for k in 00 24 44 
for k in 44
do
for i in 06 07 08 09 10 11 12
do
for j in {0..4}
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/airfoil_dataset/airfoil_$k$i\_$j.npy --model_path=/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/Logger/net/netS_airfoil_$k$i\_$j.pth --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/overfit_airfoil
done
done
done



for j in {80..99}
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/data/cylinder_$j.npy --model_path=/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/Logger/net/netS_cylinder_$j.pth --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/overfit_cylinder
done






# training log
for k in 00 24 44 
do
for i in 06 07 08 09 10 11 12
do
for j in {0..4}
do
for l in 169 179 189 199
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/airfoil_dataset/airfoil_$k$i\_$j.npy --model_path=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/dt1/netS$l.pth --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/dt1_airfoil/$l
done
done
done
done



for j in {60..99}
do
for l in 0 9 19 29 39 49 59 69 79 89 99
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/data/cylinder_$j.npy --model_path=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/dt1/netS$l.pth --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/dt1_cylinder/$l
done
done



# dt4
for k in 00 24 44 
do
for i in 06 07 08 09 10 11 12
do
for j in {0..4}
do
for l in 189 199
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/airfoil_dataset/airfoil_$k$i\_$j.npy --model_path=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/dt4/netS$l.pth --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/dt4_airfoil/$l
done
done
done
done



for j in {90..99}
do
for l in 139 149 159 169 179 189 199
do
python /csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/inference.py --qst=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/data/cylinder_$j.npy --model_path=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/dt4/netS$l.pth --inf_res_dir=/csproject/t3_lzengaf/lzengaf/fyp/Validation_Dataset/dt4_cylinder/$l
done
done


for i in {2..5}
do
mv complex$i\_v.npy complex$i.npy
done