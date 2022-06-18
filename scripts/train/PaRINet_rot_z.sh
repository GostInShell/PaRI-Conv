CUDA_VISIBLE_DEVICES=3 python train.py --dataset "ModelNetNormal" \
--batch_size 2 --batch_size_test 64 \
--dir_name log/PaRINet_rot_z --test 1 \
--network "RIDNet" --sample_points 1024 \
--nepoch 300 --lrate 0.001 --k 20 \
--rot_z \
--dp 0.8 \
--use_sgd --use_annl
