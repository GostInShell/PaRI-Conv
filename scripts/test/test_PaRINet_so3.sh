CUDA_VISIBLE_DEVICES=1 python train.py --dataset "ModelNetNormal" \
--batch_size 32 --batch_size_test 64 \
--dir_name log/PaRINet_so3/best --test 1 --training 0 --test_final 1 \
--network "RIDNet" --sample_points 1024 \
--nepoch 20 --lrate 0.001 --k 20 \
--rand_rot \
--dp 0.8 \
--use_sgd --use_annl
