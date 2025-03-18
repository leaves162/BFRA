#! /bin/bash
#CUDA_VISIBLE_DEVICES=1 python extra_adaptor_test.py --epochs 50 --support_num 3 --supp_query_num 2 --learning_rate 1.0
#CUDA_VISIBLE_DEVICES=1 python extra_adaptor_test.py --epochs 50 --support_num 12 --supp_query_num 8 --learning_rate 1.0
CUDA_VISIBLE_DEVICES=3 python extra_adaptor_test.py --epochs 50 --support_num 30 --supp_query_num 20 --learning_rate 1.0




