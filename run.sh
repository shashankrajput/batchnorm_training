# CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100

CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=1.00
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=0.50
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=0.25
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=0.10
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=0.03
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=0.01
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=0.005
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=0.001
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.1 --epochs=100 --train_only_bn=True --sparsity=0.0001