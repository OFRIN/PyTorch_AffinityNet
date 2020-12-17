re-implementation

CUDA_VISIBLE_DEVICES=0,1 python3 train_baseline.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/     # SUV - 0,1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_baseline.py --batch_size 64 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ # Finish

CUDA_VISIBLE_DEVICES=4 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ 
CUDA_VISIBLE_DEVICES=0 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ --learning_rate 0.05 # NAN
CUDA_VISIBLE_DEVICES=0 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ --alpha 0
CUDA_VISIBLE_DEVICES=1 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ --alpha 0.0005