re-implementation

CUDA_VISIBLE_DEVICES=0,1 python3 train_baseline.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/     # SUV - 0,1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_baseline.py --batch_size 64 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ # Finish

CUDA_VISIBLE_DEVICES=4 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ 
CUDA_VISIBLE_DEVICES=0 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ --learning_rate 0.05 # NAN
CUDA_VISIBLE_DEVICES=0 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ --alpha 0
CUDA_VISIBLE_DEVICES=1 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ --alpha 0.0005
CUDA_VISIBLE_DEVICES=3 python3 train_CA.py --batch_size 16 --max_epoches 15 --root_dir ../VOCtrainval_11-May-2012/ --alpha 1.0

CUDA_VISIBLE_DEVICES=2,3,6,7 python3 train_CA.py --batch_size 48 --max_epoches 50 --root_dir ../VOCtrainval_11-May-2012/

CUDA_VISIBLE_DEVICES=0 python3 make_cam_with_crf.py --model_name VOC2012@arch=resnet38@pretrained=True@lr=0.1@wd=0.0005@bs=16@epoch=15 --root_dir ../VOCtrainval_11-May-2012/
CUDA_VISIBLE_DEVICES=1 python3 make_cam_with_crf.py --model_name VOC2012@arch=resnet38@pretrained=True@lr=0.1@wd=0.0005@bs=64@epoch=15 --root_dir ../VOCtrainval_11-May-2012/
CUDA_VISIBLE_DEVICES=2 python3 make_cam_with_crf.py --model_name VOC2012@arch=resnet38@pretrained=True@alpha=0.1@num_combination=2@lr=0.01@wd=5e-05@bs=16@epoch=15 --root_dir ../VOCtrainval_11-May-2012/

CUDA_VISIBLE_DEVICES=0 python3 make_cam_with_crf.py --model_name VOC2012@arch=resnet38@pretrained=True@alpha=0.0@num_combination=2@lr=0.01@wd=5e-05@bs=16@epoch=15 --root_dir ../VOCtrainval_11-May-2012/
CUDA_VISIBLE_DEVICES=1 python3 make_cam_with_crf.py --model_name VOC2012@arch=resnet38@pretrained=True@alpha=0.1@num_combination=2@lr=0.01@wd=5e-05@bs=48@epoch=50 --root_dir ../VOCtrainval_11-May-2012/
CUDA_VISIBLE_DEVICES=2 python3 make_cam_with_crf.py --model_name VOC2012@arch=resnet38@pretrained=True@alpha=0.0005@num_combination=2@lr=0.01@wd=5e-05@bs=16@epoch=15 --root_dir ../VOCtrainval_11-May-2012/

python show.py --model_name VOC2012@arch=resnet38@pretrained=True@lr=0.1@wd=0.0005@bs=16@epoch=15   # 40.91%, 48.65%
python show.py --model_name VOC2012@arch=resnet38@pretrained=True@lr=0.1@wd=0.0005@bs=64@epoch=15   # 39.66%, 47.63%
python show.py --model_name VOC2012@arch=resnet38@pretrained=True@alpha=0.1@num_combination=2@lr=0.01@wd=5e-05@bs=16@epoch=15 # 43.33%, 49.05%

python3 show.py --model_name VOC2012@arch=resnet38@pretrained=True@alpha=0.0005@num_combination=2@lr=0.01@wd=5e-05@bs=16@epoch=15 --root_dir ../VOCtrainval_11-May-2012/  # 42.56% 46.57%
python3 show.py --model_name VOC2012@arch=resnet38@pretrained=True@alpha=0.1@num_combination=2@lr=0.01@wd=5e-05@bs=48@epoch=50 --root_dir ../VOCtrainval_11-May-2012/     # 43.79% 46.84%
python3 show.py --model_name VOC2012@arch=resnet38@pretrained=True@alpha=0.0@num_combination=2@lr=0.01@wd=5e-05@bs=16@epoch=15 --root_dir ../VOCtrainval_11-May-2012/     # 42.62% 46.48%