export CUDA_VISIBLE_DEVICES=5

conda activate metabolomicscuda
module load cuda/11.8

python -u  /users/jdvillegas/repos/TeFT_FORK_PTFI/TeFT_train.py \
  --type_model ionmode \
  --ion_mode pos \
  --loss_fcn_type CrossEntropy \
  --device cpu \
  --path_to_teft_folder /users/jdvillegas/repos

python -u  /users/jdvillegas/repos/TeFT_FORK_PTFI/TeFT_train.py \
  --type_model ionmode \
  --ion_mode neg \
  --loss_fcn_type CrossEntropy \
  --device cpu \
  --path_to_teft_folder /users/jdvillegas/repos