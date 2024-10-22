export CUDA_VISIBLE_DEVICES=5

conda activate metabolomicscuda
module load cuda/11.8

python -u  /users/jdvillegas/repos/TeFT_FORK_PTFI/TeFT_train.py \
  --collision_energy_level le \
  --loss_fcn_type CrossEntropy \
  --device cpu \

python -u  /users/jdvillegas/repos/TeFT_FORK_PTFI/TeFT_train.py \
  --collision_energy_level me \
  --loss_fcn_type CrossEntropy \
  --device cpu \

python -u  /users/jdvillegas/repos/TeFT_FORK_PTFI/TeFT_train.py \
  --collision_energy_level he \
  --loss_fcn_type CrossEntropy \
  --device cpu \