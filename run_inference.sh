export PYTHONPATH=`pwd`:`pwd`/vsegmenter
export export CUDA_VISIBLE_DEVICES=''
python vsegmenter/inference/run_inf.py $*