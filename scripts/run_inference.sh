#export export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=`echo $(dirname "$PWD")`:`echo $(dirname "$PWD")`/vsegmenter
python ../vsegmenter/inference/run_inf.py $*
