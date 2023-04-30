export PYTHONPATH=`echo $(dirname "$PWD")`:`echo $(dirname "$PWD")`/vsegmenter
python ../vsegmenter/training/run_train.py $*