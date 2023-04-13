export PYTHONPATH=`echo $(dirname "$PWD")`:`echo $(dirname "$PWD")`/vsegmenter
python ../vsegmenter/model/train.p $*