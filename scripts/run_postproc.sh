export PYTHONPATH=`echo $(dirname "$PWD")`:`echo $(dirname "$PWD")`/vsegmenter
python ../vsegmenter/postproc/run_post.py
