cd predict
echo 'util'
python util.py
echo 'predict'
python predicting.py
cd ../metric
echo 'metric'
python mrc_eval.py predicts.json ref.json v1
