conda activate DFNet_scallapy
export PYTHONPATH="../DeepFilterNet/:$PYTHONPATH"
python -W ignore ../tools/metrics.py --model_dir '/Users/leopoldh/Documents/Drawio - Memoire/My Model/DF4 gated/small 0.5/metrcis/gates/ft/model' --dataset_dir '/Users/leopoldh/DOCUMENTS_LOCAUX/ECOLE/Memoire/dataset/hdf5/all' --boxplot --valentini
