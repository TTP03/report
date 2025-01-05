
cd Machine-Translation

### install dependencies
pip install -r requirements.txt


### Preprocess data
python preprocess.py

python split_dataset.py


### Create vocabulary
python vocab/moses_vocab.py

### Training
python train.py

