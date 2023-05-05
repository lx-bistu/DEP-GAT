# DEP-GAT
Dialogue Relationship Extraction with Dependency Relation

## Setup
1. Download GloVe vectors from [here](https://www.kaggle.com/thanakomsn/glove6b300dtxt/data) and put it into `dataset/` folder, 
Then download BERT-base-uncased model from [here](https://huggingface.co/bert-base-uncased) and put it into `huggingface/bert-base-uncased/` folder.

2. Install from requirements.txt by `pip install -r requirements.txt` and run `python -m spacy download en_core_web_sm`

3. Insatll dgl library according to your cuda version using the command like below.
```sh
pip install dgl-cu110==0.6.1     # For CUDA 11.0 Build
```
4. download `stopwords` using the commands like below.
```sh
python
>>> import nltk
>>> nltk.download('stopwords')
```

## Run code

### Training
Execute the following command to train the DEP-GAN model.
```sh
# DEP-GAT-BERT model
python DEP-GAT-BERT.py

# DEP-GAT-LSTM model
python DEP-GAT-LSTM.py
```

### Testing
```sh
# DEP-GAT-BERT model
python DEP-GAT-BERT.py --mode test --ckpt_path [your_ckpt_file_path]

# DEP-GAT-LSTM model
python DEP-GAT-LSTM.py --mode test --ckpt_path [your_ckpt_file_path]
```


