# BiLSTM-CRF
Implementation of BiLSTM-CRF model for Sequence Tagging

[BiLSTM-CRF](https://arxiv.org/pdf/1508.01991v1.pdf) is a simple Deep Learning model for Sequence Tagging and Name Entity Recognition.

![BiLSTM-CRF Network](./resources/bilstm-crf-network.png)

### Repo structure
* README.md
* models - folder to store trained models for latter use
* logs - folder of training, validation, and testing loggings
* resources - folder to store training data
* notebooks - list of Jupyter Notebooks for tutorials
* bilstm_crf - folder to store utils, data loader, and model
* requirements.txt - list of dependencies
* train.py - main file to execute training task
* predict.py - tutorial script for makeing predictions locally

### Data & Word Embeddings
*  Data:
	* [Name-Entity-Recognition-Corpus](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
	* [WNUT 17](https://huggingface.co/datasets/wnut_17)
* Word Embeddings:
	* [Twitter or Wikipediea Word Embeddings](https://nlp.stanford.edu/projects/glove/)

### Instructions
#### Installation & Setup
```
git clone https://github.com/quocdat32461997/NER
pip install -r requirements.txt
```

#### Text processing
* Step 1:
	* For above datasets, please look at /notebooks/Train_Dsta_Processing.ipynb
	* In production: all text and labels are stored in a signle file (each for training, validation, and testing) that each line is a raw sentence and a sequence of tag labels. The raw sentence and the sequence of tag labels are separated by tab **\t**. 
	
	Input format sample: **I am Eric. \t O O PER**
* Step 2: Tensorflow Dataset pipeline embedded in **Dataset class of bilstm_crf/data.py** accepts the text input format in **step 1** and auto-processes text.

#### Training:
- Deveopment: run **SequenceTagger.ipynb** either locally or on Google Colab for development
- Training: run **train.py** to train BiLSTM-CRF on NER dataset & WNut 2017 dataset or your dcustom ataset. Example:
```
# need to modify path to data and word-embedding files
python3 train.py
```

#### Inference
The trained model is saved in the SavedFormat mode that the trained model could be loaded for:
* Fine-tuning
* Deployment in:
	* Local server
	```
	# run predict.py
	python3 predict.py
	```
	
	* Tensorflow Serving - TBD	
	```
	To be added
	```
### Dependencies
* Tensorflow >= 2.3.1
* Python >= 3.8

### References
* [Depends of the definition](https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/)
* [Annotated Corpus for Named Entity Recognition](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
* Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation
* @inproceedings{derczynski-etal-2017-results,
    title = "Results of the {WNUT}2017 Shared Task on Novel and Emerging Entity Recognition",
    author = "Derczynski, Leon  and
      Nichols, Eric  and
      van Erp, Marieke  and
      Limsopatham, Nut",
    booktitle = "Proceedings of the 3rd Workshop on Noisy User-generated Text",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W17-4418",
    doi = "10.18653/v1/W17-4418",
    pages = "140--147",
    abstract = "This shared task focuses on identifying unusual, previously-unseen entities in the context of emerging discussions.
                Named entities form the basis of many modern approaches to other tasks (like event clustering and summarization),
                but recall on them is a real problem in noisy text - even among annotators.
                This drop tends to be due to novel entities and surface forms.
                Take for example the tweet {``}so.. kktny in 30 mins?!{''} {--} even human experts find the entity {`}kktny{'}
                hard to detect and resolve. The goal of this task is to provide a definition of emerging and of rare entities,
                and based on that, also datasets for detecting these entities. The task as described in this paper evaluated the
                ability of participating entries to detect and classify novel and emerging named entities in noisy text.",
}
