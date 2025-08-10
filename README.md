#  GenAI – NLP & ML Edition

## **Learn Fundamental NLP Techniques with Hands-On Demos**

This branch is your **interactive sandbox** for mastering NLP the smart way—through clear examples, guided notebooks, and intuitive building blocks.

---

###  What You’ll Find Inside

| Notebook / Script                   | What You’ll Learn                            |
|------------------------------------|-----------------------------------------------|
| `tokenizationUsingNLTK.ipynb`      | Breaking text into meaningful tokens          |
| `textPreprocesssingStemming.ipynb` | Text normalization with stemming              |
| `lemmatization.ipynb`              | Smarter text processing using lemmatization   |
| `stopWords.ipynb`                  | Learn how to filter out common filler words   |
| `oneHotEncoding.ipynb`             | Intro to vectorizing text with OHE            |
| `bagOfWords.ipynb`                 | Build frequency-based word vectors (BOW)      |
| `TF=IDF.ipynb`                     | Weighted text features with TF-IDF            |
| `word2vecImplementation.ipynb`     | Using Word2Vec embeddings for meaning         |
| `namedEntityRecognition.ipynb`     | Extract names, places, and more from text     |
| `PartsOfSpeech.ipynb`              | Tag tokens with their lexical roles           |
| `first.py`                         | Utility script for quick explorations         |
| `spam.csv`                         | A small dataset for spam detection demos      |

---

###  Why This Project Rocks

- Learn **core NLP pipelines** from scratch — tokenization, normalization, feature extraction, and more.
- Explore vectorization techniques: **OHE → BOW → TF-IDF → Word2Vec**, all in one place.
- Understand **linguistic insights**—POS tagging and NER help uncover how models process meaning.
- Easy to run—no prior setup hassle.

---

###  Get Started Fast

```bash
git clone https://github.com/yashraj8703/GenAI.git
cd GenAI
git checkout NLP-ML

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
# Open any `.ipynb` and dive in!
