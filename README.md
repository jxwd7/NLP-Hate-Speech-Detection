# AI for Content Moderation: A Comparative Analysis of Hate Speech Detection Models

This project is a comprehensive academic analysis of various Natural Language Processing (NLP) techniques for automated hate speech detection. The primary goal was to evaluate and compare the performance of traditional machine learning, deep learning, and large language models (LLMs) to identify the most effective approaches for real-world business applications like content moderation and brand safety.

## Key Activities & Methodology

### 1. Data Analysis & Preprocessing
To thoroughly test the models, two distinct and challenging datasets were used:
*   **Toxic Comment Classification Dataset (Kaggle):** A multi-label classification task to identify comments across six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
*   **HateXplain Dataset (Hugging Face):** A more nuanced dataset used for a binary classification task (`hate speech` vs. `not hate speech`) to test the LLMs.

Both datasets underwent extensive text cleaning (e.g., stop-word removal, lemmatization) and exploratory data analysis (EDA).

### 2. Model Implementation
A wide spectrum of models were trained and evaluated to provide a full comparative analysis:
*   **Traditional ML:** Logistic Regression & Random Forest (using TF-IDF features).
*   **Deep Learning:** A Long Short-Term Memory (LSTM) network with pre-trained GloVe embeddings.
*   **Large Language Models (LLMs):** Experiments using zero-shot prompting on GPT-2, GPT-3, and a fine-tuned BERT model for sentiment analysis.

### 3. Interactive Demo
A simple but effective web application was built using **Gradio** to allow for real-time classification of new text using the trained Logistic Regression and Random Forest models.

## Technical Stack
*   **Core Libraries:** Python, Pandas, Scikit-learn, NLTK, TensorFlow (Keras), Transformers (Hugging Face)
*   **Models & Architectures:** Logistic Regression, Random Forest, LSTM, BERT, GPT-3, GPT-2
*   **Tools & Concepts:** Jupyter Notebooks, OpenAI API, Gradio, Plotly, Exploratory Data Analysis (EDA), Text Classification, TF-IDF, Word Embeddings (GloVe)

## How to Run
1.  The primary analysis is contained in the Jupyter Notebooks (`.ipynb` files).
2.  To run the Gradio app, execute the final cells in the `NLP_Assm_2_Part_1.ipynb` notebook.
3.  The full academic write-up can be found in `Assignment-2.pdf`.

## Key Insights
This project was a fantastic learning experience in the trade-offs of applied machine learning. The results showed that while LLMs are powerful, classic models provide a surprisingly strong baseline. The most crucial takeaway was that high accuracy doesn't tell the whole story, and the "best" model depends entirely on the business goal—whether it's maximizing user safety (prioritizing recall) or protecting a brand's image (prioritizing precision).
