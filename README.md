# 🔍 Legal Document Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An AI-powered tool designed to analyze legal contracts, break them into clauses, identify risky clauses, and explain them in plain English using LLMs.

---

## 📌 Overview

The **Legal Document Analyzer** helps lawyers, paralegals, and non-lawyers quickly identify potentially risky clauses in contracts. It uses **Natural Language Processing (NLP)** techniques to break down documents, analyze clauses, and explain risky sections in clear, simple language.

---

## 📂 Project Structure

```
LegalDocumentAnalyzer/
├── train_model/
│   ├── analyze_document.py          # Script to analyze documents and explain risky clauses
│   ├── dataset_loader.py            # Script to load CUAD dataset
│   ├── train_classifier.py           # Script to train Sentence-BERT models
│   ├── train_roberta_classifier.py   # Script to train RoBERTa models
│   ├── compare_models.py            # Script to compare CUAD models
├── data/
│   ├── legal_clauses_labeled.csv    # Processed CUAD dataset
├── saved_model/                     # Folder containing trained BERT models
├── saved_model_roberta/             # Folder containing trained RoBERTA models
├── .env                             # Environment file for storing sensitive keys
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
```

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Mohini-vashisth/DocumentAnalyser.git
cd LegalDocumentAnalyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your `.env` file

Create a `.env` file in the root directory and add your API key:

```
GROQ_API_KEY=your_groq_api_key
```

### 4. Prepare Datasets

Ensure your processed datasets (`legal_clauses_labeled.csv`) are in the `data/` folder.

---

## 📖 Usage Guide

### 🔨 Training Models

#### Sentence-BERT
```bash
python train_model/train_classifier.py
```

#### RoBERTa
```bash
python train_model/train_roberta_classifier.py
```

### 📈 Comparing Models
```bash
python train_model/compare_models.py
```
This script evaluates both models on the CUAD dataset and visualizes metrics like precision, recall, and F1-score.

### 📄 Analyzing a Contract (PDF or Text File)
```bash
python train_model/analyze_document.py
```
Ensure you have a `Sample.pdf` or `Sample.txt` file in the root directory.

---

## 📊 Explanation of Models

- **Sentence-BERT (MiniLM):** Converts legal clauses to embeddings.
- **Logistic Regression:** Classifies clauses as `risky` or `safe`. Caution: those labels come from a 7-keyword rule (liability, indemnify, termination, penalty, breach, damages, loss), so the reported accuracy measures how well the model re-learns that rule, not legal risk.
- **Groq + `openai/gpt-oss-120b`:** Explains risky clauses in simple language. Opt-in, and the model can be changed with the `GROQ_MODEL` environment variable.

---

## 🧩 Also in the repo

- **Web API** (`api/`, Django): upload or paste a contract, get clauses scored most risky first. Analyses are saved only if the caller asks (opt-in), with a 30-day expiry. See `api/DATA_POLICY.md`; the scheduled purge task on the home PC is disabled until launch, so re-enable it before going live.
- **IP-assignment check:** a keyword check for IP-ownership clauses with plain reasons and wording to send the client. Not reviewed by a lawyer.
- **Plain-English explanations:** opt-in, only the top risky clauses are sent to Groq.
- **Paid full report:** built but off until a self-hosted model is configured.
- **Clause-type detector evaluation** (`train_model/build_clause_type_dataset.py`, `train_model/evaluate_clause_types*.py`): CUAD's real clause types, a contract-level train/test split, and a comparison of keywords, embeddings, an LLM, and embeddings flagging with an LLM confirming. Best macro F1 so far is 0.487, below the target of 0.50, and not shown to beat the LLM alone. Not used by the live app yet.

This tool can be wrong and is not legal advice.

---

## 📅 Future Improvements

- Add a Flask-based web interface for better user experience.
- Implement a report generator for PDF/HTML outputs.
- Visualize results with charts and graphs.
- Extend support to DOCX files.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💬 Contact

Maintained by **Lakshya**. Feel free to reach out if you have questions or suggestions!
