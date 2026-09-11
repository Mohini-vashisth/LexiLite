# 🔍 Legal Document Analyzer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
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
- **Logistic Regression:** Classifies clauses as `risky` or `safe`.
- **Groq + Mixtral:** Explains risky clauses in simple language using LLMs.

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
