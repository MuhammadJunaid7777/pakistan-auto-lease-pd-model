Pakistan Auto Lease — Probability of Default (PD) Model

An end-to-end credit risk project built by a practicing credit officer teaching himself data science — from variable selection and exploratory analysis, through model training and evaluation, to a working Streamlit app that scores individual auto-lease borrowers in real time.

⚠️ Built on synthetic data. This is a proof-of-concept demonstrating methodology end to end — it is not trained on real portfolio data and is not validated for production use. See Limitations below.

Why this project

Credit decisions for individual vehicle-lease applications are often made manually — reading eCIB reports, calculating Debt Burden Ratio (DBR) by hand, approving or rejecting on judgment alone. That process is slow, inconsistent between officers, and produces no auditable score behind the decision.

This project builds an AI-powered PD model that uses the same variables a credit officer already reviews, turns them into a consistent probability score, and wraps that score in a simple app any officer can use — no coding required on their end.

What's in this repo
File	Description
Pakistan_Auto_Lease_PD_Dataset.xlsx	300-row synthetic dataset, 15 credit variables, data dictionary, and summary stats
01_data_exploration.py	Loads the data and produces exploratory charts (default rate by borrower type, eCIB category, DBR distribution)
02_Data_Preparation.py	Cleans the data, encodes categorical variables, splits train/test
03_model_building.py	Trains Logistic Regression and XGBoost, prints accuracy/AUC/classification report, plots feature importance
04_app.py	Streamlit app — interactive PD scoring tool
Full_Project_Documentation.pdf	Complete technical write-up: variable selection rationale, methodology, results, limitations
Simple_Explainer_Document.pdf	Plain-language companion doc for non-technical audiences
Highlights
15 credit variables, chosen to mirror real Pakistani auto-lease underwriting — borrower profile, income & obligations, credit history (eCIB), and lease structure. Full rationale for every variable (and every variable deliberately excluded) is in the documentation.
Guided eCIB grading — a decision tree turns a raw eCIB report into a consistent A–E grade, removing officer-to-officer inconsistency. Grade E (no record) is deliberately placed between C and D, since "unknown risk" is not the same as "proven default."
Logistic Regression model — chosen over XGBoost for transparency: every prediction can be explained, which matters for credit decisions.
AUC: 0.86
Accuracy: 75%
Recall on defaulters: 80% (rebalanced with class_weight='balanced' — an earlier, accuracy-only version only caught 20% of actual defaulters)
Feature importance confirms eCIB category and DBR% as the two strongest predictors — statistically validating what experienced credit officers already know from the field.
Interactive Streamlit app — enter a borrower's details, get an instant PD %, risk grade (A–D), DBR check, and an approve / senior-review / reject recommendation with the specific risk flags behind it.
Getting started
Requirements
Python 3.11+
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, streamlit, openpyxl
Install
bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit openpyxl
Run the exploration and model scripts
bash
python 01_data_exploration.py
python 02_Data_Preparation.py
python 03_model_building.py
Run the app
bash
streamlit run 04_app.py

Then open the local URL Streamlit prints (usually http://localhost:8501).

Methodology snapshot
Define variables — 15 fields a credit officer already reviews: income, eCIB grade, DBR, guarantor, product type, and more.
Explore the data — validate that default patterns match real-world credit intuition before trusting it with a model.
Prepare & encode — clean data, encode categoricals, scale features, 80/20 train-test split.
Train & compare — Logistic Regression vs. XGBoost; chosen on recall for defaulters, not just raw accuracy.
Ship the app — turn the trained model into something a non-technical officer can actually use.

Full detail, including every variable's rationale and what was deliberately left out, is in Full_Project_Documentation.pdf.

Key finding

61.3% of borrowers with a prior confirmed default (eCIB Grade D) default again — the single strongest signal in the dataset, and the foundation of the entire model.

Limitations & next steps
Trained entirely on synthetic data — every number here will shift once retrained on real, anonymized company cases.
Age currently shows a simple linear relationship with risk; real data should reveal a U-shaped curve (younger and older borrowers typically carry more risk than mid-career borrowers).
Not yet validated by risk management, compliance, or any regulatory review — required before any production use.
Planned next step: collect real historical lease outcomes using the same 15-variable structure, retrain, and run the model in parallel with a rules-based scorecard during a pilot.
About

Built by a credit officer in leasing & vehicle finance, Pakistan, as a self-directed project to combine credit domain expertise with applied AI/Python skills.

📄 Also see: Simple_Explainer_Document.pdf for a plain-language walkthrough suitable for non-technical audiences.

#CreditRisk #AI #Pakistan #Fintech #MachineLearning
