# Detecção de Fraudes em Transações com Cartão de Crédito

Desenvolvi modelos de Machine Learning em Python (Regressão Logística, Random Forest e SVM) para detectar transações fraudulentas em um dataset de **284.807 operações** altamente desbalanceado (apenas 0,17% de fraudes). Comparei os três algoritmos e escolhi o **Random Forest** como o de melhor desempenho, alcançando **F1-score de 0.85** na classe fraude (aumento de mais de 670% em relação à Regressão Logística), com precision 0.96 e recall 0.76.

### Tecnologias Utilizadas
- **Python** + **scikit-learn**
- **Modelos**: Logistic Regression, Random Forest, SVM
- **Pré-processamento**: RobustScaler, tratamento de desbalanceamento (`class_weight='balanced'`)
- **API**: Flask
- **Interface**: Streamlit
- **Deploy**: Docker + GCP (Cloud Run / App Engine)
- **Ferramentas**: pandas, joblib, matplotlib, seaborn

### Dataset
- [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284.807 transações
- 31 features (V1–V28 + Time + Amount + Class)
- Altamente desbalanceado (492 fraudes)

### Resultados dos Modelos

| Modelo                | Precision (Fraude) | Recall (Fraude) | F1-Score (Fraude) | AUC-ROC |
|-----------------------|--------------------|-----------------|-------------------|---------|
| Regressão Logística   | 0.06               | 0.92            | 0.11              | 0.9720  |
| **Random Forest**     | **0.96**           | **0.76**        | **0.85**          | 0.9572  |
| SVM                   | 0.00               | 0.37            | 0.00              | 0.5342  |

**Melhor modelo escolhido**: Random Forest
### Demonstração:

[Teste_fraude.webm](https://github.com/user-attachments/assets/f78a0d84-28c2-4099-ba16-fc63d996c211)

<img width="581" height="693" alt="Screenshot from 2026-04-01 16-31-40" src="https://github.com/user-attachments/assets/b58c9ae9-a062-4ec9-93da-e81a0f25d1a9" />


### Como Rodar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/MacViniDss/Personal-projects/Projetos_praticos/fraud-detection.git
cd fraud-detection
```
```bash
# 2. Rode a API
python app.py
```
```bash
# 3. Rode em outro terminal a interface Streamlit 
streamlit run streamlit.py
```
