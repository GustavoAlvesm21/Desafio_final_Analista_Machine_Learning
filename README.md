# Desafio Final - Machine Learning

Este repositório contém a implementação do **Desafio Final do módulo de Machine Learning** da **Pós-Graduação em Data Science com Inteligência Artificial**.

O objetivo do projeto é comparar diferentes técnicas de Machine Learning e Deep Learning aplicadas ao problema de **previsão de inadimplência**, utilizando o conjunto de dados **German Credit**. Durante o desenvolvimento foram implementados modelos clássicos, redes neurais, métodos de ensemble, técnicas de otimização de hiperparâmetros, validação cruzada e rastreamento de experimentos com **MLflow**.

---

## 📚 Conteúdo do Projeto

O projeto está organizado em tópicos, cada um abordando um conceito importante de Machine Learning e Deep Learning aplicado à previsão de inadimplência.

### Tópico 1 – Modelos de Classificação
Implementação e comparação entre **Regressão Logística**, **Árvore de Decisão** e **Support Vector Machine (SVM)**, destacando suas vantagens e desvantagens no problema de classificação de inadimplência.

### Tópico 2 – Overfitting e Underfitting
Discussão sobre os conceitos de **overfitting** e **underfitting**, além da implementação de estratégias de regularização para melhorar a capacidade de generalização dos modelos.

### Tópico 3 – Redes Neurais
Desenvolvimento de uma **Rede Neural Artificial** básica e de um modelo **Deep Learning** utilizando **PyTorch**, comparando desempenho, custo computacional e necessidade de dados.

### Tópico 4 – Métodos Ensemble
Implementação do **XGBoost** e estudo dos métodos de **Bagging (Random Forest)** e **Boosting (XGBoost)**. Também são abordados:
- Otimização de hiperparâmetros com **Grid Search** e **Random Search**;
- Validação cruzada utilizando **Stratified K-Fold**, além da explicação de **K-Fold** e **Leave-One-Out (LOOCV)**.

### Tópico 5 – Redes Recorrentes
Implementação de uma **RNN** e de uma **LSTM**, destacando como as LSTMs resolvem limitações das RNNs através do uso de células de memória para capturar dependências de longo prazo.

### Tópico 6 – MLOps
Apresentação dos conceitos de **MLOps**, incluindo rastreamento de experimentos com **MLflow**, gerenciamento de modelos e boas práticas para implantação e manutenção de modelos em produção.

---

## 🛠️ Tecnologias utilizadas

- Python 3.12
- Pandas
- Scikit-learn
- XGBoost
- PyTorch
- MLflow

---

## 🚀 Como executar o projeto

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. (Opcional) Crie um ambiente virtual

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute os códigos

Após instalar as dependências, basta executar os scripts de cada tópico.

---

## 📊 Visualizando os experimentos com MLflow

Todos os modelos registram parâmetros, métricas e modelos utilizando o **MLflow**.

Para iniciar a interface gráfica, execute:

```bash
mlflow ui
```

Depois, abra o navegador e acesse:

```
http://127.0.0.1:5000
```

Na interface será possível visualizar:

- Experimentos
- Parâmetros
- Métricas
- Modelos salvos
- Curvas de treinamento (RNN/LSTM)

---

## 👨‍💻 Autor

**Gustavo Alves**

Projeto desenvolvido como requisito de avaliação do módulo de **Machine Learning** da **Pós-Graduação em Data Science com Inteligência Artificial**.
