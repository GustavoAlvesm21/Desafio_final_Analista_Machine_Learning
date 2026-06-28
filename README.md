# Desafio Final - Machine Learning

Este repositório contém a implementação do **Desafio Final do módulo de Machine Learning** da **Pós-Graduação em Data Science com Inteligência Artificial**.

O objetivo do projeto é comparar diferentes técnicas de Machine Learning e Deep Learning aplicadas ao problema de **previsão de inadimplência**, utilizando o conjunto de dados **German Credit**. Durante o desenvolvimento foram implementados modelos clássicos, redes neurais, métodos de ensemble, técnicas de otimização de hiperparâmetros, validação cruzada e rastreamento de experimentos com **MLflow**.

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
