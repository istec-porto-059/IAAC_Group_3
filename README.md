# Cyber ML Classifier

Projeto simples e reutilizável para classificação de datasets de cibersegurança.

## Datasets preparados

O mesmo programa foi desenhado para reconhecer estes quatro casos:

1. **IoT Intrusion Detection** — target esperado: `label`
2. **Enron Fraud Email** — target esperado: `Label`/`label`; `Subject` e `Body` são tratados como texto com TF-IDF
3. **Malware Executable Detection** — target esperado: `label`; features `F_1...F_531`
4. **Keylogger Detection** — target esperado: `Class`

Também existe um modo genérico para outros CSVs de classificação.

## Arquitetura

```text
cyber_ml_classifier/
├── data/                 # colocar os datasets aqui
├── models/               # modelos treinados (.joblib)
├── results/              # métricas e matrizes de confusão
├── notebooks/
│   └── template.ipynb    # notebook que reutiliza o mesmo código
├── src/
│   ├── cli.py            # perguntas/argumentos do programa
│   ├── data.py           # leitura e limpeza
│   ├── profiles.py       # reconhece os datasets/target
│   ├── features.py       # numérico, categórico e texto
│   ├── models.py         # classificadores
│   ├── evaluation.py     # métricas
│   ├── reporting.py      # guarda resultados/modelo
│   ├── training.py       # coordena o treino
│   └── prediction.py     # previsão em novos dados
├── main.py
├── predict.py
└── requirements.txt
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Utilização simples

Coloca um CSV dentro de `data/` e executa:

```bash
python main.py
```

O programa pede apenas:

```text
1. Caminho do dataset
2. Coluna a classificar (tenta detetar automaticamente)
3. Nome do projeto/modelo
```

Exemplo IoT:

```text
1. data/IoT_Intrusion.csv
2. label
3. iot_intrusion
```

Também pode ser executado sem perguntas:

```bash
python main.py --dataset data/IoT_Intrusion.csv --target label --name iot_intrusion
```

Para testar rapidamente um dataset muito grande:

```bash
python main.py --dataset data/IoT_Intrusion.csv --target label --name iot_test --quick
```

`--quick` usa no máximo 50 000 linhas. Para os resultados finais do projeto, executa sem `--quick`.

## O que acontece automaticamente

```text
Dataset
   ↓
Detetar tipo de dados
   ├── numéricos → mediana + StandardScaler
   ├── categorias → OneHotEncoder
   └── texto → TF-IDF
   ↓
Train/Test 80/20
   ↓
Treinar 3 classificadores
   ↓
Accuracy + Precision + Recall + F1
   ↓
Escolher melhor por Macro F1
   ↓
Guardar modelo + relatórios
```

Para datasets tabulares são testados por defeito:

- Decision Tree
- Random Forest
- Logistic Regression

Para datasets com texto (por exemplo Enron):

- Logistic Regression
- Linear SVM
- SGD Classifier

Isto evita usar Random Forest sobre centenas de milhares de emails transformados em milhares de palavras, o que seria demasiado pesado. Se o professor exigir modelos específicos, podes passá-los:

```bash
python main.py --dataset data/exemplo.csv --target label --name teste \
  --models decision_tree random_forest logistic_regression
```

## Ficheiros gerados

Depois do treino:

```text
models/iot_intrusion.joblib

results/iot_intrusion/
├── metrics.json
├── confusion_matrix.csv
└── classification_report.csv
```

O `metrics.json` mostra os resultados de todos os modelos e qual ganhou.

## Fazer previsões

```bash
python predict.py \
  --model models/iot_intrusion.joblib \
  --dataset data/novos_dados.csv \
  --output results/previsoes.csv
```

O ficheiro final ganha uma coluna `prediction`.

## Notas sobre os 4 datasets

### IoT Intrusion Detection

O Kaggle indica 1 191 264 registos, 47 colunas e target `label`, com classes como DDoS, DoS, Mirai, Recon, Spoofing, Brute Force, Web attacks e BenignTraffic. O programa usa diretamente as classes existentes no CSV.

### Enron Fraud Email

É um problema binário fraud/não-fraud e altamente desequilibrado. O programa reconhece colunas de texto como `Subject`/`Body` e usa TF-IDF. A escolha do melhor modelo é feita por **Macro F1**, não apenas Accuracy.

### Malware Executable Detection

O dataset tem 531 features (`F_1` a `F_531`) e uma coluna `label` que indica malware ou não-malware.

### Keylogger Detection

O target conhecido é `Class`, com classes benign/keylogger. O restante conjunto de features é processado automaticamente consoante o tipo das colunas.

## Testes

```bash
pytest -q
```

Os testes criam versões pequenas e sintéticas dos quatro formatos para verificar que a mesma pipeline consegue treinar e prever sem depender dos ficheiros gigantes do Kaggle.
