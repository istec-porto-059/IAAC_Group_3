import argparse

from src.prediction import predict_dataset


parser = argparse.ArgumentParser(description="Classificar novos dados usando um modelo já treinado.")
parser.add_argument("--model", required=True, help="Ficheiro .joblib em models/")
parser.add_argument("--dataset", required=True, help="Novo dataset com as mesmas features")
parser.add_argument("--output", default="results/predictions.csv", help="CSV de saída")
args = parser.parse_args()

path, rows = predict_dataset(args.model, args.dataset, args.output)
print(f"{rows} previsões guardadas em: {path}")
