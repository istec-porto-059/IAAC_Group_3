from pathlib import Path
import argparse

from .data import load_dataset
from .profiles import detect_profile, resolve_target
from .training import train_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="Pipeline genérica de classificação para datasets de cibersegurança.")
    parser.add_argument("--dataset", help="Caminho para o CSV/TSV/Parquet")
    parser.add_argument("--target", help="Coluna que o modelo deve classificar")
    parser.add_argument("--name", help="Nome do projeto/modelo")
    parser.add_argument("--quick", action="store_true", help="Usa no máximo 50 mil linhas para um teste rápido")
    parser.add_argument("--models", nargs="+", help="Modelos específicos a testar")
    return parser


def _ask(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or (default or "")


def run(args):
    print("\n=== CYBER ML CLASSIFIER ===\n")

    dataset_path = args.dataset or _ask("1. Caminho do dataset")
    df = load_dataset(dataset_path)
    profile = detect_profile(list(df.columns))

    detected_target = resolve_target(list(df.columns), profile, args.target)
    if args.target and not detected_target:
        raise ValueError(f"Não encontrei a coluna target '{args.target}'.")

    print(f"Dataset: {len(df):,} linhas x {len(df.columns)} colunas")
    print(f"Perfil detetado: {profile.name}")

    target = detected_target
    if not target:
        print("\nColunas disponíveis:")
        print(", ".join(df.columns))
        target = _ask("2. Coluna que queres classificar")
    elif not args.target:
        answer = _ask("2. Coluna a classificar", target)
        target = resolve_target(list(df.columns), profile, answer)
        if not target:
            raise ValueError(f"A coluna '{answer}' não existe.")

    default_name = Path(dataset_path).stem.replace(" ", "_").lower()
    project_name = args.name or _ask("3. Nome do projeto/modelo", default_name)

    summary = train_dataset(
        df=df,
        target=target,
        profile=profile,
        project_name=project_name,
        model_names=args.models,
        quick=args.quick,
    )

    print("\n=== RESULTADOS ===")
    for model_name, metrics in summary["models"].items():
        print(f"{model_name:22s} F1 macro: {metrics['f1_macro']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

    print(f"\nMelhor modelo: {summary['best_model']}")
    print(f"F1 macro: {summary['best_f1_macro']:.4f}")
    print(f"Modelo guardado em: {summary['model_file']}")
    print(f"Relatórios em: results/{project_name}/")
