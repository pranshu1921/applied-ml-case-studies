from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from case_studies._shared.io import get_paths, safe_write_json
from case_studies.churn.src.make_dataset import make_synthetic_churn_dataset
from case_studies.churn.src.preprocess import preprocess_and_split
from case_studies.churn.src.train import train_model
from case_studies.churn.src.evaluate import evaluate_model
from case_studies.churn.src.report import build_report


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run churn case study end-to-end.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--smoke", action="store_true", help="Fast run for CI smoke test.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.smoke:
        # smaller dataset for CI
        config["data"]["n_rows"] = 800
        config["data"]["random_seed"] = 7

    case_name = config.get("case_name", "churn")
    paths = get_paths(case_name)

    # 1) Data generation (raw)
    raw_path = paths.data_raw / "churn_raw.csv"
    df_raw = make_synthetic_churn_dataset(
        n_rows=int(config["data"]["n_rows"]),
        random_seed=int(config["data"]["random_seed"]),
    )
    df_raw.to_csv(raw_path, index=False)

    # 2) Preprocess + split (processed)
    train_path = paths.data_processed / "churn_train.csv"
    test_path = paths.data_processed / "churn_test.csv"
    X_train, X_test, y_train, y_test = preprocess_and_split(
        df_raw=df_raw,
        target=config["features"]["target"],
        test_size=float(config["data"]["test_size"]),
        random_seed=int(config["data"]["random_seed"]),
        train_path=train_path,
        test_path=test_path,
    )

    # 3) Train
    model_path = paths.artifacts_case / "model.joblib"
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        categorical=config["features"]["categorical"],
        numeric=config["features"]["numeric"],
        model_type=config["model"]["type"],
        model_params=config["model"]["params"],
        out_path=model_path,
    )

    # 4) Evaluate + plots + business threshold
    metrics_path = paths.artifacts_case / "metrics.json"
    eval_out = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        plots_dir=paths.plots,
        business=config["business"],
    )
    safe_write_json(metrics_path, eval_out)

    # 5) Report (markdown)
    report_path = paths.reports / "churn_report.md"
    report_md = build_report(eval_out=eval_out, repo_root=paths.repo)
    report_path.write_text(report_md, encoding="utf-8")

    print("✅ Churn pipeline complete")
    print(f"- Raw data:        {raw_path}")
    print(f"- Processed data:  {train_path} , {test_path}")
    print(f"- Model artifact:  {model_path}")
    print(f"- Metrics:         {metrics_path}")
    print(f"- Plots:           {paths.plots}")
    print(f"- Report:          {report_path}")


if __name__ == "__main__":
    main()
