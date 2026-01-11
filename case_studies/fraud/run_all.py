from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from case_studies._shared.io import get_paths, safe_write_json
from case_studies.fraud.src.make_dataset import make_synthetic_fraud_dataset
from case_studies.fraud.src.preprocess import preprocess_and_split
from case_studies.fraud.src.train import train_model
from case_studies.fraud.src.evaluate import evaluate_model
from case_studies.fraud.src.report import build_report


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud case study end-to-end.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--smoke", action="store_true", help="Fast run for CI smoke test.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.smoke:
        config["data"]["n_rows"] = 1500
        config["data"]["random_seed"] = 7
        config["data"]["fraud_rate"] = 0.02

    case_name = config.get("case_name", "fraud")
    paths = get_paths(case_name)

    # 1) Data generation
    raw_path = paths.data_raw / "fraud_raw.csv"
    df_raw = make_synthetic_fraud_dataset(
        n_rows=int(config["data"]["n_rows"]),
        fraud_rate=float(config["data"]["fraud_rate"]),
        random_seed=int(config["data"]["random_seed"]),
    )
    df_raw.to_csv(raw_path, index=False)

    # 2) Preprocess + split
    train_path = paths.data_processed / "fraud_train.csv"
    test_path = paths.data_processed / "fraud_test.csv"
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

    # 5) Report
    report_path = paths.reports / "fraud_report.md"
    report_md = build_report(eval_out=eval_out, repo_root=paths.repo)
    report_path.write_text(report_md, encoding="utf-8")

    print("✅ Fraud pipeline complete")
    print(f"- Raw data:        {raw_path}")
    print(f"- Processed data:  {train_path} , {test_path}")
    print(f"- Model artifact:  {model_path}")
    print(f"- Metrics:         {metrics_path}")
    print(f"- Plots:           {paths.plots}")
    print(f"- Report:          {report_path}")


if __name__ == "__main__":
    main()
