"""Offline, lossless input packaging. No LLM, EDA, or feature preprocessing."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import METRICS


ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num",
                 "marital-status", "occupation", "relationship", "race", "sex",
                 "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
GROUPS = {"DSBench Modeling": "dsbench", "DataSciBench": "datascibench"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def access_check(path, task, valid_labels=None):
    """Read only headers/target; exercise the exact current stratified split recipe."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    if len(columns) != len(set(columns)) or any(not c.strip() for c in columns):
        raise ValueError("Empty or duplicate column names")
    if task["target"] not in columns or len(columns) < 2:
        raise ValueError("A target and at least one input column are required")
    if task["metric"] not in METRICS[task["problem_type"]]:
        raise ValueError("Unsupported task metric")
    y = pd.read_csv(path, usecols=[task["target"]])[task["target"]]
    errors = []
    missing = int(y.isna().sum())
    if missing:
        errors.append(f"目标列存在 {missing} 个缺失标签，未删除或填补")
    if valid_labels is not None:
        invalid = int((y.notna() & ~y.isin(valid_labels)).sum())
        if invalid:
            errors.append(f"目标列存在 {invalid} 个不属于 {valid_labels} 的标签，未改写")
    info = {"rows": len(y), "columns": len(columns), "features": len(columns) - 1,
            "target": task["target"], "missing_target": missing}
    if not errors:
        if task["problem_type"] == "regression":
            if not np.isfinite(pd.to_numeric(y, errors="raise")).all():
                errors.append("回归目标含非有限值")
        elif y.nunique() < 2:
            errors.append("分类目标少于两个类别")
    if not errors:
        indices = np.arange(len(y))
        stratify = y if task["problem_type"] == "classification" else None
        remaining, test = train_test_split(indices, test_size=task["split"]["test"],
                                           random_state=task["seed"], stratify=stratify)
        train, validation = train_test_split(
            remaining, test_size=task["split"]["validation"] / (1 - task["split"]["test"]),
            random_state=task["seed"], stratify=y.iloc[remaining] if stratify is not None else None)
        if stratify is not None and any(set(y.iloc[i]) != set(y) for i in [train, validation, test]):
            errors.append("当前划分无法使每个集合都包含全部类别")
        info["split_rows"] = {"train": len(train), "validation": len(validation), "test": len(test)}
        info["split_disjoint"] = not any(set(a) & set(b) for a, b in
                                           [(train, validation), (train, test), (validation, test)])
    info["errors"] = errors
    info["status"] = "blocked" if errors else "ready"
    return info


def package_one(row, source_root, destination):
    source = (Path(source_root) / row["data_path"]).resolve(strict=True)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "data.csv"
    changes = []
    if row["benchmark"] == "DataSciBench" and row["task"] == "human_12":
        # Add only a known schema. Preserve the entire original byte stream afterwards.
        with output.open("wb") as target, source.open("rb") as original:
            target.write((",".join(ADULT_COLUMNS) + "\n").encode())
            shutil.copyfileobj(original, target)
        changes.append("无表头 adult.data 增加固定 15 列表头；原始数据字节保持不变")
    else:
        shutil.copyfile(source, output)
        changes.append("原始 CSV 按字节复制；不改列名、行、值或顺序")
    notes = ["Only input packaging has been performed. All original feature columns, IDs, "
             "constant columns and missing markers are retained. Analyze them in EDA and perform "
             "appropriate feature handling in FE fitted on training data only.",
             "This is an independent random-split experiment, not reproduction of the original benchmark.",
             f"Original metric: {row['original_metric']}; experiment metric: {row['metric']}."]
    special = {
        "playground-series-s3e17": "TWF/HDF/PWF/OSF/RNF describe failure subtypes and can leak Machine failure; exclude these in FE before modeling.",
        "spaceship-titanic": "PassengerId includes travel groups; this random split does not measure unseen-group generalization.",
        "tabular-playground-series-aug-2022": "Random splitting does not measure unseen-product generalization.",
        "csv_excel_32": "Zero in Glucose/BloodPressure/SkinThickness/Insulin/BMI may represent missing measurements; assess in EDA and handle in FE.",
        "csv_excel_47": "All source species are retained. Unlike the earlier four-species screening task, this experiment predicts every species present in the source file.",
        "human_12": "Original field whitespace and '?' markers remain; inspect in EDA and handle feature markers in FE. Target labels retain their original whitespace.",
    }
    if row["task"] in special:
        notes.append(special[row["task"]])
    task = {"name": f"{GROUPS[row['benchmark']]}_{row['task']}", "description": " ".join(notes),
            "data_path": "data.csv", "target": row["target"], "problem_type": row["problem_type"],
            "metric": row["metric"], "seed": 42, "split": {"validation": .2, "test": .2}}
    try:
        check = access_check(output, task, valid_labels=["MALE", "FEMALE"] if
                             row["benchmark"] == "DataSciBench" and row["task"] == "human_18" else None)
    except (ValueError, TypeError, KeyError, pd.errors.ParserError) as error:
        check = {"status": "blocked", "errors": [str(error)]}
    source_hash, output_hash = sha256(source), sha256(output)
    if row["task"] != "human_12" and source_hash != output_hash:
        raise ValueError("Source and copied data differ")
    provenance = {"source": str(source), "source_sha256": source_hash, "data_sha256": output_hash,
                  "benchmark": row["benchmark"], "task": row["task"],
                  "screening_status": row["status"], "original_metric": row["original_metric"],
                  "changes": changes, "row_filtering": False, "feature_preprocessing": False,
                  "access_check": check}
    (destination / "source.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    # Invalid labels are retained, but do not publish a runnable task for them.
    task_name = "task.json" if check["status"] == "ready" else "task.blocked.json"
    (destination / task_name).write_text(json.dumps(task, ensure_ascii=False, indent=2))
    return {"benchmark": row["benchmark"], "task": row["task"], "status": check["status"],
            "data_path": f"{destination.name}/data.csv", "task_path": f"{destination.name}/{task_name}",
            "rows": check.get("rows"), "features": check.get("features"),
            "metric": task["metric"], "resource_group": row["status"],
            "reason": "; ".join(check["errors"])}


def build_catalog(candidates, source_root, output):
    with Path(candidates).open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] in {"候选", "候选-较大或高维"}]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in rows:
        name = f"{GROUPS[row['benchmark']]}__{row['task']}"
        record = package_one(row, source_root, output / name)
        records.append(record)
        # Incremental index keeps completed work discoverable even if copying later files fails.
        (output / "catalog.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"[{len(records)}/{len(rows)}] {name}: {record['status']}", flush=True)
    with (output / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    ready = sum(r["status"] == "ready" for r in records)
    lines = ["# 表格任务统一入口", "", f"检查 {len(records)} 项候选；可接入 {ready} 项，暂缓 {len(records)-ready} 项。",
             "", "只做文件接入，不调用 LLM，不做 EDA/FE，不删除 ID、常量列、缺失行或类别。",
             "已有 CSV 按字节复制；Adult 只补表头。每项 source.json 保存来源、哈希及接入检查。",
             "ready 表示目标有效且当前划分可行，不表示 EDA/FE/MT 已跑通或满足 120 秒预算。",
             "Fish 保留全部物种，不再限定四类。指标沿用候选表中的项目指标，并非原基准复现。",
             "", "运行示例（从项目根目录）：", "", "```bash",
             f"python launch_data_agent.py --task {output}/dsbench__titanic/task.json \\",
             "  --env-file /home/cmcui/InternAgent/.env", "```", "",
             "| 来源 / 任务 | 状态 | 行数 | 特征数 | 配置 | 原因 |",
             "|---|---|---:|---:|---|---|"]
    for r in records:
        lines.append(f"| {r['benchmark']} / {r['task']} | {r['status']} | {r['rows']} | {r['features']} | "
                     f"[配置]({r['task_path']}) | {r['reason']} |")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return records
