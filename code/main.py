import hashlib
import shutil
import time
import torch
import wandb
import yaml

from dotenv import load_dotenv
from pathlib import Path
from src import *


def compact_items(data):
	return "(" + ", ".join(f"{key}={data[key]}" for key in data) + ")"


def run_id():
	return hashlib.sha1(str(time.time()).encode()).hexdigest()[:6]


def ensure_run_dir(root, identifier):
	run_dir = Path(root) / "runs" / identifier
	(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
	return run_dir


def snapshot_run_files(root, run_dir):
	for name in ["config.yaml", "data.md", "src.py"]:
		shutil.copy2(root / name, run_dir / name)


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
	start = time.time()
	load_dotenv()
	root = Path(__file__).resolve().parent
	config_path = root / "config.yaml"
	data_path = root / "data.md"
	config = yaml.safe_load(config_path.read_text())
	original_config = dict(config)
	device = device_for()
	set_seed(int(config["seed"]))
	datasets = materialize_data_file(data_path)
	rows = datasets["pretrain"] + datasets["finetune"] + datasets["val"] + datasets["test"]
	tokenizer = build_tokenizer(rows)

	if config["architecture"] != "GPT2":
		raise ValueError(f"Unsupported architecture {config['architecture']}")

	model = build_model(tokenizer, rows).to(device)
	batch_size = infer_batch_size(model, rows, tokenizer, device)
	max_tokens = infer_max_tokens(rows, tokenizer)
	config = config | {"batch_size": batch_size, "max_tokens": max_tokens}
	identifier = run_id()
	run_dir = ensure_run_dir(root, identifier)
	snapshot_run_files(root, run_dir)

	with wandb.init(dir=str(run_dir), name=identifier, project="mlops") as run:
		pretrain_train, pretrain_val = split_rows(datasets["pretrain"], int(config["seed"]))
		epoch = 0
		epoch, _ = train_stage(model, "pretrain", pretrain_train, pretrain_val, tokenizer, device, run_dir, config, batch_size, epoch, run)
		epoch, _ = train_stage(model, "finetune", datasets["finetune"], datasets["val"], tokenizer, device, run_dir, config, batch_size, epoch, run)
		metric = instantiate_metric("NLS")
		val_scores, val_details = evaluate_rows(model, datasets["val"], tokenizer, metric, device, max_tokens)
		test_scores, test_details = evaluate_rows(model, datasets["test"], tokenizer, metric, device, max_tokens)
		scores = {f"val_{key}": value for key, value in val_scores.items()} | {f"test_{key}": value for key, value in test_scores.items()}
		run.log({
			"evaluation": wandb.Table(
				columns=["input", "gold", "output"],
				data=[[row["input"], row["gold"], row["output"]] for row in val_details + test_details],
			)
		})
		run.log({
			"summary": wandb.Table(
				columns=["id", "config", "scores", "duration"],
					data=[[
						identifier,
						compact_items(original_config),
						compact_items(scores),
						time.time() - start,
					]],
			)
		})


if __name__ == "__main__":
	main()
