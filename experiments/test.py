import csv
import json
import pathlib
import utils


def main():
	dir_ = pathlib.Path(__file__).resolve().parent
	cfg = utils.load_config(dir_, "test")
	dev = utils.device_for()
	print(dev.type)
	run = cfg["run"]
	train_run_dir = dir_ / "runs" / f"{run}" / "train"
	run_dir = train_run_dir.parent / "test"
	results_dir = run_dir / "results"
	results_dir.mkdir(parents=True, exist_ok=True)
	utils.write_snapshot(run_dir / "snapshot.zip", dir_)
	answers_path = results_dir / "answers.csv"
	fieldnames = ["input", "gold_room", "gold", *utils.BASELINE_NAMES]
	snap_path = run_dir / "snapshot.zip"
	model_path = train_run_dir / "model.pt"
	with answers_path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
		writer.writeheader()
		with utils.loaded_snapshot(snap_path) as (root2, ev):
			model, tok, rooms = ev.load_checkpoint(model_path, dev)
			rows = ev.load_rows(root2, "test")
			rm = ev.load_room_lookup(root2)
			seed = ev.load_seed(root2)
			wr = writer.writerow
			scores = ev.evaluate_rows_into(model, rows, tok, dev, rm, rooms, wr, seed)
	(results_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
	print(json.dumps(scores))


if __name__ == "__main__":
	main()
