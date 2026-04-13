import csv, json, pathlib
import src.utils


def checkpoint_name(value):
	name = str(value).strip()
	return name if name.endswith(".pt") else f"{name.zfill(4)}.pt"


def write_answers(path, rows):
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=["input", "gold_room", "gold", "identity", "levenshtein", "ours"], quoting=csv.QUOTE_ALL)
		writer.writeheader()
		writer.writerows(rows)


def write_scores(path, scores):
	path.write_text(json.dumps(scores, indent=2) + "\n")


def main():
	root = pathlib.Path(__file__).resolve().parent
	config = src.utils.load_config(root, "test")
	device = src.utils.device_for()
	run_dir = src.utils.ensure_run_dir(root, "E", "results")
	train_run_dir = src.utils.latest_run_dir(root, "T")
	train_src = src.utils.load_package(train_run_dir, "src")
	src.utils.copy_tree(train_run_dir, run_dir)
	src.utils.copy_data_files(root, run_dir, ["n2a.tsv", "test.tsv"])
	src.utils.copy_project_files(root, run_dir)
	checkpoint = train_run_dir / "checkpoints" / checkpoint_name(config["checkpoint"])
	model, tokenizer, rooms = train_src.models.load_checkpoint(checkpoint, device)
	rows = train_src.utils.load_test_rows(run_dir)
	room_lookup = train_src.utils.load_room_lookup(run_dir)
	scores, details = train_src.metrics.evaluate_rows(model, rows, tokenizer, device, room_lookup, rooms)
	write_answers(run_dir / "results" / "answers.csv", details)
	write_scores(run_dir / "results" / "scores.json", scores)
	print(json.dumps(scores))


if __name__ == "__main__":
	main()
