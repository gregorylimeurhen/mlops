import dataclasses, datetime, importlib, json, pathlib, random, shutil, sys, tomllib
import torch

PAD_TOKEN = "<pad>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"

PROJECT_FILES = ["config.toml", "preprocess.py", "requirements.txt", "test.py", "train.py"]


@dataclasses.dataclass
class Tokenizer:
	vocab: list
	stoi: dict
	pad_id: int
	sep_id: int
	eos_id: int

	def encode_text(self, text):
		return [self.stoi[char] for char in text]

	def decode_text(self, token_ids):
		return "".join(self.vocab[token_id] for token_id in token_ids if token_id > self.eos_id)

	def to_dict(self):
		return {
			"vocab": self.vocab,
			"pad_id": self.pad_id,
			"sep_id": self.sep_id,
			"eos_id": self.eos_id,
		}

	@staticmethod
	def from_dict(data):
		vocab = list(data["vocab"])
		return Tokenizer(vocab, {token: index for index, token in enumerate(vocab)}, int(data["pad_id"]), int(data["sep_id"]), int(data["eos_id"]))


def normalize(text):
	return text.strip().lower()


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(root, section):
	return tomllib.loads((pathlib.Path(root) / "config.toml").read_text())[section]


def ensure_run_dir(root, prefix, leaf):
	run_dir = pathlib.Path(root) / "runs" / (prefix + datetime.datetime.now().strftime("%M%S"))
	(run_dir / leaf).mkdir(parents=True, exist_ok=True)
	return run_dir


def latest_run_dir(root, prefix):
	run_dirs = [path for path in (pathlib.Path(root) / "runs").iterdir() if path.is_dir() and path.name.startswith(prefix)]
	if not run_dirs:
		raise FileNotFoundError(f"No {prefix} runs found")
	return max(run_dirs, key=lambda path: path.stat().st_mtime)


def copy_tree(source_root, run_dir):
	source_root = pathlib.Path(source_root)
	run_dir = pathlib.Path(run_dir)
	shutil.copytree(source_root / "data", run_dir / "data", dirs_exist_ok=True)
	shutil.copytree(source_root / "src", run_dir / "src", dirs_exist_ok=True)


def copy_data_files(root, run_dir, names):
	root = pathlib.Path(root)
	run_dir = pathlib.Path(run_dir)
	for name in names:
		shutil.copy2(root / "data" / name, run_dir / "data" / name)


def copy_project_files(root, run_dir):
	root = pathlib.Path(root)
	run_dir = pathlib.Path(run_dir)
	for name in PROJECT_FILES:
		shutil.copy2(root / name, run_dir / name)


def load_package(root, name):
	root = str(pathlib.Path(root))
	for module_name in list(sys.modules):
		if module_name == name or module_name.startswith(name + "."):
			del sys.modules[module_name]
	sys.path[:] = [path for path in sys.path if path != root]
	sys.path.insert(0, root)
	return importlib.import_module(name)


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_neighbors(root):
	return json.loads((pathlib.Path(root) / "data" / "neighbors.json").read_text())


def load_tsv(path):
	return [tuple(line.split("\t")) for line in pathlib.Path(path).read_text().splitlines() if line]


def load_edges(root):
	return [(normalize(room), address.strip()) for room, address in load_tsv(pathlib.Path(root) / "data" / "edges.tsv")]


def load_pairs(path):
	return [{"input": normalize(left), "gold": normalize(right)} for left, right in load_tsv(path)]


def load_training_rows(root):
	return load_pairs(pathlib.Path(root) / "data" / "train.tsv")


def load_test_rows(root):
	return load_pairs(pathlib.Path(root) / "data" / "test.tsv")


def load_room_lookup(root):
	return {normalize(room): address.strip() for room, address in load_tsv(pathlib.Path(root) / "data" / "n2a.tsv")}


def load_rooms(root):
	return sorted(load_room_lookup(root))


def build_room_trie(rooms, tokenizer):
	root = {}
	for room in sorted(rooms):
		node = root
		for token_id in tokenizer.encode_text(room):
			node = node.setdefault(token_id, {})
		node[tokenizer.eos_id] = {}
	return root


def build_tokenizer(root):
	chars = set()
	for room in load_rooms(root):
		chars.update(room)
	for key, values in load_neighbors(root).items():
		chars.update(key)
		for value in values:
			chars.update(value)
	vocab = [PAD_TOKEN, SEP_TOKEN, EOS_TOKEN] + sorted(chars)
	return Tokenizer(vocab, {token: index for index, token in enumerate(vocab)}, 0, 1, 2)


def rows_block_size(rows):
	return max(len(row["input"]) + len(row["gold"]) + 1 for row in rows)


def encode(prompt_ids, output_text, tokenizer):
	target = tokenizer.encode_text(normalize(output_text))
	tokens = list(prompt_ids) + [tokenizer.sep_id] + target + [tokenizer.eos_id]
	labels = list(tokens[1:])
	if prompt_ids:
		labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
	return {"input_ids": tokens[:-1], "labels": labels}


def encode_pair(input_text, output_text, tokenizer):
	return encode(tokenizer.encode_text(normalize(input_text)), output_text, tokenizer)


def collate_examples(examples, tokenizer, device):
	sequence_len = max(len(example["input_ids"]) for example in examples)
	input_ids = torch.full((len(examples), sequence_len), tokenizer.pad_id, dtype=torch.long, device=device)
	labels = torch.full((len(examples), sequence_len), -100, dtype=torch.long, device=device)
	for row_index, example in enumerate(examples):
		length = len(example["input_ids"])
		input_ids[row_index, :length] = torch.tensor(example["input_ids"], dtype=torch.long, device=device)
		labels[row_index, :length] = torch.tensor(example["labels"], dtype=torch.long, device=device)
	return input_ids, labels
