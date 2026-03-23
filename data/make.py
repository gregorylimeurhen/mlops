import csv, re, sys
from collections import defaultdict, deque
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent
DATASET_FILENAMES = {
	"pretraining": {
		"name": "names.txt",
		"address": "addresses.txt",
	},
	"finetuning": {
		"name-to-addresses": "n2A.txt",
		"address-to-names": "a2N.txt",
	},
}
TOKEN_CHARS = set(".-0123456789")

def normalize_text(text):
	return re.compile(r"\s+").sub(
		" ",
		re.compile(r"\s*,\s*").sub(", ", re.compile(r"\s*([<>])\s*").sub(r"\1", text)),
	).strip()

def csv_rows(path, delimiter=","):
	with path.open(newline="") as file:
		return [
			tuple(row)
			for row in csv.reader(file, delimiter=delimiter, skipinitialspace=delimiter == ",")
			if row
		]

def load_substitutions(input_root):
	rules = []
	digit_words = {"0": "zero"}
	symbol_words = {}
	for filename in ("nns.csv", "ns.csv"):
		for source, target in csv_rows(input_root / filename):
			rules.append(
				(
					source,
					target,
					re.compile(re.escape(source))
					if source in {".", "-", "–"}
					else re.compile(rf"(?<![0-9A-Za-z]){re.escape(source)}(?![0-9A-Za-z])"),
				)
			)
			if len(source) == 1 and source.isdigit():
				digit_words[source] = normalize_text(target)
			if source not in {".", "-", "–"} or not any(character.isalpha() for character in target):
				continue
			if (target := normalize_text(target)) in symbol_words.get(source, []):
				continue
			symbol_words.setdefault(source, []).append(target)
	return rules, digit_words, symbol_words

def numeric_match_is_inside_address(text, start, end):
	if start == end:
		return False
	while start > 0 and text[start - 1] in TOKEN_CHARS:
		start -= 1
	while end < len(text) and text[end] in TOKEN_CHARS:
		end += 1
	return "." in text[start:end] or "-" in text[start:end]

def replace_span(text, start, end, target):
	if target == target.strip() and any(character.isalnum() for character in target):
		if start > 0 and text[start - 1].isalnum():
			target = f" {target}"
		if end < len(text) and text[end].isalnum():
			target = f"{target} "
	return normalize_text(f"{text[:start]}{target}{text[end:]}")

def address_token_variants(token, digit_words, symbol_words):
	variants = [""]
	for character in token:
		variants = [
			normalize_text(f"{prefix} {option}")
			for prefix in variants
			for option in (
				[digit_words.get(character, character)]
				if character.isdigit()
				else [character.lower()]
				if character.isalpha()
				else symbol_words.get(character, [character])
			)
		]
	return [variant for variant in dict.fromkeys(variants) if variant and variant != token]

def line_variants(line, rules, digit_words, symbol_words):
	original = normalize_text(line)
	seen = {original}
	queue = deque([original])
	variants = [original]
	while queue:
		current = queue.popleft()
		for source, target, pattern in rules:
			for match in pattern.finditer(current):
				if source.isdigit() and numeric_match_is_inside_address(current, match.start(), match.end()):
					continue
				if (candidate := replace_span(current, match.start(), match.end(), target)) in seen:
					continue
				seen.add(candidate)
				variants.append(candidate)
				queue.append(candidate)
		for match in re.compile(r"[0-9][0-9A-Z.-]*").finditer(current):
			if "." not in (token := match.group(0)) and "-" not in token:
				continue
			for candidate in address_token_variants(token, digit_words, symbol_words):
				if (candidate := normalize_text(f"{current[:match.start()]}{candidate}{current[match.end():]}")) in seen:
					continue
				seen.add(candidate)
				variants.append(candidate)
				queue.append(candidate)
	return variants

def write_lines(path, lines):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines) + "\n")

def dataset_paths(output_root):
	return {
		phase: {name: output_root / filename for name, filename in filenames.items()}
		for phase, filenames in DATASET_FILENAMES.items()
	}

def write_base_data(edges_path, output_root):
	name_to_addresses = defaultdict(set)
	address_to_names = defaultdict(set)
	for name, address in csv_rows(edges_path, delimiter="\t"):
		name_to_addresses[name].add(address)
		address_to_names[address].add(name)
	paths = dataset_paths(output_root)
	write_lines(paths["pretraining"]["name"], sorted(name_to_addresses))
	write_lines(paths["pretraining"]["address"], sorted(address_to_names))
	write_lines(
		paths["finetuning"]["name-to-addresses"],
		[f"{name}<{', '.join(sorted(name_to_addresses[name]))}>" for name in sorted(name_to_addresses)],
	)
	write_lines(
		paths["finetuning"]["address-to-names"],
		[f"{address}<{', '.join(sorted(address_to_names[address]))}>" for address in sorted(address_to_names)],
	)
	return paths

def augment_lines(lines, rules, digit_words, symbol_words):
	output_lines = []
	seen = set()
	for line in lines:
		for variant in line_variants(line, rules, digit_words, symbol_words):
			if variant in seen:
				continue
			seen.add(variant)
			output_lines.append(variant)
	return output_lines

def split_pair(line):
	prompt, separator, target = line.partition("<")
	if not separator or not target.endswith(">"):
		raise ValueError(f"invalid pair line: {line}")
	return normalize_text(prompt), normalize_text(target[:-1])

def augment_pair_lines(lines, rules, digit_words, symbol_words):
	output_lines = []
	seen = set()
	for line in lines:
		prompt, target = split_pair(line)
		for variant in line_variants(prompt, rules, digit_words, symbol_words):
			if (candidate := f"{variant}<{target}>") in seen:
				continue
			seen.add(candidate)
			output_lines.append(candidate)
	return output_lines

def augment_file(path, rules, digit_words, symbol_words):
	lines = path.read_text().splitlines()
	output_lines = (
		augment_pair_lines(lines, rules, digit_words, symbol_words)
		if path.name in DATASET_FILENAMES["finetuning"].values()
		else augment_lines(lines, rules, digit_words, symbol_words)
	)
	write_lines(path, output_lines)
	return len(lines), len(output_lines)

def parse_augment(argv):
	if not argv:
		return True
	if argv == ["--no-augment"]:
		return False
	raise SystemExit("usage: python data/make.py [--no-augment]")

def main():
	augment = parse_augment(sys.argv[1:])
	paths = write_base_data(DATA_ROOT / "edges.tsv", DATA_ROOT)
	for phase in paths.values():
		for path in phase.values():
			print(path)
	if not augment:
		return
	rules, digit_words, symbol_words = load_substitutions(DATA_ROOT)
	for phase in paths.values():
		for path in phase.values():
			input_count, output_count = augment_file(path, rules, digit_words, symbol_words)
			print(f"{path.name}: {input_count} -> {output_count}")

if __name__ == "__main__":
	main()
