import datetime, importlib, pathlib, shutil, sys, tomllib
import torch

PROJECT_FILES = ["config.toml", "preprocess.py", "requirements.txt", "test.py", "train.py"]


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
