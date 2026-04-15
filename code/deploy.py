import argparse
import dotenv
import export
import json
import os
import pathlib
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


API = "https://api.netlify.com/api/v1"
DEF_SITE = "sutdoko"
DEF_TEAM = "gregorylimeurhen"


def load_token(root):
	dotenv.load_dotenv(root / ".env")
	tok = os.getenv("NETLIFY_PERSONAL_ACCESS_TOKEN")
	if tok:
		return tok
	raise RuntimeError("missing NETLIFY_PERSONAL_ACCESS_TOKEN in code/.env")


def err_text(err):
	data = err.read()
	if not data:
		return f"{err.code} {err.reason}"
	try:
		obj = json.loads(data)
	except json.JSONDecodeError:
		text = data.decode("utf-8", "replace").strip()
		return text or f"{err.code} {err.reason}"
	if not isinstance(obj, dict):
		return json.dumps(obj)
	msg = obj.get("message")
	if msg:
		return msg
	code = obj.get("code")
	if code is not None:
		return f"{code}"
	return json.dumps(obj)


def req(method, url, tok, body=None, ctype=None):
	headers = {"Authorization": f"Bearer {tok}"}
	if ctype:
		headers["Content-Type"] = ctype
	r = urllib.request.Request(url, body, headers, method=method)
	try:
		with urllib.request.urlopen(r, timeout=60) as res:
			data = res.read()
			if not data:
				return None
			return json.loads(data)
	except urllib.error.HTTPError as err:
		text = err_text(err)
		msg = f"{method} {url} failed: {text}"
		raise RuntimeError(msg) from None


def latest_model(root):
	pat = root / "runs"
	paths = sorted(pat.glob("*/train/model.pt"), key=lambda p: p.stat().st_mtime)
	if paths:
		return paths[-1]
	raise RuntimeError("no code/runs/*/train/model.pt found")


def site_url(team):
	return f"{API}/{team}/sites"


def get_site(team, site, tok):
	url = site_url(team)
	sites = req("GET", url, tok)
	for row in sites:
		if row.get("name") == site:
			return row
	return None


def make_site(team, site, tok):
	url = f"{API}/{team}/sites"
	body = json.dumps({"name": site}).encode()
	return req("POST", url, tok, body, "application/json")


def ensure_site(team, site, tok):
	row = get_site(team, site, tok)
	if row:
		return row
	return make_site(team, site, tok)


def build_dir(root, model):
	app = root / "app"
	tmp = tempfile.TemporaryDirectory()
	out = pathlib.Path(tmp.name)
	shutil.copytree(app, out, dirs_exist_ok=True)
	export.export_model(model, out)
	return tmp, out


def zip_path(src, dst):
	with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
		for path in sorted(src.rglob("*")):
			if path.is_dir():
				continue
			name = path.relative_to(src)
			zf.write(path, name)
	return dst


def deploy_url(site_id, title):
	query = {"production": "true", "title": title}
	qs = urllib.parse.urlencode(query)
	return f"{API}/sites/{site_id}/deploys?{qs}"


def create_deploy(site_id, title, zip_file, tok):
	url = deploy_url(site_id, title)
	body = zip_file.read_bytes()
	return req("POST", url, tok, body, "application/zip")


def get_deploy(site_id, dep_id, tok):
	url = f"{API}/sites/{site_id}/deploys/{dep_id}"
	return req("GET", url, tok)


def wait_ready(site_id, dep_id, tok):
	while True:
		row = get_deploy(site_id, dep_id, tok)
		state = row.get("state", "")
		if state == "ready":
			return row
		if state in {"error", "failed"}:
			msg = row.get("error_message") or "deploy failed"
			raise RuntimeError(msg)
		time.sleep(1)


def parse_args():
	p = argparse.ArgumentParser()
	root = pathlib.Path(__file__).resolve().parent
	model = latest_model(root)
	p.add_argument("model", nargs="?", default=str(model))
	p.add_argument("--site", default=DEF_SITE)
	p.add_argument("--team", default=DEF_TEAM)
	return p.parse_args()


def main():
	args = parse_args()
	root = pathlib.Path(__file__).resolve().parent
	tok = load_token(root)
	model = pathlib.Path(args.model).resolve()
	if not model.exists():
		raise FileNotFoundError(str(model))
	site = ensure_site(args.team, args.site, tok)
	site_id = site["id"]
	run = model.parent.parent.name
	title = f"run {run}"
	tmp, out = build_dir(root, model)
	with tmp:
		zip_file = pathlib.Path(tmp.name).with_suffix(".zip")
		zip_path(out, zip_file)
		dep = create_deploy(site_id, title, zip_file, tok)
		dep = wait_ready(site_id, dep["id"], tok)
	url = dep.get("ssl_url") or dep.get("url") or site.get("ssl_url")
	print(json.dumps({
		"deploy_id": dep["id"],
		"model": str(model),
		"model_bytes": model.stat().st_size,
		"site_id": site_id,
		"site_name": site["name"],
		"url": url,
	}, indent=2))


if __name__ == "__main__":
	main()
