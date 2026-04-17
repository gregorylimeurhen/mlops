import build
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request


class ApiError(RuntimeError):
	def __init__(self, code, method, url, text):
		super().__init__(f"{method} {url} failed: {text}")
		self.code = code


def load_token(root):
	path = pathlib.Path(root) / ".env"
	if not path.exists():
		raise RuntimeError("missing VERCEL_ACCESS_TOKEN in experiments/.env")
	for line in path.read_text().splitlines():
		key, sep, value = line.partition("=")
		if key != "VERCEL_ACCESS_TOKEN" or not sep:
			continue
		value = value.strip().strip("\"'")
		if value:
			return value
	raise RuntimeError("missing VERCEL_ACCESS_TOKEN in experiments/.env")


def load_deploy(root):
	utils = build.load_utils()
	cfg = utils.load_config(root, "deploy")
	api = str(cfg["api"]).rstrip("/")
	project = str(cfg["project"]).strip()
	team = str(cfg["team"]).strip()
	return {"api": api, "project": project, "team": team}


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


def api_url(cfg, path, query=None):
	url = f"{cfg['api']}{path}"
	if not query:
		return url
	qs = urllib.parse.urlencode(query)
	return f"{url}?{qs}"


def scope(cfg):
	team = cfg["team"]
	if team:
		return {"slug": team}
	return {}


def req(method, url, tok, body=None, ctype=None, extra=None):
	headers = {"Authorization": f"Bearer {tok}"}
	if ctype:
		headers["Content-Type"] = ctype
	if extra:
		headers.update(extra)
	req_ = urllib.request.Request(url, body, headers, method=method)
	try:
		with urllib.request.urlopen(req_, timeout=60) as res:
			data = res.read()
			if not data:
				return None
			return json.loads(data)
	except urllib.error.HTTPError as err:
		text = err_text(err)
		raise ApiError(err.code, method, url, text) from None


def get_project(cfg, tok):
	name = urllib.parse.quote(cfg["project"])
	url = api_url(cfg, f"/v9/projects/{name}", scope(cfg))
	return req("GET", url, tok)


def make_project(cfg, tok):
	url = api_url(cfg, "/v11/projects", scope(cfg))
	body = json.dumps({"name": cfg["project"]}).encode()
	return req("POST", url, tok, body, "application/json")


def ensure_project(cfg, tok):
	name = cfg["project"]
	print(f"project {name}")
	try:
		print("check project")
		return get_project(cfg, tok)
	except ApiError as err:
		if err.code != 404:
			raise
	print("create project")
	try:
		return make_project(cfg, tok)
	except ApiError as err:
		if err.code != 409:
			raise
	print("project already exists")
	return get_project(cfg, tok)


def upload_file(cfg, tok, path):
	data = pathlib.Path(path).read_bytes()
	sha = hashlib.sha1(data).hexdigest()
	url = api_url(cfg, "/v2/files", scope(cfg))
	headers = {"x-vercel-digest": sha}
	req("POST", url, tok, data, "application/octet-stream", headers)
	return sha, len(data)


def deploy_paths(root):
	root = pathlib.Path(root)
	for path in sorted(root.rglob("*")):
		if path.is_dir():
			continue
		if any(part.startswith(".") for part in path.relative_to(root).parts):
			continue
		if "__pycache__" in path.parts:
			continue
		if path.suffix == ".py":
			continue
		if path.suffix == ".pyc":
			continue
		if path.name == ".DS_Store":
			continue
		yield path


def create_deploy(cfg, tok, root):
	files = []
	print(f"upload dir {root}")
	for path in deploy_paths(root):
		name = path.relative_to(root).as_posix()
		sha, size = upload_file(cfg, tok, path)
		print(f"upload {name} {size}")
		files.append({"file": name, "sha": sha, "size": size})
	body = {
		"files": files,
		"name": cfg["project"],
		"project": cfg["project"],
		"projectSettings": {"framework": None},
		"target": "production",
	}
	query = scope(cfg)
	query["skipAutoDetectionConfirmation"] = "1"
	url = api_url(cfg, "/v13/deployments", query)
	body = json.dumps(body).encode()
	print(f"create deploy files {len(files)}")
	return req("POST", url, tok, body, "application/json")


def get_deploy(cfg, dep_id, tok):
	path = urllib.parse.quote(dep_id)
	url = api_url(cfg, f"/v13/deployments/{path}", scope(cfg))
	return req("GET", url, tok)


def wait_ready(cfg, dep_id, tok):
	last = None
	print(f"wait deploy {dep_id}")
	while True:
		try:
			row = get_deploy(cfg, dep_id, tok)
		except urllib.error.URLError as err:
			print(f"poll retry {err.reason}")
			time.sleep(1)
			continue
		state = row.get("readyState") or row.get("status") or ""
		if state != last:
			print(f"state {state}")
			last = state
		if state == "READY":
			print("deploy ready")
			return row
		if state in {"ERROR", "CANCELED"}:
			msg = row.get("errorMessage") or "deploy failed"
			raise RuntimeError(msg)
		time.sleep(1)


def full_url(host):
	if host.startswith("http://") or host.startswith("https://"):
		return host
	return f"https://{host}"


def require_build(root):
	root = pathlib.Path(root)
	if not (root / "assets.json").exists():
		raise RuntimeError("missing app/assets.json; run app/build.py first")
	if not (root / "weights.bin").exists():
		raise RuntimeError("missing app/weights.bin; run app/build.py first")


def main():
	app = build.app_root()
	exp = build.experiments_root()
	cfg = load_deploy(exp)
	tok = load_token(exp)
	print(f"app {app}")
	print(f"experiments {exp}")
	print(f"api {cfg['api']}")
	require_build(app)
	project = ensure_project(cfg, tok)
	dep = create_deploy(cfg, tok, app)
	print(f"deploy id {dep['id']}")
	dep = wait_ready(cfg, dep["id"], tok)
	url = dep.get("aliasFinal") or dep.get("url")
	url = full_url(url)
	print(json.dumps({
		"deploy_id": dep["id"],
		"project_id": project["id"],
		"project_name": project["name"],
		"url": url,
	}, indent=2))


if __name__ == "__main__":
	main()
