import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://api.github.com/repos/Johnbrick123/sp500-data/releases/tags/data"
REQUIRED = {"prices.parquet", "membership_intervals.parquet", "no_trade_days.csv"}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    folder = ROOT / "data/external/sp500-data/rolling"
    folder.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(RELEASE, headers={"User-Agent": "RLPMResearch/1.0"}), timeout=30) as response:
        release = json.load(response)
    assets = {asset["name"]: asset for asset in release["assets"]}
    missing = REQUIRED - set(assets)
    if missing:
        raise ValueError("Release assets missing: " + ", ".join(sorted(missing)))
    for name in sorted(REQUIRED):
        asset = assets[name]
        path = folder / name
        expected = asset.get("digest")
        if path.exists() and path.stat().st_size == asset["size"]:
            if expected and expected == "sha256:" + file_hash(path):
                print(f"Cached: {name}", flush=True)
                continue
        temporary = path.with_suffix(path.suffix + ".part")
        request = Request(asset["browser_download_url"], headers={"User-Agent": "RLPMResearch/1.0"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
        if temporary.stat().st_size != asset["size"]:
            raise ValueError(f"Download size mismatch: {name}")
        if expected and expected != "sha256:" + file_hash(temporary):
            raise ValueError(f"Download hash mismatch: {name}")
        temporary.replace(path)
        print(f"Downloaded: {name}", flush=True)
    (folder / "release.json").write_text(json.dumps(release, indent=2) + "\n")


if __name__ == "__main__":
    main()
