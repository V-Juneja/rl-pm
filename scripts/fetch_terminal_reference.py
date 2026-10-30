import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def main():
    reference = json.loads((ROOT / "data/terminal-reference-sources.json").read_text())
    output = ROOT / "data/external/sp500-data/terminal-reference"
    output.mkdir(parents=True, exist_ok=True)
    for name, info in reference["files"].items():
        path = output / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]:
            print(f"Cached: {name}")
            continue
        request = Request(info["url"], headers={"User-Agent": "RLPMResearch/1.0"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != info["sha256"]:
            raise ValueError(f"Reference changed: {name}; review before updating its pinned hash")
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(path)
        print(f"Downloaded: {name}")
    (output / "provenance.json").write_text(json.dumps(reference, indent=2) + "\n")


if __name__ == "__main__":
    main()
