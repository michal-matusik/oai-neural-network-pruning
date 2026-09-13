"""Download the official task repository archive and copy its checked-in arrays."""

import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "https://github.com/OlimpiadaAI/I-OlimpiadaAI/archive/refs/heads/main.tar.gz"


if __name__ == "__main__":
    output = Path("data")
    output.mkdir(exist_ok=True)
    archive = output / "official-task.tar.gz"
    urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as source:
        members = [member for member in source.getmembers() if "/first_stage/pruning/" in member.name and member.name.endswith(".npy")]
        source.extractall(output / "source", members=members, filter="data")
    located = next((output / "source").glob("*/first_stage/pruning"))
    for path in located.glob("*.npy"):
        shutil.copy2(path, output / path.name)
