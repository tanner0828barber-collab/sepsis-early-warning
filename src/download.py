"""Download the public challenge data from PhysioNet.

The data is never stored in this repository. This fetches it from the public
source into data/, skipping any file that is already there, so a second run
is fast.

Source: PhysioNet / Computing in Cardiology Challenge 2019, license CC BY 4.0
https://physionet.org/content/challenge-2019/1.0.0/
"""
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from load import DATA_DIR, HOSPITAL_FOLDERS

INDEX_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/{folder}/"
FILE_URL = "https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/{folder}/{name}"
HEADERS = {"User-Agent": "python-urllib (student research project)"}


def _get(url):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def list_patient_files(hospital):
    """Read the folder index on physionet.org and return every patient file name."""
    folder = HOSPITAL_FOLDERS[hospital]
    page = _get(INDEX_URL.format(folder=folder)).decode("utf-8", errors="replace")
    return sorted(set(re.findall(r'href="(p\d+\.psv)"', page)))


def download_hospital(hospital, data_dir=DATA_DIR, workers=16, limit=None):
    """Download one hospital's patient files. Returns how many were fetched.

    `limit` downloads only the first few files, for a quick check.
    """
    folder = HOSPITAL_FOLDERS[hospital]
    target = Path(data_dir) / folder
    target.mkdir(parents=True, exist_ok=True)

    names = list_patient_files(hospital)
    if limit:
        names = names[:limit]
    missing = [n for n in names if not (target / n).exists()]

    def fetch(name):
        (target / name).write_bytes(_get(FILE_URL.format(folder=folder, name=name)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, missing))
    return len(missing)
