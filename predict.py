"""Name the chemical in a Raman spectrum, using the checkpoint raman_model.npz.

Give the spectrum (3648 intensities) in any of these ways:

    python predict.py --row 5              row 5 of spec_log_test.csv
    python predict.py my_spectrum.csv      a file of numbers, one spectrum per row
    Get-Clipboard | python predict.py      numbers you have copied
    python predict.py "-30.24,-38.3,..."   numbers typed on the command line

Or from Python, with a list or numpy array:

    from predict import predict_chemical
    name = predict_chemical(spectrum)
"""
import re
import sys
from pathlib import Path

import numpy as np

from raman_detector import HERE, N_PIXELS, load_model, match, preprocess, read_log

NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def identify(spectra):
    """Chemical name and similarity (0 to 0.5) for each spectrum."""
    x = np.asarray(spectra, dtype=float).reshape(-1, N_PIXELS)
    library, labels, _ = load_model()
    best, score, _, _ = match(library, labels, preprocess(x))
    return best, score


def predict_chemical(spectrum):
    """Chemical name for one spectrum of 3648 intensities."""
    return str(identify(spectrum)[0][0])


def parse_numbers(text):
    values = np.array(NUMBER.findall(text), dtype=float)
    if len(values) == 0 or len(values) % N_PIXELS:
        raise SystemExit(f"expected {N_PIXELS} numbers per spectrum, found {len(values)}")
    return values.reshape(-1, N_PIXELS)


if __name__ == "__main__":
    args = sys.argv[1:]
    actual = None
    if args and args[0] in ("-h", "--help"):
        raise SystemExit(__doc__)
    if args and args[0] == "--row":
        row = int(args[1])
        lines = (HERE / "spec_log_test.csv").read_text(encoding="utf-8-sig").splitlines()
        if not 1 <= row <= len(lines):
            raise SystemExit(f"row {row} is out of range (spec_log_test.csv has {len(lines)} rows)")
        spectra = parse_numbers(lines[row - 1])
        actual = read_log(HERE / "chem_log_test.csv")[row - 1][1].strip()
    elif len(args) == 1 and Path(args[0]).is_file():
        spectra = parse_numbers(Path(args[0]).read_text(encoding="utf-8-sig"))
    elif args:
        spectra = parse_numbers(" ".join(args))
    else:
        spectra = parse_numbers(sys.stdin.read())

    for name, score in zip(*identify(spectra)):
        print(f"{name}  (similarity {score:.3f})" + (f"  actual: {actual}" if actual else ""))
