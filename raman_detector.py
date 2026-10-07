"""Raman chemical detector built from chem_log.csv + spec_log.csv.

It works by library matching: every labelled spectrum is cleaned and stored,
and a new spectrum is given the label of the stored spectrum it most resembles.
If nothing resembles it closely enough, it is reported as "unknown".

    python raman_detector.py train
    python raman_detector.py split
    python raman_detector.py train --chem chem_log_train.csv --spec spec_log_train.csv
    python raman_detector.py test
    python raman_detector.py predict my_spectra.csv
    python raman_detector.py export my_spectra.csv --rows 1 2

Spectra files have one spectrum per row, 3648 comma-separated intensities,
no header (the same layout as spec_log.csv).

"export" writes a spectrum as a two-column wavenumber,intensity file, the
format online Raman libraries accept. The wavenumber axis is an estimate, see
WAVENUMBER_COEFFS below.
"""
import argparse
import csv
import datetime as dt
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "raman_model.npz"
N_PIXELS = 3648
MIN_SAMPLES = 5          # chemicals with fewer spectra are left out of the library
UNLABELLED = "not in lib"
BASELINE_WINDOW = 151    # pixels; wider than any Raman peak, narrower than the fluorescence hump
DATE_FORMATS = ("%m/%d/%Y %I:%M:%S %p", "%d-%m-%Y %H:%M:%S")

# Raman shift (cm-1) as a cubic in pixel index. The CSVs carry no axis, so this
# was fitted to 23 known peaks of naphthalene, urea, acetone, hexamine, PETN,
# RDX and the nitrates found in spec_log.csv. Fit error is about 5 cm-1 (10 at
# worst), and the instrument drifts by a few cm-1 between sessions.
WAVENUMBER_COEFFS = (1.3176252236910815e-08, -0.00015867381837508107, 1.3467718440322787, -115.59933085906822)
EXPORT_RANGE = (100.0, 3300.0)   # below 100 cm-1 is laser line, not sample


def wavenumbers():
    return np.polyval(WAVENUMBER_COEFFS, np.arange(N_PIXELS))


def preprocess(spectra):
    """Trim detector edges, smooth, remove the broad background, scale to unit length."""
    x = np.asarray(spectra, dtype=float)[:, 20:-8]
    x = savgol_filter(x, 11, 3, axis=1)
    baseline = uniform_filter1d(minimum_filter1d(x, BASELINE_WINDOW, axis=1), BASELINE_WINDOW, axis=1)
    x = x - baseline
    x = x - x.min(axis=1, keepdims=True)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x[:, ::2]


def load_spectra(path):
    x = np.loadtxt(path, delimiter=",", ndmin=2)
    if x.shape[1] != N_PIXELS:
        raise SystemExit(f"{path}: expected {N_PIXELS} values per row, found {x.shape[1]}")
    return x


def parse_time(text):
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognised timestamp: {text!r}")


def match(library, labels, queries):
    """Best and second-best chemical for each query, by cosine similarity."""
    sim = queries @ library.T
    classes = np.unique(labels)
    per_class = np.stack([sim[:, labels == c].max(axis=1) for c in classes], axis=1)
    order = np.argsort(-per_class, axis=1)
    rows = np.arange(len(queries))
    best, second = order[:, 0], order[:, 1]
    return classes[best], per_class[rows, best], classes[second], per_class[rows, second]


def read_log(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.reader(f) if r]


def split(args):
    """Hold back a share of every chemical's spectra as a test set."""
    chem = [l for l in (HERE / "chem_log.csv").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    spec = [l for l in (HERE / "spec_log.csv").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if len(chem) != len(spec):
        raise SystemExit("chem_log.csv and spec_log.csv have different row counts")
    labels = np.array([next(csv.reader([l]))[1].strip() for l in chem])

    rng = np.random.default_rng(args.seed)
    test = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        rows = rng.permutation(np.flatnonzero(labels == c))
        test[rows[:round(len(rows) * args.test_fraction)]] = True

    for name, part in (("train", ~test), ("test", test)):
        for stem, lines in (("chem_log", chem), ("spec_log", spec)):
            out = HERE / f"{stem}_{name}.csv"
            out.write_text("".join(lines[i] + "\n" for i in np.flatnonzero(part)), encoding="utf-8")
        print(f"{name}: {part.sum()} spectra -> chem_log_{name}.csv, spec_log_{name}.csv")


def train(args):
    log = read_log(args.chem)
    labels = np.array([r[1].strip() for r in log])
    days = np.array([parse_time(r[0]).strftime("%Y-%m-%d") for r in log])
    spectra = load_spectra(args.spec)
    if len(spectra) != len(labels):
        raise SystemExit(f"{args.chem} and {args.spec} have different row counts")

    counts = Counter(labels)
    keep = np.array([l != UNLABELLED and counts[l] >= MIN_SAMPLES for l in labels])
    x, y, g = preprocess(spectra[keep]), labels[keep], days[keep]
    skipped = sorted(l for l, n in counts.items() if l != UNLABELLED and n < MIN_SAMPLES)
    print(f"Library: {len(y)} spectra, {len(set(y))} chemicals")
    print(f"Left out (fewer than {MIN_SAMPLES} spectra): {', '.join(skipped)}")

    # Test on whole measurement days the library has not seen. Spectra taken
    # seconds apart are near-copies, so a random split would flatter the result.
    pred = np.empty_like(y)
    score = np.zeros(len(y))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in folds.split(x, y, g):
        pred[te], score[te], _, _ = match(x[tr], y[tr], x[te])
    correct = pred == y

    print(f"\nAccuracy on unseen days: {correct.mean():.1%} ({correct.sum()}/{len(y)})")
    print(f"{'chemical':22s} {'spectra':>7s} {'days':>5s} {'correct':>8s}")
    for c in sorted(set(y), key=lambda c: -counts[c]):
        m = y == c
        n_days = len(set(g[m]))
        note = "" if n_days > 1 else "  (one day only: cannot be tested fairly)"
        print(f"{c:22s} {m.sum():7d} {n_days:5d} {correct[m].mean():8.0%}{note}")

    # Below this similarity a match is reported as unknown.
    threshold = float(np.percentile(score[correct], 5))
    accepted = score >= threshold
    print(f"\nUnknown threshold (similarity): {threshold:.3f}")
    print(f"  matches kept: {accepted.mean():.1%}, of which correct: {correct[accepted].mean():.1%}")
    print(f"  matches rejected: {(~accepted).sum()}, of which were wrong: {(~correct[~accepted]).sum()}")

    np.savez_compressed(MODEL_PATH, library=x.astype(np.float32), labels=y, threshold=threshold)
    print(f"\nSaved {MODEL_PATH.name}")


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit("No model found. Run: python raman_detector.py train")
    model = np.load(MODEL_PATH)
    return model["library"].astype(float), model["labels"], float(model["threshold"])


def test(args):
    """Score the saved model against spectra whose chemical is known."""
    library, labels, threshold = load_model()
    truth = np.array([r[1].strip() for r in read_log(args.chem)])
    queries = preprocess(load_spectra(args.spec))
    if len(queries) != len(truth):
        raise SystemExit(f"{args.chem} and {args.spec} have different row counts")
    best, score, _, _ = match(library, labels, queries)
    verdict = np.where(score >= threshold, best, "unknown")

    known = np.isin(truth, labels)
    correct = verdict == truth
    print(f"Chemicals in the library: {correct[known].mean():.1%} correct ({correct[known].sum()}/{known.sum()})")
    print(f"{'chemical':22s} {'spectra':>7s} {'correct':>8s} {'unknown':>8s}")
    for c in sorted(set(truth[known]), key=lambda c: -(truth == c).sum()):
        m = truth == c
        print(f"{c:22s} {m.sum():7d} {correct[m].mean():8.0%} {(verdict[m] == 'unknown').mean():8.0%}")
    if (~known).any():
        rejected = verdict[~known] == "unknown"
        print(f"\nNot in the library: {rejected.mean():.1%} reported as unknown ({rejected.sum()}/{(~known).sum()})")


def predict(args):
    library, labels, threshold = load_model()
    queries = preprocess(load_spectra(args.spectra))
    best, score, second, score2 = match(library, labels, queries)

    out = Path(args.output) if args.output else Path(args.spectra).with_name(Path(args.spectra).stem + "_predictions.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "prediction", "closest_match", "similarity", "second_match", "second_similarity"])
        for i in range(len(queries)):
            verdict = best[i] if score[i] >= threshold else "unknown"
            w.writerow([i + 1, verdict, best[i], f"{score[i]:.3f}", second[i], f"{score2[i]:.3f}"])
    verdicts = Counter(b if s >= threshold else "unknown" for b, s in zip(best, score))
    print(f"{len(queries)} spectra -> {out}")
    for name, n in verdicts.most_common():
        print(f"  {n:4d}  {name}")


def export(args):
    spectra = load_spectra(args.spectra)
    axis = wavenumbers()
    keep = (axis >= EXPORT_RANGE[0]) & (axis <= EXPORT_RANGE[1])
    for row in args.rows:
        if not 1 <= row <= len(spectra):
            raise SystemExit(f"row {row} is out of range (file has {len(spectra)} rows)")
        y = spectra[row - 1]
        if args.clean:
            y = savgol_filter(y, 11, 3)
            y = y - uniform_filter1d(minimum_filter1d(y, BASELINE_WINDOW), BASELINE_WINDOW)
            y = np.clip(y, 0, None)
        out = Path(args.spectra).with_name(f"{Path(args.spectra).stem}_row{row}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wavenumber", "intensity"])
            w.writerows((f"{a:.2f}", f"{b:.2f}") for a, b in zip(axis[keep], y[keep]))
        print(f"row {row} -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("split", help="split chem_log.csv and spec_log.csv into train and test files")
    p.add_argument("--test-fraction", type=float, default=0.2, help="share of each chemical held back for testing")
    p.add_argument("--seed", type=int, default=0, help="random seed, so the split can be repeated")
    p.set_defaults(func=split)
    p = sub.add_parser("train", help="build the library from a chemical log and its spectra")
    p.add_argument("--chem", default=HERE / "chem_log.csv", help="CSV of timestamp,chemical rows")
    p.add_argument("--spec", default=HERE / "spec_log.csv", help="CSV of the matching spectra")
    p.set_defaults(func=train)
    p = sub.add_parser("test", help="score the saved model against labelled spectra")
    p.add_argument("--chem", default=HERE / "chem_log_test.csv", help="CSV of timestamp,chemical rows")
    p.add_argument("--spec", default=HERE / "spec_log_test.csv", help="CSV of the matching spectra")
    p.set_defaults(func=test)
    p = sub.add_parser("predict", help="identify the spectra in a CSV file")
    p.add_argument("spectra", help="CSV with one 3648-value spectrum per row")
    p.add_argument("-o", "--output", help="where to write the predictions CSV")
    p.set_defaults(func=predict)
    p = sub.add_parser("export", help="write spectra as wavenumber,intensity files for online tools")
    p.add_argument("spectra", help="CSV with one 3648-value spectrum per row")
    p.add_argument("--rows", type=int, nargs="+", default=[1], help="row numbers to export, starting at 1")
    p.add_argument("--clean", action="store_true", help="smooth and remove the fluorescence background first")
    p.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)
