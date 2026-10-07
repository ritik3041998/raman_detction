# Raman chemical detector

Names the chemical in a Raman spectrum by comparing it with a library of labelled spectra.
A spectrum is 3648 intensity values, one per detector pixel.

## Setup

Needs Python 3.10 or newer.

```
git clone https://github.com/ritik3041998/raman_detction.git
cd raman_detction
pip install -r requirements.txt
```

The trained model (`raman_model.npz`) is included, so you can predict straight away.

## Predict a chemical

Give `predict.py` a spectrum in any of these ways:

```
python predict.py --row 5               # row 5 of spec_log_test.csv, also prints the actual chemical
python predict.py my_spectrum.csv       # a file of numbers, one spectrum per row
python predict.py "-30.24,-38.3,..."    # numbers typed directly
```

On Windows PowerShell you can also copy the numbers and run `Get-Clipboard | python predict.py`.

Output looks like this:

```
Napthalene  (similarity 0.491)  actual: Napthalene
```

Similarity runs from 0 to 0.5. Close to 0.5 is a near-exact match; below about 0.3 the answer is doubtful.
The model always gives the closest name, even for a chemical it has never seen.

From Python:

```python
from predict import predict_chemical
name = predict_chemical(spectrum)   # list or numpy array of 3648 values
```

## Test on the held-back data

```
python raman_detector.py test
```

This scores the model on `chem_log_test.csv` and `spec_log_test.csv` and prints the accuracy for each chemical.
To score your own labelled data:

```
python raman_detector.py test --chem my_chem_log.csv --spec my_spec_log.csv
```

## Retrain the model

```
python raman_detector.py train --chem chem_log_train.csv --spec spec_log_train.csv
```

This rebuilds `raman_model.npz`. Chemicals with fewer than 5 spectra, and rows labelled `not in lib`, are left out.

## Files

| File | What it is |
|---|---|
| `predict.py` | Names the chemical for a spectrum |
| `raman_detector.py` | Training, testing and export commands |
| `raman_model.npz` | Trained model, built from the training files |
| `chem_log_train.csv`, `spec_log_train.csv` | Training data (80%): labels and spectra, matching row for row |
| `chem_log_test.csv`, `spec_log_test.csv` | Test data (20%), with `not in lib` rows removed |

Data format: `chem_log` files have `timestamp,chemical` per row. `spec_log` files have 3648 comma-separated values per row and no header.

## Chemicals the model knows

Acetone, Ammonium Nitrate, H2O2, Hexamine, HMX, Napthalene, PETN, PNBA, Potassium Chlorate, Potassium Nitrate, RDX, Sodium Azide, Sodium chlorate, Sodium Nitrate, TATB, TNT, UREA.

On the test data it names 95.5% correctly (105 of 110). The test spectra were picked at random from the same sessions as the training spectra, so expect lower accuracy on new measurement days: about 87%.
