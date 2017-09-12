# Feature Matrices and Target Data

Description of the data files for reproducing the results in the
paper. Note that the data for grain boundary 158 was missing from the
dataset, so the SOAP vectors and LER vectors for that one are set to
zeros.

All the CSV files have a column called PID, which ranges from 1-388
and is the identifier for each GB.

## `asr.npy`

Can be restored using `numpy.load` to produce an array with (388,
3250).

## `ler.csv`

Columns are labelled `T-*` for each of the 145 unique LAEs.

## `mobility.csv`

- TA: 1 if thermally activated, otherwise 0.
- TD: 1 if thermally damped, otherwise 0.
- I: 1 if immobile, otherwise 0.
- C: 1 if constant, otherwise 0.
- X: 1 if unknown mobility, else 0.
- M3: three-class training vector for TA, TD and I as referenced in the paper.

## `shear.csv`

- S4: four-class shear coupling identifiers.
- SC: 1 if shear-coupled, otherwise not shear-coupled; this column was
used for producing results in the paper.
