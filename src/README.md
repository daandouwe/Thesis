# RNNG
Code for the RNNG.

## Data
We use the Penn Treebank. To extract parse configurations, type:
```bash
cd scripts
./prepare-data.sh
```

## Run
For a simple test run with the discriminative model, use
```bash
./main.py train disc --max-lines 1000 --print-every 100
```
