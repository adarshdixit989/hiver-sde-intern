# Data

The full TWCS dataset is not redistributed. Download it from Kaggle and place it at `data/twcs.csv`.

Primary source: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Run:

```bash
python src/prepare_data.py --input data/twcs.csv --output data/apple_pairs.csv --brand AppleSupport
```
