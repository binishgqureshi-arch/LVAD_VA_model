# src/my_model/infer.py
import argparse
import pandas as pd
from .model import predict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--which", choices=["early", "late"], required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(predict(df, which=args.which))

if __name__ == "__main__":
    main()
