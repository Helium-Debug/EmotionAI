import pandas as pd

def load_goemotions():
    df1 = pd.read_csv("goemotions_1.csv")
    df2 = pd.read_csv("goemotions_2.csv")
    df3 = pd.read_csv("goemotions_3.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)
    return df
