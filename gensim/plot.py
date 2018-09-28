import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_boxplot(file_path):
    df = pd.read_csv(file_path, index_col=False,  escapechar='|', sep=',')
    fig = plt.figure()
    plt.boxplot(df["top_minus_correct_similarity"])
    plt.title("Boxplot")
    plt.xlabel("eixo x")
    plt.ylabel("eixo y")
    fig.savefig(file_path[:-4]+".jpg")
    df.describe().to_csv(file_path[:-4]+"-"+"results.csv")



# w2v-200-5-1-5-5-0.csv
plot_boxplot("w2v-200-5-1-5-5-0.csv")
