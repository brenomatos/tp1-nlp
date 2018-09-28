import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot(file_path):
    
    df = pd.read_csv(file_path, index_col=False,  escapechar='|', sep=',')
    boxplot = plt.figure()
    plt.boxplot(df["top_minus_correct_similarity"])
    plt.title("Boxplot "+file_path[:-4])
    boxplot.savefig(file_path[:-4]+".jpg")

    df.describe().to_csv("boxplot-"+file_path[:-4]+"-"+"results.csv")

    scatter = plt.figure()
    plt.scatter(df["correct_word_similarity"], df["top_minus_correct_similarity"])
    plt.title("Scatterplot "+file_path[:-4])
    plt.xlabel("Similaridade - Palavra Correta")
    plt.ylabel("Erro - Similaridade")
    scatter.savefig("scatter-"+file_path[:-4]+".jpg")
    # plt.show()



plot("w2v-200-5-1-5-5-0.csv")
