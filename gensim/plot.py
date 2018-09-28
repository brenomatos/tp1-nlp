import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot(file_path):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r''+str(file_path[:-4])+'-results')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    df = pd.read_csv(file_path, index_col=False,  escapechar='|', sep=',')
    boxplot = plt.figure()
    plt.boxplot(df["top_minus_correct_similarity"])
    plt.title("Boxplot "+file_path[:-4])
    boxplot.savefig(str(file_path[:-4])+'-results'+"/"+file_path[:-4]+".jpg")

    print(df.head())
    df.describe().to_csv(str(file_path[:-4])+'-results'+"/"+"boxplot-"+file_path[:-4]+"-"+"results.csv")

    scatter = plt.figure()
    plt.scatter(df["correct_word_similarity"], df["top_minus_correct_similarity"])
    plt.title("Scatterplot "+file_path[:-4])
    plt.xlabel("Similaridade - Palavra Correta")
    plt.ylabel("Erro - Similaridade")
    scatter.savefig(str(file_path[:-4])+'-results'+"/scatter-"+file_path[:-4]+".jpg")
    # plt.show()

    try:#mover todos os arquivos de um treino para a mesma pasta
        os.rename(current_directory+"/"+file_path[:-4]+".log",final_directory+'/'+file_path[:-4]+".log" )
        os.rename(current_directory+"/"+file_path,final_directory+'/'+file_path)
    except Exception as e:
        pass



# plot("w2v-200-5-1-5-5-0.csv")
