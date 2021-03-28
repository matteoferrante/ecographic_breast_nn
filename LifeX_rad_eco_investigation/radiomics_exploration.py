"""Questo codice è per l'esplorazione delle correlazioni e dei dati trovati in diversi casi da LifeX"""
import pandas as pd
from scipy import stats
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# boxplot generation
from sklearn.model_selection import cross_val_score


def boxplot(data, column,target_dir):


    act_dir=os.path.join(target_dir,"boxplots")
    os.makedirs(act_dir, exist_ok=True)

    print(f"[RUN] Running boxplot analysis for {column}", end=" ")
    fig, ax = plt.subplots(1)
    sns.boxplot(x="status", y=column, data=data, hue="status", dodge=False, ax=ax)

    # run t test
    s, p = stats.ttest_ind(data[data.status == 0][column], data[data.status == 1][column],
                           equal_var=False)

    print(f" -- p value: {p}")
    ax.text(0.5, 0.8, f'p value: {round(p, 4)}', horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, bbox={'facecolor': 'grey', 'alpha': 0.3, 'pad': 10})

    plt.savefig(f"{os.path.join(act_dir, f'box_plot_{column}.png')}")
    return p

def study_corr(data,target_dir,correlation_threshold=0.8):

    print(f"[INFO] 2. Analysis of redudant features..")

    d=data.dropna(axis="columns")
    c=d.corr()
    fig,ax=plt.subplots(1,figsize=(60,60))
    sns.heatmap(c,linewidths=1.,annot=True,ax=ax)
    plt.savefig(os.path.join(target_dir,"correlations.png")
                )
    ##here selection of features correlated

    print(f"[INFO] Searching for correlated features with pearson greater than {correlation_threshold}")
    c=abs(c)
    correlations={}
    for col in c.columns:
        x=c[c>correlation_threshold]

        correlations[col]=x[col].dropna().to_dict().keys()
        print(f"[RUN] correlated with {col}:\n{correlations[col]}")


    #selection data using COV
    print(f"[INFO] Choosing bewteen redudant features using CoV")

    selected=[]
    for fl in correlations.values():
        fl=list(fl)
        cov=stats.variation(d[fl])
        selected.append(fl[cov.argmax()])

    selected=list(set(selected))
    print(f"[INFO] {len(selected)}/{len(d.columns)} features was selected as not redudant..")

    print(f"[INFO] Not redudant features: \n{selected}")
    return correlations,selected


def study_stability(datas,stable_threshold):
    print(f"[INFO] 0. Analysis of stable features using Kruskal-Wallis test:\n Each features that shows a p-value below {stable_threshold} for stability Kruskal test will be discarded")

    columns=datas["abs64"].columns
    stable = []
    kruskal={}
    print(
        f"[INFO] 0. Analysis of stable features using Kruskal-Wallis test:\n Each features that shows a p-value below {stable_threshold} for stability Kruskal test will be discarded")
    for c in columns:
        print(f"[RUN] Running analysis for {c}")
        try:
            s, p = stats.kruskal(datas["abs64"][c], datas["abs128"][c], datas["abs256"][c])
            kruskal[c] = p
            if p > stable_threshold:
                stable.append(c)
        except:
            pass

    print(f"[STABILITY] {len(stable)}/{len(columns)} features passed the stability test.")

    datas["abs64"] = datas["abs64"][stable]
    datas["abs128"] = datas["abs128"][stable]
    datas["abs256"] = datas["abs256"][stable]
    return datas, stable, kruskal



###INZIO


datas={"abs64":pd.read_csv("TextureSession_abs64bin_first_analysis.csv",sep=";"),"abs128":pd.read_csv("TextureSession_abs128bin_first_analysis.csv",sep=";"),"abs256":pd.read_csv("TextureSession_abs256bin_first_analysis.csv",sep=";")}
reports=[]
correlations=[]

datas,stable,kruskal=study_stability(datas,stable_threshold=0.05)



datas_not_redudant={}

for k,v in datas.items():

    ###BOX PLOT ANALYSIS
    print(f"[INFO] 1. Statistically significant indipendent features..")
    os.makedirs(k,exist_ok=True)
    data_num=v.select_dtypes("number")
    data_num["status"]=v["status"].map({"benigno":0,"maligno":1})
    p = {}
    for c in data_num.columns:
        p[c] = boxplot(data_num, c,k)


    p=dict(sorted(p.items(), key=lambda item: item[1]))
    reports.append(p)


    ### REDUDANT FEATURES
    cor,not_redudant=study_corr(data_num,k,correlation_threshold=0.8)

    datas_not_redudant[k]=data_num[not_redudant]
    correlations.append(cor)


    ###LAVORARE QUA -> Aggiungere Correlazioni, Clustering su correlazioni, Wilcoxon Test tra dati e Modelli, CN2 RULE?


os.makedirs("output",exist_ok=True)
p_values=pd.DataFrame.from_dict(reports)
p_values.to_csv(os.path.join("output","p_values.csv"))


print(f"[INFO] Not redudant features in each case ")
for k,v in datas_not_redudant.items():
    print(f"[{k}] {len(v.columns)},{v.columns}")


print(f"[INFO] Selecting the mininum number of not redudant and stable features..")

datas_not_redudant=dict(sorted(datas_not_redudant.items(),key=lambda x:len(x[1])))

final_features=list(datas_not_redudant.values())[0]

print(f"[INFO] Selected  {len(final_features)} features: \n{final_features}")

print(f"[INFO] Fitting models using those features..")

clf={}
target="status"
clf_results=[]
f = open("models.txt", "w")
for k,v in datas_not_redudant.items():

    print(f"[INFO] Running models on {k}")
    clf_dict={}
    clf_rf=RandomForestClassifier()
    clf_lr=LogisticRegression()
    X=v.drop(target,axis=1)
    y=v[target]
    scores = cross_val_score(clf_rf, X, y, cv=5)


    print(f"[INFO] Results for 5-fold cross validation using RandomForest {scores}")

    f.write(f"Analysis for {k}\n\n Random Forest:\t {np.mean(scores)}\n")

    scores = cross_val_score(clf_lr, X, y, cv=5)
    print(f"[INFO] Results for 5-fold cross validation using LogisticRegression {scores}")

    f.write(f"Logistic Regression:\t {np.mean(scores)}\n\n\n")


f.close()

### TUTTO NON SIGNIFICATIVO -> POSSO PROVARE A FARE CLUSTERING SULLE MATRICI DELLE DISTANZE