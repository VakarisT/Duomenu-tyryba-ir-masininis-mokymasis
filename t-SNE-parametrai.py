import pandas as pd 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
from sklearn.neighbors import NearestNeighbors
import os  # reikalingas failų egzistavimo tikrinimui

# --- Sąrašas failų, kuriuos apdorosime ---
path = "EKG_norm_duomenys_filled.csv"

def detect_outliers_iqr(data, column):
    """
    Aptinka išskirtis pagal IQR metodą
    Grąžina: vidinės išskirtys, išorinės išskirtys, Q1, Q3, H
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    H = Q3 - Q1
    
    inner_lower = Q1 - 1.5 * H
    inner_upper = Q3 + 1.5 * H
    outer_lower = Q1 - 3 * H
    outer_upper = Q3 + 3 * H
    
    # Vidinės išskirtys
    inner_outliers = (data[column] < inner_lower) | (data[column] > inner_upper)
    
    # Išorinės išskirtys
    outer_outliers = (data[column] < outer_lower) | (data[column] > outer_upper)
    
    return inner_outliers, outer_outliers, Q1, Q3, H

def analizuoti_faila(df, X, bandymas,
                     perplexity=30, 
                     learning_rate=200, 
                     metric='euclidean',  # 🔧 pridėtas parametras
                     early_exaggeration=12.0, 
                     angle=0.0, 
                     n_iter_=1500, 
                     random_state=42):

    feature_names = X.columns.tolist()

    # --- Išskirčių aptikimas ---
    outlier_info = {}
    for column in feature_names:
        inner_outliers, outer_outliers, Q1, Q3, H = detect_outliers_iqr(X, column)
        outlier_info[column] = {
            'inner_outliers': inner_outliers,
            'outer_outliers': outer_outliers,
            'Q1': Q1,
            'Q3': Q3,
            'H': H
        }

    has_inner_outlier = pd.Series([False] * len(X))
    has_outer_outlier = pd.Series([False] * len(X))
    for column in feature_names:
        has_inner_outlier |= outlier_info[column]['inner_outliers']
        has_outer_outlier |= outlier_info[column]['outer_outliers']

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        init='random',
        angle=angle,
        max_iter=n_iter_,
        random_state=random_state,
        metric=metric
    )

    tsne_results = tsne.fit_transform(X)

    tsne_df = pd.DataFrame({
        'TSNE1': tsne_results[:, 0],
        'TSNE2': tsne_results[:, 1],
        'label': df['label'],
        'classification': df['label'],
        'has_inner_outlier': has_inner_outlier,
        'has_outer_outlier': has_outer_outlier
    })
    
    # 7. Vizualizacija su išskirčių paryškinimu
    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid", font_scale=1.1)

    # Pagrindinis scatterplot
    scatter = sns.scatterplot(
        data=tsne_df,
        x='TSNE1', y='TSNE2',
        hue='classification',
        palette='tab10',
        s=80,
        alpha=0.8
    )
    
    # Paryškiname išskirtis
    # Vidinės išskirtys - storesnis kraštas
    inner_outliers_df = tsne_df[tsne_df['has_inner_outlier']]
    if not inner_outliers_df.empty:
        plt.scatter(
            x=inner_outliers_df['TSNE1'],
            y=inner_outliers_df['TSNE2'],
            facecolors='none',
            edgecolors='black',
            linewidths=1,
            s=120,
            alpha=0.8,
            label='Vidinės išskirtys'
        )
    
    # Išorinės išskirtys - dar storesnis kraštas
    outer_outliers_df = tsne_df[tsne_df['has_outer_outlier']]
    if not outer_outliers_df.empty:
        plt.scatter(
            x=outer_outliers_df['TSNE1'],
            y=outer_outliers_df['TSNE2'],
            facecolors='none',
            edgecolors='red',
            linewidths=1,
            s=120,
            alpha=0.8,
            label='Išorinės išskirtys'
        )

    # --- artimiausių kaimynų skaičiavimas ---
    n_neighbors = 5  # kiek kaimynų skaičiuoti
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nbrs.fit(tsne_results)
    distances, indices = nbrs.kneighbors(tsne_results)

    # Pridėsime vidutinį atstumą iki kaimynų
    tsne_df['avg_neighbor_dist'] = distances[:, 1:].mean(axis=1)  # atmetame save (pirmą kaimyną)
    mean_distance = tsne_df['avg_neighbor_dist'].mean()

    plt.title("Vizualizacija " + bandymas)
    plt.xlabel("t-SNE komponentė 1")
    plt.ylabel("t-SNE komponentė 2")
    plt.legend(title='Klasifikacija ir išskirtys', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_plot = path.replace("EKG_norm_duomenys_filled.csv", f"Vizualizacija_" + bandymas + ".png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Grąžiname rezultatus su kaimynų duomenimis ---
    return {
        'filename': bandymas,
        'data': df,
        'X': X,
        'tsne_results': tsne_results,
        'tsne_df': tsne_df,
        'outlier_info': outlier_info,
        'feature_names': feature_names,
        'stress': tsne.kl_divergence_,
        'perplexity': perplexity,
        'learning_rate': learning_rate,
        'metric': metric,
        'early_exaggeration': early_exaggeration,
        'angle': angle,
        'n_iter': n_iter_,
        'random_state': random_state,
        'mean_neighbor_distance': mean_distance,
        'neighbor_indices': indices,
        'neighbor_distances': distances
    }

# --- Ataskaitos generavimas ---
def generate_report(results_list):
    print ("Generuoja išsamią ataskaitą .xlsx faile")
    output_path = "Parametrų_ataskaita.xlsx"
    for result in results_list:
            # t-SNE parametrų ir išskirčių santrauka
            summary_data = []
                
            # t-SNE parametrai
            tsne_params = {
                'stress': result.get('stress', 'N/A'),
                'perplexity': result.get('perplexity', 'N/A'),
                'learning_rate': result.get('learning_rate', 'N/A'),
                'early_exaggeration': result.get('early_exaggeration', 'N/A'),
                'angle': result.get('angle', 'N/A'),
                'n_iter': result.get('n_iter', 'N/A'),
                'random_state': result.get('random_state', 'N/A')
            }
                
            # Išskirčių kiekiai
            total_samples = len(result['data'])
            total_inner = result['tsne_df']['has_inner_outlier'].sum()
            total_outer = result['tsne_df']['has_outer_outlier'].sum()
                
            summary_data.append({
                'Parametras': 'Stress',
                'Reikšmė': tsne_params['stress']
            })
            summary_data.append({
                'Parametras': 'Perplexity',
                'Reikšmė': tsne_params['perplexity']
            })
            summary_data.append({
                'Parametras': 'Learning Rate',
                'Reikšmė': tsne_params['learning_rate']
            })
            summary_data.append({
                'Parametras': 'Early Exaggeration',
                'Reikšmė': tsne_params['early_exaggeration']
            })
            summary_data.append({
                'Parametras': 'Learning Rate', 
                'Reikšmė': result['learning_rate']
            })
            summary_data.append({
                 'Parametras': 'Metric', 
                 'Reikšmė': result['metric']
            })
            summary_data.append({
                'Parametras': 'Angle',
                'Reikšmė': tsne_params['angle']
            })
            summary_data.append({
                'Parametras': 'n_iter',
                'Reikšmė': tsne_params['n_iter']
            })
            summary_data.append({
                'Parametras': 'Random State',
                'Reikšmė': tsne_params['random_state']
            })
            summary_data.append({
                'Parametras': 'Vidinių išskirčių kiekis',
                'Reikšmė': total_inner
            })
            summary_data.append({
                'Parametras': 'Išorinių išskirčių kiekis',
                'Reikšmė': total_outer
            })
            summary_data.append({
                'Parametras': 'Vidinių išskirčių procentas',
                'Reikšmė': f"{(total_inner / total_samples) * 100:.2f}%"
            })
            summary_data.append({
                'Parametras': 'Išorinių išskirčių procentas',
                'Reikšmė': f"{(total_outer / total_samples) * 100:.2f}%"
            })
            summary_data.append({
                'Parametras': 'Bendras įrašų kiekis',
                'Reikšmė': total_samples
            })
                
            summary_df = pd.DataFrame(summary_data)
            sheet_name = result['filename']
            with pd.ExcelWriter(output_path, mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
                summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    wb = load_workbook(output_path)
    for sheet in wb.worksheets:
        sheet.sheet_state = "visible"
    wb.save(output_path)
    print(f"Ataskaita išsaugota faile: {output_path}")

# --- Vykdymas ---
def main():
    results = []
    df = pd.read_csv(path, sep=";")
    X = df[['RR_r_0/RR_r_1','R_val','R_pos']]

    # ČIA keičiami parametrus prieš paleidimą
    bandymas = "metric=cosine"
    perplexity = 40 # default - 40
    learning_rate = 300 # default - 300
    metric = 'cosine'  # galimos reikšmės: 'euclidean' (default), 'manhattan', 'cosine', 'correlation'

    try:
        result = analizuoti_faila(df, X, bandymas, perplexity=perplexity, learning_rate=learning_rate, metric=metric)
        results.append(result)
    except Exception as e:
        print(f"Klaida apdorojant failą: {e}")

    if results:
        generate_report(results)

if __name__ == "__main__":
    main()
