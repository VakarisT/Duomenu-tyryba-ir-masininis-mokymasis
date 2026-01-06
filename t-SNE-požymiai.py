import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook

# --- Sąrašas failų, kuriuos apdorosime ---
path = "EKG_duomenys_filled.csv"

# --- Funkcija išskirčių aptikimui ---
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

# --- Pagrindinė funkcija vienam failui apdoroti ---
def analizuoti_faila(df,X,i,perplexity=10, learning_rate='auto', early_exaggeration=12.0, angle=0.0, n_iter_=1500, random_state=42):

    feature_names = X.columns.tolist()

    # Išskirčių aptikimas kiekvienam požymiui
    outlier_info = {}
    for column in feature_names:
        inner_outliers, outer_outliers, Q1, Q3, H = detect_outliers_iqr(X, column)
        outlier_info[column] = {
            'inner_outliers': inner_outliers,
            'outer_outliers': outer_outliers,
            'Q1': Q1,
            'Q3': Q3,
            'H': H,
            'inner_lower': Q1 - 1.5 * H,
            'inner_upper': Q3 + 1.5 * H,
            'outer_lower': Q1 - 3 * H,
            'outer_upper': Q3 + 3 * H
        }
    
    # Bendros išskirčių žymės
    has_inner_outlier = pd.Series([False] * len(X))
    has_outer_outlier = pd.Series([False] * len(X))
    
    for column in feature_names:
        has_inner_outlier = has_inner_outlier | outlier_info[column]['inner_outliers']
        has_outer_outlier = has_outer_outlier | outlier_info[column]['outer_outliers']
    
    # t-SNE skaičiavimas
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        init='random',
        angle=angle,
        max_iter = n_iter_,
        random_state=random_state
    )
    tsne_results = tsne.fit_transform(X)

    # Sukuriame DataFrame su rezultatais ir išskirčių informacija
    tsne_df = pd.DataFrame({
        'TSNE1': tsne_results[:, 0],
        'TSNE2': tsne_results[:, 1],
        'label': df['label'],
        'classification': df['label'],
        'has_inner_outlier': has_inner_outlier,
        'has_outer_outlier': has_outer_outlier
    })

    # Vizualizacija su išskirčių paryškinimu
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
    
    # Vidinės išskirtys
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
    
    # Išorinės išskirtys
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

    # Pavadinimai, legenda, rodymas
    plt.title(f"t-SNE vizualizacija su išskirčių aptikimu - {i} variantas", fontsize=14, pad=15)
    plt.xlabel("t-SNE komponentė 1")
    plt.xticks([-125,-100,-75,-50,-25,0,25,50,75,100,125])
    plt.ylabel("t-SNE komponentė 2")
    plt.yticks([-125,-100,-75,-50,-25,0,25,50,75,100,125])
    plt.legend(title='Klasifikacija ir išskirtys', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Išsaugome grafiką
    output_plot = path.replace("EKG_duomenys_filled.csv", "Vizualizacija_požymiai_" + str(i) + ".png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    # plt.show()

    # print(f"Vizualizacija išsaugota kaip '{output_plot}'")

    # Grąžiname duomenis metrikų skaičiavimui
    return {
        'filename': str(i),
        'data': df,
        'X': X,
        'tsne_results': tsne_results,
        'tsne_df': tsne_df,
        'outlier_info': outlier_info,
        'feature_names': feature_names,
        'stress': tsne.kl_divergence_,  # arba kita metrika
        'perplexity': perplexity,
        'learning_rate': learning_rate,
        'early_exaggeration': early_exaggeration,
        'angle': angle,
        'n_iter': n_iter_,
        'random_state': random_state
    }

# --- Funkcija ataskaitos generavimui ---
def generate_report(results_list):
    """Generuoja išsamią ataskaitą .xlsx faile"""
    output_path = "Požymių_ataskaita.xlsx"

    # Sukuriame failą
    for result in results_list:
        summary_data = []

        tsne_params = {
            'stress': result.get('stress', 'N/A'),
            'perplexity': result.get('perplexity', 'N/A'),
            'learning_rate': result.get('learning_rate', 'N/A'),
            'early_exaggeration': result.get('early_exaggeration', 'N/A'),
            'angle': result.get('angle', 'N/A'),
            'n_iter': result.get('n_iter', 'N/A'),
            'random_state': result.get('random_state', 'N/A')
        }

        total_samples = len(result['data'])
        total_inner = result['tsne_df']['has_inner_outlier'].sum()
        total_outer = result['tsne_df']['has_outer_outlier'].sum()

        summary_data.extend([
            {'Parametras': 'Stress', 'Reikšmė': tsne_params['stress']},
            {'Parametras': 'Perplexity', 'Reikšmė': tsne_params['perplexity']},
            {'Parametras': 'Learning Rate', 'Reikšmė': tsne_params['learning_rate']},
            {'Parametras': 'Early Exaggeration', 'Reikšmė': tsne_params['early_exaggeration']},
            {'Parametras': 'Angle', 'Reikšmė': tsne_params['angle']},
            {'Parametras': 'n_iter', 'Reikšmė': tsne_params['n_iter']},
            {'Parametras': 'Random State', 'Reikšmė': tsne_params['random_state']},
            {'Parametras': 'Vidinių išskirčių kiekis', 'Reikšmė': total_inner},
            {'Parametras': 'Išorinių išskirčių kiekis', 'Reikšmė': total_outer},
            {'Parametras': 'Vidinių išskirčių procentas', 'Reikšmė': f"{(total_inner / total_samples) * 100:.2f}%"},
            {'Parametras': 'Išorinių išskirčių procentas', 'Reikšmė': f"{(total_outer / total_samples) * 100:.2f}%"},
            {'Parametras': 'Bendras įrašų kiekis', 'Reikšmė': total_samples}
        ])

        summary_df = pd.DataFrame(summary_data)

        # Unikalus lapo pavadinimas — kad nepasikartotų
        sheet_name = result['filename'] + " bandymas"

        with pd.ExcelWriter(output_path, mode='a', engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # --- IŠSLAPTINAME VISUS LAPUS (jei kuris nors paslėptas) ---
    wb = load_workbook(output_path)
    for sheet in wb.worksheets:
        sheet.sheet_state = "visible"  # atidaro visus
    wb.active = 0  # pirmas lapas aktyvus
    wb.save(output_path)

    print(f"Ataskaita išsaugota faile: {output_path}")

# --- Vykdymas ---
def main():
    results = []

    # 1. Nuskaitymas
    df = pd.read_csv(path, sep=";")

    # 2. Pasirenkame stulpelius (skaitinius požymius)
    for i in range(4):
        if (i == 0):
            X = df.iloc[:, 0:6]
        if (i == 1):
            pozymiai = ['RR_r_0/RR_r_1','R_val','S_val']
            X = df[pozymiai]
        if (i == 2):
            pozymiai = ['RR_r_0/RR_r_1','R_val','R_pos']
            X = df[pozymiai] 
        if (i == 3):
            pozymiai = ['R_val','S_pos','R_pos']
            X = df[pozymiai] 
    # 3. Analizuojame failą
    try:
        result = analizuoti_faila(df,X,i+1)
        print(result['filename'])
        results.append(result)
    except Exception as e:
        print(f"Klaida apdorojant failą {path}: {e}")

    
    # 4. Generuojame ataskaitą
    if results:
        generate_report(results)
        print("\n=== Ataskaita sugeneruota 'Požymių_ataskaita.xlsx' ===")

if __name__ == "__main__":
    main()