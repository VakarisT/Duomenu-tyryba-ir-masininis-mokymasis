"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sys

# --- 1. Duomenų įkėlimas ir paruošimas ---

# Čia įrašomas failo pavadinimas
file_path = "EKG_visi.csv"
print("Modifikuotas kodas")

try:
    df = pd.read_csv(file_path, sep=';')
except FileNotFoundError:
    print(f"Failas '{file_path}' nerastas.")
    sys.exit()

df_clean = df.dropna()

y = None 
if 'label' in df_clean.columns:
    y = df_clean['label'].astype(str).values 
    X = df_clean.drop('label', axis=1)
    print("Stulpelis 'label' rastas ir bus naudojamas kombinuotai vizualizacijai.")
else:
    X = df_clean.copy()
    print("Stulpelis 'label'  nerastas. Bus kuriama tik DBSCAN vizualizacija.")

print(f"Duomenys '{file_path}' sėkmingai įkelti. Požymių skaičius: {X.shape[1]}, Eilučių skaičius: {X.shape[0]}")

# --- 2. Duomenų normavimas ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Optimalaus 'eps' nustatymas ---
try:
    min_samples_value = int(input("Įveskite 'min_samples' vertę: "))
except ValueError:
    print("Netinkama įvestis. Naudojama numatytoji vertė 5.")
    min_samples_value = 5

k = min_samples_value 
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
k_distances = np.sort(distances[:, k-1], axis=0)

print("\n--- 'eps' nustatymas ---")
print("Rodomas k-atstumo grafikas. Raskite 'alkūnės' tašką.")

plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.title(f'k-atstumo grafikas (k={min_samples_value})')
plt.xlabel('Taškai (surūšiuoti pagal atstumą)')
plt.ylabel(f'{min_samples_value}-asis artimiausias atstumas (būsimas "eps")')
plt.grid(True)
plt.figtext(0.5, 0.01, 
            "Išnagrinėkite šį grafiką. Optimali 'eps' vertė yra ta, kur kreivė pradeda staigiai kilti (prasideda alkūnė).\n", 
            ha="center", fontsize=10, style='italic')
plt.show()

try:
    eps_value = float(input("Įveskite 'eps' vertę, kurią matėte grafiko alkūnėje: "))
except ValueError:
    print("Netinkama įvestis. Naudojama numatytoji vertė 5.0")
    eps_value = 5.0

# --- 4. DBSCAN klasterizavimas ---
print(f"\nVykdomas DBSCAN su eps={eps_value} ir min_samples={min_samples_value}...")
db = DBSCAN(eps=eps_value, min_samples=min_samples_value)
clusters = db.fit_predict(X_scaled)
# Konvertuojame klasterius į 'str'
clusters_str = [str(c) for c in clusters]

# --- 5. Rezultatų analizė ---
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)
total_points = len(clusters)
noise_percentage = (n_noise_ / total_points) * 100

print("\n--- KLASTERIZAVIMO REZULTATAI ---")
print(f"Bendras rastų klasterių skaičius (be triukšmo): {n_clusters_}")
print(f"Atsiskyrėlių ('-1') dalis: {noise_percentage:.2f}%")

print("\nKlasterių dydžiai (narių skaičius):")
# Naudojame pandas, kad suskaičiuotume kiekvienos etiketės pasikartojimus ir surūšiuotume
cluster_counts = pd.Series(clusters).value_counts().sort_index()
for cluster_label, count in cluster_counts.items():
    if cluster_label == -1:
        print(f"  Klasteris -1 (Triukšmas): {count} narių")
    else:
        print(f"  Klasteris {cluster_label}: {count} narių")

if n_clusters_ > 1:
    # Teisingas silueto skaičiavimas - tik taškams, kurie yra klasteriuose
    # mask_no_outliers = (clusters != -1)
    # X_clustered = X_scaled[clusters]
    # labels_clustered = clusters[mask_no_outliers]
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"Vidutinis silueto balas: {silhouette_avg:.4f}")
else:
    print("\nVidutinis silueto balas: N/A (Nerasta klasterių arba rastas tik vienas)")

# --- 6. t-SNE transformacija ---
print("\nVykdoma t-SNE transformacija vizualizacijai...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1500)
X_tsne = tsne.fit_transform(X_scaled)

# Paruošiame DataFrame vizualizacijoms
df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
df_tsne['dbscan_cluster'] = clusters_str
if y is not None:
    df_tsne['true_label'] = y


# --- 7. Kombinuota vizualizacija ---
print("Braižomas kombinuotas t-SNE grafikas...")

# Naudojame didesnį plotą, kad tilptų legendos
fig, ax = plt.subplots(figsize=(14, 10))

# A. Paruošiame spalvas DBSCAN klasterių užpildymui (fill)
db_clusters_unique = sorted(df_tsne['dbscan_cluster'].unique())
db_palette_list = sns.color_palette("hsv", n_colors=n_clusters_)
fill_map = {}
color_idx = 0
for c in db_clusters_unique:
    if c == '-1':
        fill_map['-1'] = (0.7, 0.7, 0.7) # Pilka triukšmui
    else:
        fill_map[c] = db_palette_list[color_idx]
        color_idx += 1
fill_colors = df_tsne['dbscan_cluster'].map(fill_map)

# B. Paruošiame spalvas 'label' KRAŠTELIAMS (edge)
# Tikriname, ar 'y' egzistuoja
if y is not None:
    labels_unique = sorted(df_tsne['true_label'].unique())
    n_labels = len(labels_unique)
    edge_palette_list = sns.color_palette("tab10", n_colors=n_labels) 
    edge_map = {label: color for label, color in zip(labels_unique, edge_palette_list)}
    edge_colors = df_tsne['true_label'].map(edge_map)
    edge_width = 1.5
else:
    # Jei 'label' nėra, kraštelių nebus
    edge_colors = 'none' 
    edge_width = 0

# C. Braižome grafiką
ax.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], 
           c=fill_colors,          # Užpildymo spalva = DBSCAN klasteris
           edgecolors=edge_colors, # Kraštelio spalva = Tikra klasė ('label')
           s=60,                   # Taško dydis
           linewidths=edge_width,  # Kraštelio plotis
           alpha=0.8)

# D. Kuriame rankines legendas
legend_elements_dbscan = []
for label, color in fill_map.items():
    legend_elements_dbscan.append(mlines.Line2D([0], [0], marker='o', color='none', 
                                                 label=f'Klasteris {label}',
                                                 markerfacecolor=color, markersize=10))
# Pirmoji legenda (DBSCAN)
legend1 = ax.legend(handles=legend_elements_dbscan, title="DBSCAN klasteriai", 
                    loc='upper left', bbox_to_anchor=(1.00, 1))
ax.add_artist(legend1)

if y is not None:
    legend_elements_label = []
    for label, color in edge_map.items():
        legend_elements_label.append(mlines.Line2D([0], [0], marker='o', color='none', 
                                                   label=f'Klasė {label}',
                                                   markerfacecolor='none', 
                                                   markeredgecolor=color, 
                                                   markeredgewidth=edge_width, 
                                                   markersize=10))
    # Antroji legenda (Tikros klasės)
    ax.legend(handles=legend_elements_label, title="Klasės (Label)", 
              loc='upper left', bbox_to_anchor=(1.00, 0.5))

# E. Grafiko nustatymai
plot_title = f'{file_path.split('.')[0]} (eps={eps_value}, min={min_samples_value})'
ax.set_title(plot_title)
ax.set_xlabel('t-SNE komponentas 1')
ax.set_ylabel('t-SNE komponentas 2')
ax.grid(True)

# Padarome vietos legendoms
fig.tight_layout(rect=[0, 0, 0.85, 1]) 

# F. Išsaugome
plot_filename = f"tsne_vizualizacija_{file_path.split('.')[0]}_eps{eps_value}_min{min_samples_value}.png"
try:
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nVizualizacija sėkmingai išsaugota kaip: '{plot_filename}'")
except Exception as e:
    print(f"\nKLAIDA: Nepavyko išsaugoti vizualizacijos. {e}")

plt.show()

# --- 8. Išskirčių pašalinimo ir dimensijos mažinimo įtaka ---
# Išskirčių pašalinimas naudojant bazinį DBSCAN
# db_base = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(X_scaled)
mask_no_outliers = db.labels_ != -1
X_no_outliers = X_scaled[mask_no_outliers]

tsne_2d = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1500).fit_transform(X_scaled)

# Funkcija rezultatams gauti
def run_dbscan(X, eps=eps_value, min_samples=min_samples_value):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels)
    
    if n_clusters > 1:
        # Čia taip pat ištaisome silueto skaičiavimą (neįtraukiame -1)
        mask = labels != -1
        if np.sum(mask) > 1: # Tikriname ar liko taškų
            sil = silhouette_score(X[mask], labels[mask], metric='euclidean')
        else:
            sil = np.nan
    else:
        sil = np.nan
    return n_clusters, noise_ratio, sil

# Apskaičiuojame visus 4 variantus
results = pd.DataFrame([
    {"variant": "A_raw", **dict(zip(["clusters","noise","silhouette"], run_dbscan(X_scaled)))},
    {"variant": "B_no_outliers", **dict(zip(["clusters","noise","silhouette"], run_dbscan(X_no_outliers)))},
    {"variant": "C_tSNE", **dict(zip(["clusters","noise","silhouette"], run_dbscan(tsne_2d)))},
    {"variant": "D_tSNE_no_outliers", **dict(zip(["clusters","noise","silhouette"], run_dbscan(tsne_2d[mask_no_outliers])))},
])
print("\n--- Papildoma analizė (8 sekcija) ---")
print(results)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines # Reikalinga legendoms
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sys

# --- 1. Duomenų įkėlimas ir paruošimas ---

# Čia įrašomas failo pavadinimas
file_path = "tsne_projekcija_upd.csv"

try:
    df = pd.read_csv(file_path, sep=';')
except FileNotFoundError:
    print(f"Failas '{file_path}' nerastas.")
    sys.exit()

df_clean = df.dropna()

y = None 
if 'label' in df_clean.columns:
    y = df_clean['label'].astype(str).values 
    X = df_clean.drop('label', axis=1)
    print("Stulpelis 'label' rastas ir bus naudojamas kombinuotai vizualizacijai.")
else:
    X = df_clean.copy()
    print("Stulpelis 'label'  nerastas. Bus kuriama tik DBSCAN vizualizacija.")

print(f"Duomenys '{file_path}' sėkmingai įkelti. Požymių skaičius: {X.shape[1]}, Eilučių skaičius: {X.shape[0]}")

# --- 2. Duomenų normavimas ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Optimalaus 'eps' nustatymas ---
try:
    min_samples_value = int(input("Įveskite 'min_samples' vertę: "))
except ValueError:
    print("Netinkama įvestis. Naudojama numatytoji vertė 5.")
    min_samples_value = 5

k = min_samples_value 
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
k_distances = np.sort(distances[:, k-1], axis=0)

print("\n--- 'eps' nustatymas ---")
print("Rodomas k-atstumo grafikas. Raskite 'alkūnės' tašką.")

plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.title(f'k-atstumo grafikas (k={min_samples_value})')
plt.xlabel('Taškai (surūšiuoti pagal atstumą)')
plt.ylabel(f'{min_samples_value}-asis artimiausias atstumas (būsimas "eps")')
plt.grid(True)
plt.figtext(0.5, 0.01, 
            "Išnagrinėkite šį grafiką. Optimali 'eps' vertė yra ta, kur kreivė pradeda staigiai kilti (prasideda alkūnė).\n", 
            ha="center", fontsize=10, style='italic')
plt.show()

try:
    eps_value = float(input("Įveskite 'eps' vertę, kurią matėte grafiko alkūnėje: "))
except ValueError:
    print("Netinkama įvestis. Naudojama numatytoji vertė 5.0")
    eps_value = 5.0

# --- 4. DBSCAN klasterizavimas ---
print(f"\nVykdomas DBSCAN su eps={eps_value} ir min_samples={min_samples_value}...")
db = DBSCAN(eps=eps_value, min_samples=min_samples_value)
clusters = db.fit_predict(X_scaled)
# Konvertuojame klasterius į 'str'
clusters_str = [str(c) for c in clusters]

# --- 5. Rezultatų analizė ---
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)
total_points = len(clusters)
noise_percentage = (n_noise_ / total_points) * 100

print("\n--- KLASTERIZAVIMO REZULTATAI ---")
print(f"Bendras rastų klasterių skaičius (be triukšmo): {n_clusters_}")
print(f"Atsiskyrėlių ('-1') dalis: {noise_percentage:.2f}%")

print("\nKlasterių dydžiai (narių skaičius):")
cluster_counts = pd.Series(clusters).value_counts().sort_index()
for cluster_label, count in cluster_counts.items():
    if cluster_label == -1:
        print(f"  Klasteris -1 (Triukšmas): {count} narių")
    else:
        print(f"  Klasteris {cluster_label}: {count} narių")

if n_clusters_ > 1:
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"Vidutinis silueto balas: {silhouette_avg:.4f}")
else:
    print("Vidutinis silueto balas: N/A (Nerasta klasterių arba rastas tik vienas)")

# --- 6. t-SNE transformacija ---
print("\nVykdoma t-SNE transformacija vizualizacijai...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1500)
X_tsne = tsne.fit_transform(X_scaled)

# Paruošiame DataFrame vizualizacijoms
df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
df_tsne['dbscan_cluster'] = clusters_str
if y is not None:
    df_tsne['true_label'] = y


# --- 7. Kombinuota vizualizacija ---
print("Braižomas kombinuotas t-SNE grafikas...")

# Naudojame didesnį plotą, kad tilptų legendos
fig, ax = plt.subplots(figsize=(14, 10))

# A. Paruošiame spalvas DBSCAN klasterių užpildymui (fill)
db_clusters_unique = sorted(df_tsne['dbscan_cluster'].unique())
db_palette_list = sns.color_palette("hsv", n_colors=n_clusters_)
fill_map = {}
color_idx = 0
for c in db_clusters_unique:
    if c == '-1':
        fill_map['-1'] = (0.7, 0.7, 0.7) # Pilka triukšmui
    else:
        fill_map[c] = db_palette_list[color_idx]
        color_idx += 1
fill_colors = df_tsne['dbscan_cluster'].map(fill_map)

# B. Paruošiame spalvas 'label' KRAŠTELIAMS (edge)
# Tikriname, ar 'y' egzistuoja
if y is not None:
    labels_unique = sorted(df_tsne['true_label'].unique())
    n_labels = len(labels_unique)
    edge_palette_list = sns.color_palette("tab10", n_colors=n_labels) 
    edge_map = {label: color for label, color in zip(labels_unique, edge_palette_list)}
    edge_colors = df_tsne['true_label'].map(edge_map)
    edge_width = 1.5
else:
    # Jei 'label' nėra, kraštelių nebus
    edge_colors = 'none' 
    edge_width = 0

# C. Braižome grafiką
ax.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], 
           c=fill_colors,          # Užpildymo spalva = DBSCAN klasteris
           edgecolors=edge_colors, # Kraštelio spalva = Tikra klasė ('label')
           s=60,                   # Taško dydis
           linewidths=edge_width,  # Kraštelio plotis
           alpha=0.8)

# D. Kuriame rankines legendas
legend_elements_dbscan = []
for label, color in fill_map.items():
    legend_elements_dbscan.append(mlines.Line2D([0], [0], marker='o', color='none', 
                                                label=f'Klasteris {label}',
                                                markerfacecolor=color, markersize=10))
# Pirmoji legenda (DBSCAN)
legend1 = ax.legend(handles=legend_elements_dbscan, title="DBSCAN klasteriai", 
                    loc='upper left', bbox_to_anchor=(1.00, 1))
ax.add_artist(legend1)

if y is not None:
    legend_elements_label = []
    for label, color in edge_map.items():
        legend_elements_label.append(mlines.Line2D([0], [0], marker='o', color='none', 
                                                   label=f'Klasė {label}',
                                                   markerfacecolor='none', 
                                                   markeredgecolor=color, 
                                                   markeredgewidth=edge_width, 
                                                   markersize=10))
    # Antroji legenda (Tikros klasės)
    ax.legend(handles=legend_elements_label, title="Klasės (Label)", 
              loc='upper left', bbox_to_anchor=(1.00, 0.5))

# E. Grafiko nustatymai
plot_title = f'{file_path.split('.')[0]} (eps={eps_value}, min={min_samples_value})'
ax.set_title(plot_title)
ax.set_xlabel('t-SNE komponentas 1')
ax.set_ylabel('t-SNE komponentas 2')
ax.grid(True)

# Padarome vietos legendoms
fig.tight_layout(rect=[0, 0, 0.85, 1]) 

# F. Išsaugome
plot_filename = f"tsne_vizualizacija_{file_path.split('.')[0]}_eps{eps_value}_min{min_samples_value}.png"
try:
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nVizualizacija sėkmingai išsaugota kaip: '{plot_filename}'")
except Exception as e:
    print(f"\nKLAIDA: Nepavyko išsaugoti vizualizacijos. {e}")

plt.show()

# --- 8. Išskirčių pašalinimo ir dimensijos mažinimo įtaka ---
# Išskirčių pašalinimas naudojant bazinį DBSCAN
# db_base = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(X_scaled)
mask_no_outliers = db.labels_ != -1
X_no_outliers = X_scaled[mask_no_outliers]

tsne_2d = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1500).fit_transform(X_scaled)

# Funkcija rezultatams gauti
def run_dbscan(X, eps=eps_value, min_samples=min_samples_value):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels)
    
    if n_clusters > 1:
        sil = silhouette_score(X, labels, metric='euclidean')
    else:
        sil = np.nan
    return n_clusters, noise_ratio, sil

# Apskaičiuojame visus 4 variantus
results = pd.DataFrame([
    {"variant": "A_raw", **dict(zip(["clusters","noise","silhouette"], run_dbscan(X_scaled)))},
    {"variant": "B_no_outliers", **dict(zip(["clusters","noise","silhouette"], run_dbscan(X_no_outliers)))},
    {"variant": "C_tSNE", **dict(zip(["clusters","noise","silhouette"], run_dbscan(tsne_2d)))},
    {"variant": "D_tSNE_no_outliers", **dict(zip(["clusters","noise","silhouette"], run_dbscan(tsne_2d[mask_no_outliers])))},
])
print(results)