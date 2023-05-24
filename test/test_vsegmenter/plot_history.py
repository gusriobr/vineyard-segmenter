import os

import pandas as pd
import matplotlib.pyplot as plt
## Lanzando con dtaset v6
# Cargar los datos desde los archivos CSV
import cfg

df1 = pd.read_csv('/workspaces/wml/vineyard-segmenter/results/tmp/unet_7/2023-05-11T07-30_20/history.csv')
df2 = pd.read_csv('/workspaces/wml/vineyard-segmenter/results/tmp/unet_7/2023-05-11T09-38_15/history.csv')
df3 = pd.read_csv('/workspaces/wml/vineyard-segmenter/results/tmp/unet_7/2023-05-12T14-01_05/history.csv')
df4 = pd.read_csv('/workspaces/wml/vineyard-segmenter/results/tmp/unet_7/2023-05-12T15-13_45/history.csv')
# df3 = pd.read_csv('/workspaces/wml/vineyard-segmenter/results/tmp/unet_7/2023-05-11T13-29_15/history.csv')
# df4 = pd.read_csv('/workspaces/wml/vineyard-segmenter/results/tmp/unet_7/2023-05-11T15-54_17/history.csv')
# df5 = pd.read_csv('/workspaces/wml/vineyard-segmenter/results/tmp/unet_7/2023-05-11T22-32_15/history.csv')

# Filtrar las columnas que no comienzan con "val_"

# Crear las gráficas
fig, axs = plt.subplots(len(df1.columns) - 1, figsize=(10, 40))
fig.suptitle('Comparación de variables', fontsize=16)


def plot_serie(df, name):
    columns = [c for c in df.columns if c != "epoch"]
    for i, col in enumerate(columns):
        axs[i].plot(df['epoch'], df[col], label=name)
        axs[i].set_xlabel('epoch')
        axs[i].set_ylabel(col)
        axs[i].legend()


def plot_model_history(file_name, name, chart_type="line", alpha=0.5):
    file_path = file_name if os.path.isabs(file_name) else os.path.join(cfg.results('tmp'), f'{file_name}/history.csv')
    df = pd.read_csv(file_path)
    columns = [c for c in df.columns if c != "epoch"]
    for i, col in enumerate(columns):
        if chart_type == "line":
            axs[i].plot(df['epoch'], df[col], label=name, alpha=alpha)
        elif chart_type == "point":
            axs[i].scatter(df['epoch'], df[col], label=name, marker='.', alpha=alpha)  # Ajusta el valor de alpha
        else:
            raise ValueError(f"Invalid chart type: {chart_type}")
        axs[i].set_xlabel('epoch')
        axs[i].set_ylabel(col)
        axs[i].legend()


def compare_models(folder_names: list, chart_type="line", alpha=0.5):
    for folder_name in folder_names:
        folder_path = cfg.results(f"tmp/{folder_name}")
        # get
        dirs = [os.path.join(folder_path, item) for item in os.listdir(folder_path) if
                os.path.isdir(os.path.join(folder_path, item))]
        history_folder = dirs[0]
        history_file = os.path.join(history_folder, "history.csv")
        plot_model_history(history_file, folder_name, chart_type=chart_type, alpha=alpha)


compare_models(['unet_7_exp_0.001_7',
                'unet_7_exp_0.0001_7',
                # 'unet_7_exp_0.05_7',
                'unet_7_exp_0.0005_7',
                'unet_7_exp_1e-05_7',
                'unet_7_exp_5e-05_7'], chart_type="point", alpha=0.35)

# Graficar las variables que no comienzan con "val_"

# plot_model_history('unet_all_7/2023-05-17T08-22_35', 'all_1k')
# plot_model_history('unet_green_7/2023-05-17T07-50_16', 'green_1k')
# plot_model_history('unet_noise_7/2023-05-17T07-17_42', 'noise_1k')
# plot_model_history('unet_simple_7/2023-05-17T06-46_04', 'simple_1K')
# plot_model_history('unet_7/2023-05-12T14-01_05', 'noaug')

# plot_model_history('unet_all_7/2023-05-18T00-33_07', 'all_2k')
# plot_model_history('unet_green_7/2023-05-17T23-02_46', 'green_2k')
# plot_model_history('unet_noise_7/2023-05-17T21-32_00', 'noise_2k')
# plot_model_history('unet_simple_7/2023-05-17T20-03_01', 'simple_2k')

# plot_model_history('unet_all_7/2023-05-17T08-22_35', 'all_1k')
# plot_model_history('unet_all_7/2023-05-18T00-33_07', 'all_2k')
# plot_model_history('unet_all_7/2023-05-18T07-39_01', 'all_3k')
# plot_model_history('unet_7/2023-05-12T14-01_05', 'noaug')
#
# plot_model_history('unet_all_7/2023-05-17T08-22_35', 'all_1k')
# plot_model_history('unet_all_7/2023-05-18T00-33_07', 'all_2k')
# plot_model_history('unet_all_7/2023-05-18T07-39_01', 'all_3k')
# plot_model_history('unet_7/2023-05-12T14-01_05', 'noaug')

# plot_serie(df1, 'v6_sin_aug')
# plot_serie(df2, 'v6_con_aug')
# plot_serie(df3, 'v7_sin_aug')
# plot_serie(df4, 'v7_con_aug')
# plot_serie(df5, 'v7_con_aug_full')
#
# plot_model_history('unet_all_7/2023-05-18T22-18_47', 'all_1k')
# plot_model_history('unet_all_7/2023-05-18T23-07_08', 'all_2k')
# plot_model_history('unet_all_7/2023-05-18T23-55_43', 'all_3k')
# plot_model_history('unet_all_7/2023-05-19T00-44_15', 'all')
# plot_model_history('unet_7/2023-05-12T14-01_05', 'noaug')

# Graficar las variables que comienzan con "va
plt.tight_layout()
plt.show()
