import optuna
import configparser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse, os
import metrics as me
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Auto tune parameters of a DeepFilterNet model")
parser.add_argument(
    "--model_path",
    type=str,
    help="path/to/model.pt",
    required=True,
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    help="path/to/dataset.hdf5",
    required=True,
)
parser.add_argument(
    "--nb_samples",
    type=int,
    help="Number of samples to evaluate",
    required=True,
)
args_t = parser.parse_args()

config = configparser.ConfigParser()

def compute_loss_with_bonus(
    pesq: float,
    stoi: float,
    gerb: float,
    gdf: float,
    gspeech: float,
    pesq_ref: float = 2.83,
    stoi_ref: float = 0.9360,
    weight_gate: float = 1.0,
    bonus_weight_pesq: float = 0.5,  # Poids du gain quand PESQ > 2.56
    bonus_weight_stoi: float = 2.0,  # Poids du gain quand STOI > 0.9146
) -> float:
    # 1. Deltas (positif = dégradation, négatif = amélioration)
    delta_pesq = pesq_ref - pesq
    delta_stoi = stoi_ref - stoi

    # 2. Perte / Récompense PESQ
    if delta_pesq < 0:
        # Gain linéaire si PESQ > 2.56 (diminue la loss globale)
        loss_pesq = bonus_weight_pesq * delta_pesq  # Valeur négative
    else:
        # Pénalité douce au début + pénalité forte au-delà de 2.4 (delta > 0.16)
        loss_pesq = 10.0 * (delta_pesq ** 2) + 500.0 * (max(0.0, delta_pesq - 0.16) ** 3)

    # 3. Perte / Récompense STOI
    if delta_stoi < 0:
        # Gain linéaire si STOI > 0.9146
        loss_stoi = bonus_weight_stoi * delta_stoi  # Valeur négative
    else:
        # Pénalité douce au début + pénalité forte sous 0.89 (delta > 0.025)
        loss_stoi = 15.0 * ((delta_stoi / 0.025) ** 2) + 1000.0 * (max(0.0, delta_stoi - 0.025) ** 3)

    # 4. Coût des portes (toujours positif)
    avg_gate = (gerb + gdf + gspeech) / 3.0
    loss_gates = weight_gate * avg_gate

    print(f"pesq_diff: {loss_pesq}, stoi_diff: {loss_stoi}, gate_vad_mean: {gerb}, gate_erb_mean: {gdf}, gate_df_mean: {gspeech}")

    return loss_pesq + loss_stoi + loss_gates


def Loss(snr_df, df_stats):
    print(snr_df)
    pesq_no_gate = 2.569759
    stoi_no_gate = 0.914639
    pesq_diff = (snr_df["PESQ"] - pesq_no_gate - 0.5)**5
    stoi_diff = ((snr_df["STOI"] - stoi_no_gate )/0.089)**5
    # if pesq_diff <= -1:
    #     pesq_diff = pesq_diff**3
    # if stoi_diff <= -1:
    #     stoi_diff = stoi_diff**3
    # print(snr_df)
    # print(f"pesq_diff: {pesq_diff}, stoi_diff: {stoi_diff}, gate_vad_mean: {df_stats['gate_vad_mean']}, gate_erb_mean: {df_stats['gate_erb_mean']}, gate_df_mean: {df_stats['gate_df_mean']}")
    return compute_loss_with_bonus(snr_df["PESQ"], snr_df["STOI"], df_stats["gate_vad_mean"], df_stats["gate_erb_mean"], df_stats["gate_df_mean"])

    #res = snr_df["PESQ"] * 3 + snr_df["STOI"] * 3 + df_stats["bypass_rate"] + df_stats["erb_only_rate"]
    #return pesq_diff + stoi_diff - df_stats["gate_vad_mean"] - df_stats["gate_erb_mean"] - df_stats["gate_df_mean"]
def edit_and_run(gate_snr_threshold, gate_vad_threshold, gate_temperature_start, gate_stationarity_threshold, smoothing_window_size, smoothing_alpha_attack, smoothing_alpha_release, DurationFilter_vad_1, DurationFilter_vad_2, DurationFilter_noise_1, DurationFilter_noise_2, DurationFilter_stationarity_1, DurationFilter_stationarity_2):
    # Read config file
    config.read(os.path.join(args_t.model_path, "config.ini"))
    # Edit parameters
    config["deepfilternet"]["gate_snr_threshold"] = str(gate_snr_threshold)
    config["deepfilternet"]["gate_vad_threshold"] = str(gate_vad_threshold)
    #config["deepfilternet"]["gate_temperature_start"] = str(gate_temperature_start)
    config["deepfilternet"]["gate_stationarity_threshold"] = str(gate_stationarity_threshold)
    # config["Smoothing"]["smoothing_window_size"] = str(smoothing_window_size)
    # config["Smoothing"]["smoothing_alpha_attack"] = str(smoothing_alpha_attack)
    # config["Smoothing"]["smoothing_alpha_release"] = str(smoothing_alpha_release)
    config["durationfilter"]["vad"] = str(DurationFilter_vad_1)+","+str(DurationFilter_vad_2)
    config["durationfilter"]["noise"] = str(DurationFilter_noise_1)+","+str(DurationFilter_noise_2)
    config["durationfilter"]["stationarity"] = str(DurationFilter_stationarity_1)+","+str(DurationFilter_stationarity_2)
    config["deepfilternet"]["symbolic_gate"] = str("True")
    # Save config file
    with open(os.path.join(args_t.model_path, "config.ini"), mode="w", encoding="utf-8") as configfile:
        config.write(configfile)
    # Run training
    snr_df, df_stats = me.main(ex=True, model_dir=args_t.model_path, dataset_dir=args_t.dataset_dir, nb=args_t.nb_samples)
    snr_df = snr_df.mean()
    df_stats = df_stats.mean()
    return Loss(snr_df, df_stats)

#edit_and_run(15, 0.01, 1, 0.01)
#quit()
def objective(trial):
    # Define search space
    gate_snr_threshold = trial.suggest_float('gate_snr_threshold', 15, 25, step=2.5)
    gate_vad_threshold = trial.suggest_float('gate_vad_threshold', 0.01, 0.2, step=0.005)
    #gate_temperature_start = trial.suggest_float('gate_temperature_start', 0.1, 5, step=0.05)
    gate_temperature_start = 0
    gate_stationarity_threshold = trial.suggest_float('gate_stationarity_threshold', 0.01, 0.3, step=0.005)
    smoothing_window_size = 0#trial.suggest_int('smoothing_window_size', 1, 30, step=2)
    smoothing_alpha_attack = 0#trial.suggest_float('smoothing_alpha_attack', 0, 1, step=0.1)
    smoothing_alpha_release = 0#trial.suggest_float('smoothing_alpha_release', 0, 1, step=0.1)
    DurationFilter_vad_1 = 1#trial.suggest_int('DurationFilter_vad_1', 0, 20, step=2)
    DurationFilter_vad_2 = trial.suggest_int('DurationFilter_vad_2', 1, 15, step=1)
    DurationFilter_noise_1 = 1#trial.suggest_int('DurationFilter_noise_1', 0, 20, step=2)
    DurationFilter_noise_2 = trial.suggest_int('DurationFilter_noise_2', 1, 15, step=1)
    DurationFilter_stationarity_1 = 1#trial.suggest_int('DurationFilter_stationarity_1', 0, 20, step=2)
    DurationFilter_stationarity_2 = trial.suggest_int('DurationFilter_stationarity_2', 1, 15, step=1)
    
    # Train and evaluate model
    return edit_and_run(gate_snr_threshold, gate_vad_threshold, gate_temperature_start, gate_stationarity_threshold, smoothing_window_size, smoothing_alpha_attack, smoothing_alpha_release, DurationFilter_vad_1, DurationFilter_vad_2, DurationFilter_noise_1, DurationFilter_noise_2, DurationFilter_stationarity_1, DurationFilter_stationarity_2)


# Define the database file name and study name
db_name = "optuna_study.db"
study_name = "audio_gating_new_vad_loss_and_gate_mecha_no_hloss1p7"
storage_url = f"sqlite:///{db_name}"
if os.path.exists(db_name):
    print(f"\033[93mFound existing checkpoint database '{db_name}'. Resuming study... \033[0m")
else:
    print(f"\033[94mNo checkpoint found. Creating a new study... \033[0m")
# Run optimization study
study = optuna.create_study(direction='minimize', storage=storage_url, study_name=study_name, load_if_exists=True)
study.optimize(objective, n_trials=100)

print("Best parameters: ", study.best_params)
config.read(os.path.join(args_t.model_path, "config.ini"))
# Edit parameters
config["deepfilternet"]["gate_snr_threshold"] = str(study.best_params["gate_snr_threshold"])
config["deepfilternet"]["gate_vad_threshold"] = str(study.best_params["gate_vad_threshold"])
#config["deepfilternet"]["gate_temperature_start"] = str(study.best_params["gate_temperature_start"])
config["deepfilternet"]["gate_stationarity_threshold"] = str(study.best_params["gate_stationarity_threshold"])
# config["Smoothing"]["smoothing_window_size"] = str(study.best_params["smoothing_window_size"])
# config["Smoothing"]["smoothing_alpha_attack"] = str(study.best_params["smoothing_alpha_attack"])
# config["Smoothing"]["smoothing_alpha_release"] = str(study.best_params["smoothing_alpha_release"])
# config["DurationFilter"]["vad"] = str(study.best_params["DurationFilter_vad_1"])+","+str(study.best_params["DurationFilter_vad_2"])
# config["DurationFilter"]["noise"] = str(study.best_params["DurationFilter_noise_1"])+","+str(study.best_params["DurationFilter_noise_2"])
# config["DurationFilter"]["stationarity"] = str(study.best_params["DurationFilter_stationarity_1"])+","+str(study.best_params["DurationFilter_stationarity_2"])
config["durationfilter"]["vad"] = "1"+","+str(study.best_params["DurationFilter_vad_2"])
config["durationfilter"]["noise"] = "1"+","+str(study.best_params["DurationFilter_noise_2"])
config["durationfilter"]["stationarity"] = "1"+","+str(study.best_params["DurationFilter_stationarity_2"])
# Save config file
with open(os.path.join(args_t.model_path, "config.ini"), mode="w", encoding="utf-8") as configfile:
    config.write(configfile)