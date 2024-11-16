DATA_DICT = {
    "ettm2": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "Dataset/ETTm2.csv",
        "freq": "15min",
    },
    "ettm1": {
        # "boundaries": [960, 46080, 57600],
        "boundaries": [34560, 46080, 57600],
        "data_path": "Dataset/ETTm1.csv",
        "freq": "15min",
    },
    "etth2": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "Dataset/ETTh2.csv",
        "freq": "H",
    },
    "etth1": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "Dataset/ETTh1.csv",
        "freq": "H",
    },
    "elec": {
        "boundaries": [18413, 21044, 26304],
        "data_path": "Dataset/electricity.csv",
        "freq": "H",
    },
    "traffic": {
        "boundaries": [12280, 14036, 17544],
        "data_path": "Dataset/traffic.csv",
        "freq": "H",
    },
    "weather": {
        "boundaries": [36887, 42157, 52696],
        "data_path": "Dataset/weather.csv",
        "freq": "10min",
    },
    "bitcoin": {
        "boundaries": [1920, 2664, 2760],
        "data_path": "Dataset/BTC_USD_Price_Prediction_Data.csv",
        "freq": "D",
    },
}

import timesfm
import pandas as pd
import numpy as np
from timesfm import data_loader
from tqdm import tqdm

dataset = "bitcoin"
data_path = DATA_DICT[dataset]["data_path"]
freq = DATA_DICT[dataset]["freq"]
int_freq = timesfm.freq_map(freq)
boundaries = DATA_DICT[dataset]["boundaries"]

data_df = pd.read_csv(open(data_path, "r"))

ts_cols = [col for col in data_df.columns if col != "date"]
num_cov_cols = None
cat_cov_cols = None

context_len = 512
pred_len = 96

num_ts = len(ts_cols)
batch_size = 16

dtl = data_loader.TimeSeriesdata(
      data_path=data_path,
      datetime_col="date",
      num_cov_cols=num_cov_cols,
      cat_cov_cols=cat_cov_cols,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundaries[0]],
      val_range=[boundaries[0], boundaries[1]],
      test_range=[boundaries[1], boundaries[2]],
      hist_len=context_len,
      pred_len=pred_len,
      batch_size=num_ts,
      freq=freq,
      normalize=True,
      epoch_len=None,
      holiday=False,
      permute=True,
  )

display(data_df.head())

# モデル定義
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          context_len=512,
          input_patch_len=32,
          output_patch_len=128,
          num_layers=20,
          model_dims=1280,
          # per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"), # timesfm-1.0-200m は最初のオープンモデルのチェックポイント
  )

train_batches = dtl.tf_dataset(mode="train", shift=1).batch(batch_size)
val_batches = dtl.tf_dataset(mode="val", shift=pred_len)
test_batches = dtl.tf_dataset(mode="test", shift=pred_len)

for tbatch in tqdm(train_batches.as_numpy_iterator()):
    pass
print(tbatch[0].shape) # batch size, column size (列数), data size (行数)

# 結果可視化
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 各バッチの実測値と予測値を格納するリスト
actuals_list = []
forecasts_list = []
history_list = []

# MAE損失の計算と実測値と予測値の収集
mae_losses = []
for batch in tqdm(val_batches.as_numpy_iterator()):
    with torch.no_grad():
        past = batch[0]  # 履歴データ（train部分）
        actuals = batch[3][0]  # 実測データ（test部分） # 0は予測対象の値の位置を示している

        # 予測を実行
        mean_outputs, full_outputs = tfm.forecast(list(past), [0] * past.shape[0])
        
        # 実測値と予測値の対応する部分を取り出す
        forecasts = mean_outputs[0, 0:actuals.shape[0]] # 必要な範囲の予測を取り出し # 1要素目の0は予測対象の値の位置を示している
        mae_losses.append(np.abs(forecasts - actuals).mean())
        
        # 実測値、予測値、履歴データをリストに追加
        actuals_list.append(actuals)
        forecasts_list.append(forecasts)
        history_list.append(past)
  
print(f"MAE: {np.mean(mae_losses)}")

# 最後のバッチを使用して可視化（必要に応じて他のバッチも可視化可能）
history = history_list[-1]  # 過去データ
actuals = actuals_list[-1]  # 実測データ
forecasts = forecasts_list[-1]  # 予測データ

plt.figure(figsize=(12, 6))

# 履歴部分のプロット（青色の線）
plt.plot(range(history.shape[1]), history[0, :], label='History', color='blue')

# 実測値部分のプロット（青色の破線）
offset = history.shape[1]
plt.plot(range(offset, offset + actuals.shape[1]), actuals[0, :], label='Actuals', color='blue', linestyle='--')

# 予測値部分のプロット（赤色の破線）
plt.plot(range(offset, offset + forecasts.shape[1]), forecasts[0, :], label='Forecasts', color='red', linestyle='--')

# プロットの設定
plt.title('History and Forecast', fontsize=16)
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)

# プロット表示
plt.show()
