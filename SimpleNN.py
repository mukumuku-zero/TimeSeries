import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# データの準備 (時系列の順序を保持)
def prepare_time_series_data(df, target_column, train_ratio=0.7, val_ratio=0.15):
    # 日付と目的変数のカラムを除いた説明変数
    X = df.drop(columns=[target_column, 'date']).values  
    y = df[target_column].values  # 目的変数
    
    # データの標準化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # 順序を保ったままトレーニング、検証、テストデータに分割
    train_size = int(len(X_scaled) * train_ratio)
    val_size = int(len(X_scaled) * val_ratio)
    
    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]
    
    X_val = X_scaled[train_size:train_size + val_size]
    y_val = y_scaled[train_size:train_size + val_size]
    
    X_test = X_scaled[train_size + val_size:]
    y_test = y_scaled[train_size + val_size:]

    # NumPyからPyTorchのTensorに変換
    X_train, X_val, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_val, X_test])
    y_train, y_val, y_test = map(lambda y: torch.tensor(y, dtype=torch.float32), [y_train, y_val, y_test])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_y

# 単純なニューラルネットワークモデルの構築
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 学習プロセス (ベストモデルの保存)
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001, weight_decay=0.01, model_path='best_model.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf') 
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 検証データでの評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_y).item()

        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path) 
            print(f'Best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
    
    print("Training complete")

# 予測プロセス (モデル読み込み)
def predict(model, X_test, model_path='best_model.pth'):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    return predictions

def run_pipeline(df, target_column, epochs=50, batch_size=32):
    # データ準備
    X_train, X_test, y_train, y_test, scaler_y = prepare_data(df, target_column)
    
    # モデル構築
    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim)

    # 学習
    train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    predictions = predict(model, X_test)
    predictions = scaler_y.inverse_transform(predictions.numpy())
    
    return predictions, model
