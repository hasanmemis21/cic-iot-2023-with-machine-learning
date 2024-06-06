import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Sınıflandırma etiketlerini tanımlayalım
labels = np.array([
    'BenignTraffic', 'DDoS-ACK_Fragmentation', 'DDoS-HTTP_Flood', 'DDoS-ICMP_Flood', 'DDoS-SYN_Flood',
    'DDoS-TCP_Flood', 'DDoS-UDP_Flood', 'DoS-ACK_Fragmentation', 'DoS-HTTP_Flood', 'DoS-ICMP_Flood',
    'DoS-SYN_Flood', 'DoS-TCP_Flood', 'DoS-UDP_Flood', 'Mirai-ACK_Flood', 'Mirai-greip_flood',
    'Mirai-greeth_flood', 'Mirai-greup_flood', 'Mirai-ICMP_Flood', 'Mirai-PSHACK_Flood', 'Mirai-SYN_Flood',
    'Mirai-UDP_Flood', 'Recon-HostDiscovery', 'Recon-PingSweep', 'Recon-PortScan', 'Recon-OSSscan',
    'Recon-PingSweep2', 'Recon-PortScan2', 'Recon-OSSscan2', 'Recon-PingSweep3', 'Recon-PortScan3',
    'Recon-OSSscan3', 'Recon-PingSweep4', 'Recon-OSSscan4'
])

# Veri yükleme ve işleme fonksiyonları
def load_data(data_folder):
    all_files = os.listdir(data_folder)
    selected_files = np.random.choice(all_files, 42, replace=False)
    data_list = [pd.read_csv(os.path.join(data_folder, file)) for file in selected_files]
    data = pd.concat(data_list, ignore_index=True)
    return data

# Eksik verileri doldurma
def fill_missing_values(data):
    return data.fillna(data.mean())

# Verileri normalleştirme
def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

# Modeli eğitmek için fonksiyon
def train_model(model, X_train, y_train):
    ovr = OneVsRestClassifier(model)
    ovr.fit(X_train, y_train)
    return ovr

# Modeli eğitmek için verileri yükleme ve işleme
data_folder = 'dataset'
data = load_data(data_folder)
features = data.iloc[:, :46]
target = data.iloc[:, 46]

# Sütun adlarını kontrol edip uyumlu hale getirme
features.columns = [col.strip() for col in features.columns]
target.name = 'label'

# Eksik verileri doldurma
features_filled = fill_missing_values(features)

# Verileri normalleştirme
features_normalized = normalize_data(features_filled)

# Etiketleri binarize etme
y_bin = label_binarize(target, classes=labels)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(features_normalized, y_bin, test_size=0.2, random_state=42)

# Modelleri eğitme ve değerlendirme
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

accuracy_results = {}
for model_name, model in models.items():
    trained_model = train_model(model, X_train, y_train)
    y_pred = trained_model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    accuracy_results[model_name] = accuracy
    print(f"{model_name} Accuracy:", accuracy)
    joblib.dump(trained_model, f'{model_name.replace(" ", "_")}_model.pkl')

# Eğitim sonuçlarını kaydetme
accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Model', 'Accuracy'])
accuracy_df.to_csv('/mnt/data/model_accuracies.csv', index=False)
