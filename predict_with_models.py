import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize

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

# Eksik verileri doldurma
def fill_missing_values(data):
    return data.fillna(data.mean())

# Verileri normalleştirme
def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

# Yeni verilerle saldırı türünü tahmin etmek için fonksiyon
def predict_attack_type(model, input_data):
    prediction = model.predict(input_data)
    predicted_label = labels[np.argmax(prediction, axis=1)]
    return predicted_label

# Örnek veri setleri (4 tahmin verisi)
input_data_1 = pd.DataFrame([{
    'flow_duration': 0, 'Header_Length': 54, 'Protocol Type': 6, 'Duration': 64, 'Rate': 0.329807, 'Srate': 0.329807, 'Drate': 0,
    'fin_flag_number': 0, 'syn_flag_number': 1, 'rst_flag_number': 0, 'psh_flag_number': 1, 'ack_flag_number': 0, 'ece_flag_number': 0, 'cwr_flag_number': 0,
    'ack_count': 0, 'syn_count': 0, 'fin_count': 0, 'urg_count': 0, 'rst_count': 0, 'HTTP': 0, 'HTTPS': 0, 'DNS': 0, 'Telnet': 0, 'SMTP': 0, 'SSH': 0, 'IRC': 0,
    'TCP': 0, 'UDP': 1, 'DHCP': 0, 'ARP': 0, 'ICMP': 0, 'IPv': 0, 'LLC': 1, 'Tot sum': 567, 'Min': 54, 'Max': 54, 'AVG': 54, 'Std': 0, 'Tot size': 54, 'IAT': 83343832,
    'Number': 9.5, 'Magnitue': 10.3923, 'Radius': 0, 'Covariance': 0, 'Variance': 0, 'Weight': 141.55
}])

input_data_2 = pd.DataFrame([{
    'flow_duration': 0.328175, 'Header_Length': 76175, 'Protocol Type': 17, 'Duration': 64, 'Rate': 4642.133, 'Srate': 4642.133, 'Drate': 0,
    'fin_flag_number': 0, 'syn_flag_number': 0, 'rst_flag_number': 0, 'psh_flag_number': 0, 'ack_flag_number': 0, 'ece_flag_number': 0, 'cwr_flag_number': 0,
    'ack_count': 0, 'syn_count': 0, 'fin_count': 0, 'urg_count': 0, 'rst_count': 0, 'HTTP': 0, 'HTTPS': 0, 'DNS': 0, 'Telnet': 0, 'SMTP': 0, 'SSH': 0, 'IRC': 0,
    'TCP': 0, 'UDP': 1, 'DHCP': 0, 'ARP': 0, 'ICMP': 0, 'IPv': 0, 'LLC': 1, 'Tot sum': 525, 'Min': 50, 'Max': 50, 'AVG': 50, 'Std': 0, 'Tot size': 50, 'IAT': 83015696,
    'Number': 9.5, 'Magnitue': 10, 'Radius': 0, 'Covariance': 0, 'Variance': 0, 'Weight': 141.55
}])

input_data_3 = pd.DataFrame([{
    'flow_duration': 0, 'Header_Length': 0, 'Protocol Type': 1, 'Duration': 64, 'Rate': 33.3968, 'Srate': 33.3968, 'Drate': 0,
    'fin_flag_number': 0, 'syn_flag_number': 0, 'rst_flag_number': 0, 'psh_flag_number': 0, 'ack_flag_number': 0, 'ece_flag_number': 0, 'cwr_flag_number': 0,
    'ack_count': 0, 'syn_count': 0, 'fin_count': 0, 'urg_count': 0, 'rst_count': 0, 'HTTP': 0, 'HTTPS': 0, 'DNS': 1, 'Telnet': 0, 'SMTP': 0, 'SSH': 0, 'IRC': 0,
    'TCP': 0, 'UDP': 0, 'DHCP': 0, 'ARP': 0, 'ICMP': 0, 'IPv': 0, 'LLC': 1, 'Tot sum': 441, 'Min': 42, 'Max': 42, 'AVG': 42, 'Std': 0, 'Tot size': 42, 'IAT': 83127994,
    'Number': 9.5, 'Magnitue': 9.165151, 'Radius': 0, 'Covariance': 0, 'Variance': 0, 'Weight': 141.55
}])

input_data_4 = pd.DataFrame([{
    'flow_duration': 0, 'Header_Length': 57.04, 'Protocol Type': 6.33, 'Duration': 64, 'Rate': 4.290556, 'Srate': 4.290556, 'Drate': 0,
    'fin_flag_number': 0, 'syn_flag_number': 0, 'rst_flag_number': 0, 'psh_flag_number': 0, 'ack_flag_number': 0, 'ece_flag_number': 0, 'cwr_flag_number': 0,
    'ack_count': 0, 'syn_count': 0, 'fin_count': 0, 'urg_count': 0, 'rst_count': 0, 'HTTP': 0, 'HTTPS': 1, 'DNS': 0, 'Telnet': 0, 'SMTP': 0, 'SSH': 0, 'IRC': 0,
    'TCP': 1, 'UDP': 0, 'DHCP': 1, 'ARP': 0, 'ICMP': 0, 'IPv': 0, 'LLC': 1, 'Tot sum': 581.33, 'Min': 54, 'Max': 66.3, 'AVG': 54.7964, 'Std': 2.822973, 'Tot size': 57.04, 'IAT': 82926067,
    'Number': 9.5, 'Magnitue': 10.46467, 'Radius': 4.010353, 'Covariance': 160.9878, 'Variance': 0.05, 'Weight': 141.55
}])

input_data = pd.concat([input_data_1, input_data_2, input_data_3, input_data_4], ignore_index=True)

# Eksik verileri doldurma ve normalizasyon uygulama
input_data_filled = fill_missing_values(input_data)
input_data_normalized = normalize_data(input_data_filled)

# Modellerle tahmin yapma
models = {
    "Decision Tree": joblib.load('Decision_Tree_model.pkl'),
    "Logistic Regression": joblib.load('Logistic_Regression_model.pkl'),
    "Naive Bayes": joblib.load('Naive_Bayes_model.pkl'),
    "Random Forest": joblib.load('Random_Forest_model.pkl')
}

predictions = {}
for model_name, model in models.items():
    predictions[model_name] = predict_attack_type(model, input_data_normalized)

# Tahmin sonuçlarını kaydetme
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('/mnt/data/model_predictions.csv', index=False)
