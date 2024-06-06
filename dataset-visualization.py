import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns

# Klasördeki dosyaları listeleyin
data_path = 'veri'  # Veri klasörünüzün yolu
file_list = os.listdir(data_path)

# İlk 42 dosyayı seçin
selected_files = file_list[:42]

# Gerekli sütunları belirleyin
selected_columns = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
                    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
                    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
                    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP',
                    'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
                    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight', 'label']

# Veri çerçevelerini birleştirin
df_list = []
for file in selected_files:
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path, usecols=selected_columns)
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)

# Sayısal ve kategorik sütunları ayırın
numerical_columns = ['flow_duration', 'Header_Length', 'Duration', 'Rate', 'Srate', 'Drate',
                     'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
                     'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count',
                     'syn_count', 'fin_count', 'urg_count', 'rst_count', 'Tot sum', 'Min',
                     'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius',
                     'Covariance', 'Variance', 'Weight']

categorical_columns = ['Protocol Type', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
                       'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'label']

# Sayısal verileri impute et
numerical_imputer = SimpleImputer(strategy='mean')
data_numerical = pd.DataFrame(numerical_imputer.fit_transform(data[numerical_columns]),
                              columns=numerical_columns)

# Kategorik verileri impute et
categorical_imputer = SimpleImputer(strategy='most_frequent')
data_categorical = pd.DataFrame(categorical_imputer.fit_transform(data[categorical_columns]),
                                columns=categorical_columns)

# Normalizasyon
scaler = MinMaxScaler()
data_numerical_normalized = pd.DataFrame(scaler.fit_transform(data_numerical), columns=numerical_columns)

# Etiketleri (label) kodlayın
label_encoder = LabelEncoder()
data_categorical['label_encoded'] = label_encoder.fit_transform(data_categorical['label'])

# Normalleştirilmiş sayısal veriler ile kategorik verileri birleştirin
data_clean = pd.concat([data_numerical_normalized, data_categorical], axis=1)

# Temizlenmiş verilerin ilk birkaç satırını görüntüleyin
print(data_clean.head())

# Saldırı türlerini sayın
attack_counts = Counter(data_categorical['label_encoded'])

# Saldırı türlerini ve sayıları bir DataFrame'e dönüştürün
attack_df = pd.DataFrame(attack_counts.items(), columns=['label_encoded', 'Count'])

# Saldırı türlerini etiketlerden geri çevirin
attack_df['Attack Type'] = label_encoder.inverse_transform(attack_df['label_encoded'])

# Grafik oluşturma ve kaydetme
plt.figure(figsize=(14, 7))
plt.barh(attack_df['Attack Type'], attack_df['Count'], color='skyblue')
plt.xlabel('Count')
plt.ylabel('Attack Type')
plt.title('Distribution of Attack Types')
plt.grid(axis='x')

# Grafiği kaydet
output_path = 'grafikler/attack_distribution.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)

# Grafiği göster
plt.show()

# Protokol türüne göre saldırı sayısını hesaplayın
protocol_attack_counts = data_clean.groupby('Protocol Type')['label_encoded'].value_counts().unstack().fillna(0)

# Saldırı türlerini etiketlerden geri çevirin
protocol_attack_counts.columns = label_encoder.inverse_transform(protocol_attack_counts.columns.astype(int))

# Grafik oluşturma ve kaydetme
protocol_attack_counts.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.xlabel('Protocol Type')
plt.ylabel('Count')
plt.title('Attack Count by Protocol Type')
plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')

# Grafiği kaydet
output_path = 'grafikler/protocol_attack_distribution.png'
plt.savefig(output_path, bbox_inches='tight')

# Grafiği göster
plt.show()

# Flow Duration by Attack Type
plt.figure(figsize=(14, 7))
sns.boxplot(x='label_encoded', y='flow_duration', data=data_clean)
plt.title('Flow Duration by Attack Type')
plt.xticks(rotation=90)
plt.grid(axis='y')

# Grafiği kaydet
output_path = 'grafikler/flow_duration_by_attack_type.png'
plt.savefig(output_path)

# Grafiği göster
plt.show()

# Syn flag number by Attack Type
plt.figure(figsize=(14, 7))
sns.boxplot(x='label_encoded', y='syn_flag_number', data=data_clean)
plt.title('Syn Flag Number by Attack Type')
plt.xticks(rotation=90)
plt.grid(axis='y')

# Grafiği kaydet
output_path = 'grafikler/syn_flag_by_attack_type.png'
plt.savefig(output_path)

# Grafiği göster
plt.show()

# Korelasyon matrisi oluşturma
correlation_matrix = data_numerical.corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

# Korelasyon matrisini kaydetme
output_path = 'grafikler/correlation_matrix.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)

# Korelasyon matrisini gösterme
plt.show()

# Rate by Attack Type
plt.figure(figsize=(14, 7))
sns.boxplot(x='label_encoded', y='Rate', data=data_clean)
plt.title('Rate by Attack Type')
plt.xticks(rotation=90)
plt.grid(axis='y')

# Grafiği kaydet
output_path = 'grafikler/rate_by_attack_type.png'
plt.savefig(output_path)

# Grafiği göster
plt.show()

# Srate by Attack Type
plt.figure(figsize=(14, 7))
sns.boxplot(x='label_encoded', y='Srate', data=data_clean)
plt.title('Srate by Attack Type')
plt.xticks(rotation=90)
plt.grid(axis='y')

# Grafiği kaydet
output_path = 'grafikler/srate_by_attack_type.png'
plt.savefig(output_path)

# Grafiği göster
plt.show()

# Zaman Serisi Analizi (Eğer Zaman Verisi Varsa)
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # Zaman serisi analizi
    time_series_df = data.resample('M')['label_encoded'].value_counts().unstack().fillna(0)

    # Grafik oluşturma ve kaydetme
    plt.figure(figsize=(14, 7))
    time_series_df.plot(figsize=(14, 7), colormap='viridis')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Attack Count Over Time')
    plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y')

    # Grafiği kaydet
    output_path = 'grafikler/attack_count_over_time.png'
    plt.savefig(output_path, bbox_inches='tight')

    # Grafiği göster
    plt.show()