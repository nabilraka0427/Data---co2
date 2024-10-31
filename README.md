## DATA CO2
Library yg di butuhkan
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```
import data 
``` python
df = pd.read_csv('E:\semester 3\IPUSD\data\DataCO2.csv')
```
Memunculkan 5 baris data awal agar mmepermudah pengecekan data
``` python
df.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nama</th>
      <th>Model</th>
      <th>Volume</th>
      <th>Bobot</th>
      <th>CO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toyoty</td>
      <td>Aygo</td>
      <td>1000</td>
      <td>790</td>
      <td>99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mitsubishi</td>
      <td>Space Star</td>
      <td>1200</td>
      <td>1160</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Skoda</td>
      <td>Citigo</td>
      <td>1000</td>
      <td>929</td>
      <td>95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fiat</td>
      <td>500</td>
      <td>900</td>
      <td>865</td>
      <td>90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mini</td>
      <td>Cooper</td>
      <td>1500</td>
      <td>1140</td>
      <td>105</td>
    </tr>
  </tbody>
</table>
</div>

Apakah ada tipe data non numerik pada DataCO2.csv?
``` python
print(df.dtypes)

non_numeric_columns = df.select_dtypes(include=['object']).columns
if len(non_numeric_columns) > 0:
    print("\nKolom non-numerik ditemukan:")
    print(non_numeric_columns)

    print("\nPenentuan tipe (nominal atau ordinal):")
    for col in non_numeric_columns:
        unique_values = df[col].unique()
        print(f"- Kolom '{col}' memiliki nilai unik: {unique_values}")

        if df[col].nunique() <= 10:  
            print(f"  Tipe '{col}': Cenderung nominal")
        else:
            print(f"  Tipe '{col}': Cenderung ordinal")
else:
    print("\nTidak ada kolom non-numerik.")
```
## OUTPUT
Nama      object
Model     object
Volume     int64
Bobot      int64
CO2        int64
dtype: object

Kolom non-numerik ditemukan:
Index(['Nama', 'Model'], dtype='object')

Penentuan tipe (nominal atau ordinal):
- Kolom 'Nama' memiliki nilai unik: ['Toyoty' 'Mitsubishi' 'Skoda' 'Fiat' 'Mini' 'VW' 'Mercedes' 'Ford' 'Audi'
 'Hyundai' 'Suzuki' 'Honda' 'Hundai' 'Opel' 'BMW' 'Mazda' 'Volvo']
  Tipe 'Nama': Cenderung ordinal
- Kolom 'Model' memiliki nilai unik: ['Aygo' 'Space Star' 'Citigo' '500' 'Cooper' 'Up!' 'Fabia' 'A-Class'
 'Fiesta' 'A1' 'I20' 'Swift' 'Civic' 'I30' 'Astra' '1' '3' 'Rapid' 'Focus'
 'Mondeo' 'Insignia' 'C-Class' 'Octavia' 'S60' 'CLA' 'A4' 'A6' 'V70' '5'
 'E-Class' 'XC70' 'B-Max' '216' 'Zafira' 'SLK']
  Tipe 'Model': Cenderung ordinal

### Membangun model regresu untuk data CO2
Dalam data ini model regresi yang paling tepat ada "Linear Regresion", karena:
  
- Fokus pada Satu Variabel: Anda hanya menggunakan satu fitur (Volume) untuk memprediksi satu target (CO2), sehingga regresi linear sangat sesuai.
- Prediksi Berbasis Data: Jika Anda memiliki data historis tentang volume kendaraan dan emisi CO2, regresi linear dapat memberikan model yang baik untuk membuat prediksi berdasarkan data tersebut.

## Pengolahan Data Kategorikal
``` python
columns = df.select_dtypes(include=['object']).columns

for col in columns:
    unique_values = df[col].unique()
    print(f"{col}:\n{unique_values}\n")
```
- fungsi: Bagian kode ini pertama-tama mengidentifikasi kolom-kolom dalam DataFrame df yang memiliki tipe data objek (string). Kemudian, untuk setiap kolom tersebut, kode ini menampilkan nilai unik yang terdapat di dalam kolom tersebut.
- Tujuan: Ini membantu untuk memahami data kategorikal yang ada sebelum dilakukan encoding.
## OUTPUT
Nama:
['Toyoty' 'Mitsubishi' 'Skoda' 'Fiat' 'Mini' 'VW' 'Mercedes' 'Ford' 'Audi'
 'Hyundai' 'Suzuki' 'Honda' 'Hundai' 'Opel' 'BMW' 'Mazda' 'Volvo']

Model:
['Aygo' 'Space Star' 'Citigo' '500' 'Cooper' 'Up!' 'Fabia' 'A-Class'
 'Fiesta' 'A1' 'I20' 'Swift' 'Civic' 'I30' 'Astra' '1' '3' 'Rapid' 'Focus'
 'Mondeo' 'Insignia' 'C-Class' 'Octavia' 'S60' 'CLA' 'A4' 'A6' 'V70' '5'
 'E-Class' 'XC70' 'B-Max' '216' 'Zafira' 'SLK']

## Encoding Kolom Kategorikal

``` python
le = LabelEncoder()

# Melakukan encoding pada semua kolom non-numerik
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Menampilkan hasil setelah encoding
print("Data setelah encoding:\n", df.head())

```
- Fungsi: Di sini, LabelEncoder digunakan untuk mengonversi nilai-nilai kategorikal dalam kolom-kolom non-numerik menjadi format numerik. Ini penting karena model regresi linear tidak dapat menangani data kategorikal secara langsung.
- Tujuan: Memastikan semua kolom dalam DataFrame df adalah numerik sebelum model regresi dibangun, sehingga model dapat berfungsi dengan baik.

|  Nama |  Model |  Volume |  Bobot |  CO2 |
|-------|--------|---------|--------|------|
|   14  |   10   |   1000  |   790  |  99  |
|   10  |   29   |   1200  |  1160  |  95  |
|   12  |   14   |   1000  |   929  |  95  |
|   2   |   4    |   900   |   865  |  90  |
|   9   |   16   |   1500  |  1140  |  105 |

## Model Regresi Linear
``` pyhton
data = pd.DataFrame(df)

X = df[['Volume']]
y = df['CO2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)
```
- Fungsi: Di bagian ini, Anda mempersiapkan data untuk pelatihan model regresi linear. X berisi fitur (dalam hal ini, hanya kolom Volume), dan y berisi target (kolom CO2). Kemudian, dataset dibagi menjadi data latih dan data uji, model dibuat dan dilatih, lalu prediksi dilakukan dan hasil evaluasi dicetak.
- Tujuan: Membangun dan mengevaluasi model regresi linear berdasarkan data yang telah diproses.
## OUTPUT 
- Mean Squared Error: 43.36127961542357
- Koefisien: [0.01076841]
- Intercept: 84.80350970486572

## Model regresi multivariat
``` python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

print("\nKoefisien Regresi (slope):", model.coef_)
print("Intercept:", model.intercept_)
```
## OUTPUT
- Mean Squared Error: 39.66503640551568
- R-squared Score: 0.4322159852487132

- Koefisien Regresi (slope): [-0.18598243  0.16720208  0.00760237  0.00701463]
- Intercept: 79.4421088510575
## Kelebihan regresi multivariat
- Multivariat: Kode terakhir dapat digunakan untuk membangun model regresi multivariat, di mana beberapa variabel independen digunakan untuk memprediksi variabel dependen. Ini dapat memberikan pemahaman yang lebih baik tentang hubungan antara variabel jika ada lebih dari satu faktor yang mempengaruhi hasil.

- Fleksibilitas: Dengan mengambil semua kolom kecuali kolom terakhir, Anda dapat dengan mudah menambahkan atau menghapus kolom dalam dataset tanpa harus mengubah bagian kode yang memilih kolom untuk X.

- Struktur: Kode terakhir memberikan struktur yang lebih baik dalam pemilihan variabel, yang membuatnya lebih mudah dibaca dan dipahami.
