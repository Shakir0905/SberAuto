import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Загрузите данные в DataFrame df
df = pd.read_csv(r'G:\Мой диск\AI\data\df.csv')
print("Data loaded successfully.")

def binarize_event_action(data, col):
    # Create a new column called event_action_target
    target_values = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                     'sub_open_dialog_click', 'sub_custom_question_submit_click',
                     'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                     'sub_car_request_submit_click']
    return data[col].apply(lambda x: 1 if x in target_values else 0)

# Преобразование таргета
df['event_action'] = binarize_event_action(df, 'event_action')

# Удаление неинформативных признаков
df = df.drop(['utm_keyword', 'device_os', 'device_model', 'geo_country'], axis=1)

# Замена пропущенных значений методом mode
df = df.fillna(df.mode().iloc[0])

# Удаление дубликатов
df = df.drop_duplicates()

def categorize_feature(feature):
    # Вычисляем границы разбиения на 4 категории
    bins = np.linspace(feature.min(), feature.max(), 5)

    # Определяем метки категорий
    labels = ['Very Low', 'Low', 'Medium', 'High']

    # Используем метод cut для разбиения и категоризации признака
    return pd.cut(feature, bins=bins, labels=labels, right=False)

df['visit_number'] = categorize_feature(df['visit_number'])

df['visit_date'] = pd.to_datetime(df['visit_date'])
df['visit_time'] = pd.to_datetime(df['visit_time'])
df['visit_hour'] = df['visit_time'].dt.hour
df['part_of_day'] = pd.cut(df['visit_hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], include_lowest=True)
df['day_of_week'] = df['visit_date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Количество визитов по источникам
df['visit_counts_by_source'] = df.groupby(['client_id', 'utm_source'])['visit_time'].transform('count')

# Предпочитаемый канал пользователя
df['preferred_utm_source'] = df.groupby('client_id')['utm_source'].transform(lambda x: x.mode()[0])

# Удаляем исходные колонки
df = df.drop(['visit_date', 'visit_time'], axis=1)

X = df.drop(['session_id', 'client_id', 'event_action'], axis=1)
y = df[['event_action']]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

columns = ['utm_source',
           'utm_medium',
           'utm_campaign',
           'utm_adcontent',
           'device_category',
           'device_brand',
           'device_screen_resolution',
           'device_browser',
           'geo_city']

class ProbabilityEncoder:
    def __init__(self, columns=None, id_column=None):
        self.columns = columns
        self.id_column = id_column
        self.percentage_dicts = {}

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        self.columns = self.columns if self.columns else data.columns.tolist()
        self.id_column = self.id_column if self.id_column else data.columns[0]
        self.target = data.columns[-1]

        for column in self.columns:
            total_counts = data.groupby(column)[self.target].count()
            positive_counts = data.groupby(column)[self.target].sum()
            percentages = (positive_counts / total_counts) * 100
            self.percentage_dicts[column] = percentages.to_dict()

        return self

    def transform(self, X):
        data = X.copy()
        for column, percentages in self.percentage_dicts.items():
            data[column] = data[column].map(percentages).fillna(0)

        return data

    def fit_transform(self, X, y):
        self.fit(X, y)
        transformed_data = self.transform(X)
        return transformed_data, self.percentage_dicts

prob_encoder = ProbabilityEncoder(columns=columns, id_column='client_id')

X_train_encoded, percentage_dicts = prob_encoder.fit_transform(X_train, y_train)

X_train_encoded = prob_encoder.transform(X_train)
X_test_encoded = prob_encoder.transform(X_test)
print("Data prepare successfully.")

# Автоматически определите числовые и категориальные столбцы
numeric_features = X.select_dtypes(include=['int', 'float']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Создайте обработчики для числовых и категориальных столбцов
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Соедините обработчики в один пайплайн с использованием ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Определите модель
gb = GradientBoostingClassifier(random_state=42)

# Объедините предварительную обработку и модель в один пайплайн
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', gb)])

print("Training the model...")
clf.fit(X_train, y_train)
print("Model training completed.")

# Теперь вы можете использовать clf для предсказания на новых данных
predictions = clf.predict(X_test)
print("Predictions completed.")

# Сохранение модели
model_filename = 'gradient_boosting_model.joblib'
joblib.dump(clf, model_filename)
print("Model saved as", model_filename)

# Сохранение percentage_dicts
percentage_dicts_filename = 'percentage_dicts.joblib'
joblib.dump(percentage_dicts, percentage_dicts_filename)
print("Percentage dicts saved as", percentage_dicts_filename)