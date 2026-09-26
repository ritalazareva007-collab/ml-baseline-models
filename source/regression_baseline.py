import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# загружаем набор данных в датафрейм pandas
df = pd.read_csv("data/housing.csv")
# выводим информацию об этом датафрейме чтобы узнать:
# название колонок, типы данных, и наличие пропусков
print(df.info())
print(df.isna().sum())

# обрабатываем пропуски, заполняя их медианой
df.fillna({'total_bedrooms' : df['total_bedrooms'].median()}, inplace=True)
print(df.isna().sum())

# извлекаем целевую переменную и разделяем данные на обучающую и тестовую выборки
Y = df['median_house_value'].values
X = df.drop(columns=['median_house_value'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)

# конвертируем массивы в тензоры
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# реализуем алгоритм случайного предсказания 
def Random_Prediction_Algorithm(y_train, y_test):
    # оставляем только уникальные значения из обучающей выборки
    unique = torch.unique(y_train)
    # генерируем случайные индексы для тестовой выборки
    random_index = torch.randint(low=0, high=len(unique), size=(len(y_test),))
    # генерируем предсказания 
    predicted = unique[random_index]
    return predicted

# реализуем алгоритм нулевого правила 
def Zero_Rule_Prediction_Algorithm(y_train, y_test):
    # считаем среднее арифметическое
    average = torch.mean(y_train)
    # заполняем средним арифемтическим вектор предсказаний 
    predicted = torch.full((len(y_test),),  fill_value=average.item())
    return predicted


# фиксируем функцию seed для воспроизводимости экспериментов и получаем предсказания обоих алгоритмов 
torch.manual_seed(42)
# запускаем алгоритмы
y_pred_random = Random_Prediction_Algorithm(y_train, y_test)   
y_pred_ZeroRule = Zero_Rule_Prediction_Algorithm(y_train, y_test)

# считаем средне-квадратичную ошибку
mse_random = F.mse_loss(y_pred_random, y_test)
mse_ZeroRule = F.mse_loss(y_pred_ZeroRule, y_test)
# среднюю абсолютную ошибку
mae_random = F.l1_loss(y_pred_random, y_test)
mae_ZeroRule = F.l1_loss(y_pred_ZeroRule, y_test)
# считаем корень из среднеквадратичной ошибки
rmse_random = torch.sqrt(mse_random)
rmse_ZeroRule = torch.sqrt(mse_ZeroRule)


# выводим результаты
#print(f"MSE for Random Prediction Algorithm {mse_random.item():.2f}")
#print(f"MSE for Zero Rule Prediction Algorithm {mse_ZeroRule.item():.2f}")
print("--------------------------------------------------------------------------------")
print(f"MAE for Random Prediction Algorithm {mae_random.item():.0f}")
print(f"MAE for Zero Rule Prediction Algorithm {mae_ZeroRule.item():.0f}")
print("--------------------------------------------------------------------------------")
print(f"RMSE for Random Prediction Algorithm {rmse_random.item():.0f}")
print(f"RMSE for Zero Rule Prediction Algorithm {rmse_ZeroRule.item():.0f}")

