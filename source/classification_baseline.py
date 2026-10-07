import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# загружаем набор данных
df = pd.read_csv("data/heart.csv")
# выводим информацию об этом датафрейме чтобы узнать:
# название колонок, типы данных, и наличие пропусков
print(df.info())
print(df.isna().sum())

# извлекаем целевую переменную и делим признаки на обучающую и тестовую выборки 
Y = df["target"].values 
X = df.drop(columns=["target"])

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)

# конвертируем в тензоры pytorch
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# реализуем алгоритм нулевого правила
def Zero_Rule_Prediction_Algorithm(y_train, y_test):
    # находим значение класса чаще всего встречающегося в обучающей выборке 
    max_value = torch.mode(y_train)
    # достаем значение этого класса из кортежа
    max_class = max_value.values.item()
    predicted = torch.full((len(y_test),), fill_value=max_class)
    return predicted

# реализуем алгоритм случайного предсказания 
def Random_Prediction_Algorithm(y_train, y_test):
    # отбираем только уникальные значения
    unique_classes = torch.unique(y_train)
    # генерируем случайный индекс
    random_index = torch.randint(low=0, high=len(unique_classes), size=(len(y_test),))
    predicted = unique_classes[random_index]
    return predicted

# фиксируем функцию seed для воспроизводимости экспериментов и получаем предсказания обоих алгоритмов 
torch.manual_seed(42)
# запускаем алгоритмы
y_pred_random = Random_Prediction_Algorithm(y_train, y_test)   
y_pred_ZeroRule = Zero_Rule_Prediction_Algorithm(y_train, y_test)

# рассчитываем метрику accuracy - долю правильных ответов
accuracy_random = (y_pred_random == y_test).float().mean().item()
accuracy_ZeroRule = (y_pred_ZeroRule == y_test).float().mean().item()


# выводим результаты 
print("--------------------------------------------------------------------------------")
print(f"Случайное предсказание (Random): {accuracy_random:.4f} ({accuracy_random * 100:.1f}%)")
print(f"Нулевое правило (Zero Rule):     {accuracy_ZeroRule:.4f} ({accuracy_ZeroRule * 100:.1f}%)")
print("--------------------------------------------------------------------------------")



