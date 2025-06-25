# Personality Prediction from Text using DistilBERT

Предсказание черт личности Big Five (OCEAN) по тексту с помощью модели DistilBERT и интерпретации через Integrated Gradients.

## Описание проекта

Цель: по транскрипциям речи (из видеозаписей) предсказать 5 черт личности:
- Openness
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

**Задача формулируется как множественная регрессия.** Модель обучается предсказывать пять вещественных значений от 0 до 1.

---

## Используемые данные

- Датасет: [ChaLearn First Impressions](https://chalearnlap.cvc.uab.cat/dataset/24/description/)
- Входные данные: транскрипции речи
- Выход: 5 значений по шкале Big Five

---

## Архитектура модели

- Текстовый энкодер: `distilbert-base-uncased`
- Вектор `[CLS]` используется как представление текста
- Голова: Dropout → Linear(768 → 5)
- Функция потерь: `HuberLoss`
- Оптимизатор: `AdamW` (lr=2e-5, weight_decay=0.01)

---

## Интерпретация предсказаний

Модель объясняется с помощью **Integrated Gradients (Captum)**:
- Подсчитывается вклад каждого токена во входе
- Отбираются наиболее значимые слова
- Исключаются:
  - Стоп-слова (NLTK)
  - Спецтокены ([CLS], [SEP], ## и др.)
  - 50 самых частотных общих слов
