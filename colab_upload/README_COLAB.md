# Запуск проекта в Google Colab

## 1. Что загрузить

В Google Colab загрузи файл:

```text
Lane_Segmentation_TuSimple_Colab.ipynb
```

Файл лежит в этой папке:

```text
colab_upload/
```

## 2. Куда положить датасет

Датасет большой, около 24 GB, поэтому лучше не загружать его прямо в Colab как временный файл. Положи папку `TUSimple` в Google Drive:

```text
MyDrive/TUSimple/
└── train_set/
    ├── clips/
    ├── label_data_0313.json
    ├── label_data_0531.json
    └── label_data_0601.json
```

В ноутбуке путь должен быть:

```python
DATASET_PATH = "/content/drive/MyDrive/TUSimple/train_set"
```

## 3. Включить GPU

В Colab открой:

```text
Среда выполнения -> Сменить среду выполнения -> Аппаратный ускоритель -> GPU
```

Потом запусти все ячейки сверху вниз.

## 4. Быстрый первый запуск

Для первой проверки оставь:

```python
LIMIT = 200
```

После проверки можно увеличить:

```python
LIMIT = 1000
```

## 5. Результат

После обучения ноутбук покажет:

- график loss;
- графики Dice и IoU;
- исходное изображение;
- настоящую маску;
- предсказанную маску.

Модель сохраняется в Colab:

```text
/content/unet_tusimple.keras
```
