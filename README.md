# Lane Segmentation on TuSimple with U-Net

Учебный проект по сегментации дорожных полос на датасете TuSimple. Проект содержит полный pipeline: чтение разметки TuSimple, генерацию бинарных масок, обучение U-Net, метрики Dice/IoU и визуализацию результата.

## Что есть в проекте

- генерация mask-изображений из `lanes` и `h_samples`;
- исправленная обработка TuSimple JSON;
- U-Net на TensorFlow/Keras;
- метрики `Dice` и `IoU`;
- Colab-ноутбук для быстрого запуска;
- локальный скрипт обучения;
- локальный интерфейс для проверки модели на своих изображениях;
- обученная demo-модель на 1000 изображениях;
- сохранение модели и графиков обучения.

## Структура

```text
lane-segmentation-tusimple/
├── notebooks/
│   └── tusimple_unet_colab.ipynb
├── models/
│   └── unet_tusimple_1000.keras
├── results/
│   └── example_segmentation_1000.png
├── src/
│   ├── data.py
│   ├── local_app.py
│   ├── metrics.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Датасет

Скачай TuSimple Lane Detection Dataset и положи папку `train_set` в Google Drive или локально. Обычно внутри есть:

```text
train_set/
├── clips/
├── label_data_0313.json
├── label_data_0531.json
└── label_data_0601.json
```

Датасет не добавляется в GitHub-репозиторий, потому что он большой. Для него в `.gitignore` уже добавлены папки `data/`, `datasets/` и `TUSimple/`.

## Запуск в Google Colab

Открой ноутбук:

```text
notebooks/tusimple_unet_colab.ipynb
```

В Colab поменяй путь:

```python
DATASET_PATH = "/content/drive/MyDrive/TUSimple/train_set"
```

Для быстрой проверки оставь:

```python
LIMIT = 200
```

Когда всё заработает, можно увеличить до `1000` или поставить `None`, если хватает памяти.

## Локальный запуск

Установка:

```bash
pip install -r requirements.txt
```

Для уже скачанного датасета на этом компьютере путь такой:

```text
C:\Users\qerte\Downloads\archive\TUSimple\train_set
```

Пример обучения:

```bash
python src/train.py ^
  --dataset-root "C:/Users/qerte/Downloads/archive/TUSimple/train_set" ^
  --json "C:/Users/qerte/Downloads/archive/TUSimple/train_set/label_data_0313.json" ^
  --limit 200 ^
  --epochs 10 ^
  --batch-size 8
```

Можно указать несколько JSON-файлов:

```bash
python src/train.py ^
  --dataset-root "C:/Users/qerte/Downloads/archive/TUSimple/train_set" ^
  --json ^
  "C:/Users/qerte/Downloads/archive/TUSimple/train_set/label_data_0313.json" ^
  "C:/Users/qerte/Downloads/archive/TUSimple/train_set/label_data_0531.json" ^
  "C:/Users/qerte/Downloads/archive/TUSimple/train_set/label_data_0601.json"
```

После обучения результаты появятся в папке `outputs/`:

- `unet_tusimple.keras` - обученная модель;
- `training_loss.png` - график ошибки;
- `training_metrics.png` - график Dice и IoU.

## Получить готовую сегментацию

Сначала обучи модель:

```bash
python src/train.py ^
  --dataset-root "C:/Users/qerte/Downloads/archive/TUSimple/train_set" ^
  --json "C:/Users/qerte/Downloads/archive/TUSimple/train_set/label_data_0313.json" ^
  --limit 200 ^
  --epochs 10 ^
  --batch-size 8
```

После этого запусти сегментацию на одной картинке:

```bash
python src/predict.py ^
  --model "outputs/unet_tusimple.keras" ^
  --image "C:/Users/qerte/Downloads/archive/TUSimple/train_set/clips/0313-1/6040/20.jpg" ^
  --output "outputs/example_segmentation.png"
```

Готовый результат будет сохранён здесь:

```text
outputs/example_segmentation.png
```

На изображении будут три части: исходная картинка, предсказанная маска и наложение полос на дорогу.

## Локальная проверка через окно

В репозитории уже есть demo-модель, обученная на 1000 изображениях:

```text
models/unet_tusimple_1000.keras
```

Запусти локальное приложение:

```bash
python src/local_app.py
```

В окне:

1. нажми `Load model` и выбери `models/unet_tusimple_1000.keras`;
2. нажми `Open image` и выбери любую дорожную фотографию;
3. посмотри `Image`, `Predicted mask` и `Overlay`;
4. при необходимости поменяй `Threshold`;
5. нажми `Save result`, чтобы сохранить картинку с наложенными полосами.

Если хочешь проверить на готовых кадрах из датасета, можно взять изображения из TuSimple `train_set/clips/...`.

## GitHub

В репозиторий можно заливать:

- исходный код из `src/`;
- Colab-ноутбуки из `notebooks/`;
- README и requirements;
- пример результата `results/example_segmentation_1000.png`;
- demo-модель `models/unet_tusimple_1000.keras`.

В репозиторий не нужно заливать:

- полный датасет TuSimple;
- zip-архивы с изображениями;
- временные файлы Colab;
- дополнительные большие модели.

## Почему Dice и IoU важнее accuracy

В задаче сегментации большая часть изображения является фоном. Поэтому `accuracy` может быть высокой даже у плохой модели, которая почти везде предсказывает фон. Метрики `Dice` и `IoU` лучше показывают качество совпадения предсказанной полосы с настоящей маской.

## Идеи для улучшения

- увеличить количество изображений;
- добавить аугментации;
- сравнить U-Net с DeepLabV3+;
- обучать дольше на GPU;
- добавить post-processing для сглаживания линий.
