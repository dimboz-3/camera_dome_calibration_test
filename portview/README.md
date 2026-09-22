# portview — камера за толстым сферическим иллюминатором

Генератор синтетических калибровочных данных (Mitsuba 3) + независимая проверка
тем же кодом сцены в Blender (Cycles) + компаратор изображений.

## Сцена
- Система координат «автомобильная», правая: X — вперёд, Y — влево, Z — вверх,
  начало = центр авто = центр купола.
- Купол: сферическая стеклянная оболочка (r_inner 6 м, толщина 50 мм, IOR 1.5,
  среда с обеих сторон — воздух).
- Камера: 1920×1535, f = 1036 px, положение в полярных координатах относительно
  купола (pitch 45° → z = x, азимут 0°, на 1 см внутрь от внутренней поверхности),
  смотрит вдоль +X, up = +Z.
- Шахматка: строго перед камерой, плоскость ⊥ оптической оси, 28×20 клеток
  (параметризуется). Дистанция считается в общем виде:
  `D = cell_size * f / (coverage * max(W/cols, H/rows))`, покрытие ≥80% по обеим
  осям; дистанция автоматически увеличивается, если доска попала внутрь купола.

## Установка
```bash
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Приложение 1: генератор
```bash
cd portview && .venv/bin/python generate.py                 # сцена по умолчанию
.venv/bin/python generate.py --pitch 45 --dry-run           # только размещение
.venv/bin/python generate.py --batch 5 --pitch-min 35 --pitch-max 55
```
Выход — плоская папка `data/`:
- `img_0001_<timestamp>.png` — что видит камера (Mitsuba, scalar_rgb, path tracer);
- `img_0001_<timestamp>.json` — полное описание сцены = ground truth
  (купол, камера, доска, GT-координаты узлов шахматки `corners_gt`);
- `checker_28x20.png` — общая текстура шахматки для обоих движков.

## Рендер той же сцены в Blender (независимая проверка)
```bash
blender -b -P blender_build.py -- data/img_0001_<ts>.json -o data/render
blender -b -P blender_build.py -- data -o data/render        # батч: все *.json
# либо, если установлен python-пакет bpy:
.venv/bin/python blender_build.py data/img_0001_<ts>.json -o data/render
```

## Приложение 2: компаратор
```bash
.venv/bin/python compare.py data/img_0001_<ts>.png data/render/img_0001_<ts>.png
.venv/bin/python compare.py --dir-a data --dir-b data/render --out data/compare
```
Считает узлы шахматки (27×19 внутренних углов) в обоих изображениях, выводит
mean/median/p95/max/rms в пикселях, сохраняет overlay и `summary.csv`.

## Соглашение об именах
`<prefix>_<index:04d>_<YYYYMMDDTHHMMSS>` — пары (.png/.json) и рендер Blender
носят одно имя; timestamp продублирован в JSON (`meta.timestamp`).
