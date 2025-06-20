# Fire Alarm ML Dashboard

Веб-приложение для анализа и инференса моделей машинного обучения на основе данных о потенциально опасных небесных объектах (астероидов), приближающихся к Земле.

---

## Функциональность

- Информация о разработчике и датасете
- Визуализация данных:
  - Распределения
  - Корреляции
  - Boxplot и scatter-пар
- Предсказание:
  - Ручной ввод признаков
  - Загрузка CSV-файла с данными
  - Выбор одной или нескольких моделей для инференса

---

## Структура проекта

```
rgr_ml/
├── app/
│   └── app.py             
├── models/
│   ├── knn_model.pkl
│   ├── bagging.pkl
│   ├── boosting.pkl
│   ├── stacking.pkl
│   └── catboost_model.cbm
├── data/
│   └── neo_task.csv        
├── requirements.txt
└── README.md
```

---

## Используемые признаки

- `est_diameter_min`: минимальный расчетный диаметр (км)
- `est_diameter_max`: максимальный расчетный диаметр (км)
- `relative_velocity`: относительная скорость (км/ч)
- `miss_distance`: расстояние до Земли (км)
- `absolute_magnitude`: яркость объекта
- `est_diameter_avg`: средний диаметр (автоматически рассчитывается)

---


## Автор

**Ессина Анастасия Антоновна**  
Группа: ФИТ-231  
Тема РГР: *Разработка Web-приложения для инференса моделей ML и анализа данных*
