import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from catboost import CatBoostClassifier
#from sklearn.preprocessing import StandardScaler


@st.cache_resource
def load_model(model_name):
    if model_name == "CatBoost":
        model = CatBoostClassifier()
        model.load_model("D:/ML/rgr_ml/models/catboost.cbm")
        return model
    with open(model_name, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data(path_to_csv):
    return pd.read_csv(path_to_csv, index_col=0)


st.set_page_config(page_title="Objects closest to Earth", layout="wide")
menu = st.sidebar.radio("Навигация", ["О разработчике", "О датасете", "Визуализации", "Предсказание"])

if menu == "О разработчике":
    st.title("Разработчик проекта")
    st.subheader("Ессина Анастасия Антоновна")
    st.text("Группа: ФИТ-231")
    st.markdown("**Тема РГР:** Разработка Web-приложения для инференса моделей ML и анализа данных")

elif menu == "О датасете":
    st.title("Описание датасета")
    st.markdown("""
    **Область:** Предсказание опасности ближайших к Земле объектов на основе их характеристик.

    **Целевая переменная:** `hazardous` (1 — опасен, 0 — неопасен)

    **Признаки:**
    - `id`: id  
    - `name`: название  
    - `est_diameter_min`:  минимальный расчетный диаметр в километрах   
    - `est_diameter_max`: максимальный расчетный диаметр в километрах
    - `relative_velocity`: скорость относительно Земли   
    - `miss_distance`: расстояние в километрах от Земли    
    - `absolute_magnitude`: собственная светимость (мера внутренней яркости небесного объекта) 
    """)

    st.subheader("Предобработка данных")
    st.markdown("""
    - В данных было обнаружено небольшое количество пропущенных значений.
    - Пропущенные значения заменены медианой либо модой соответствующих столбцов.
    - Проверено отсутствие дубликатов – дубликаты отсутствуют.
    - Удален столбец id.
    """)


elif menu == "Визуализации":
    df = load_data("D:/ML/rgr_ml/data/neo_task.csv")
    st.title("Анализ данных")

    # Обработка пропусков (при необходимости)
    df = df.dropna(subset=[
        "est_diameter_min", "est_diameter_max",
        "relative_velocity", "miss_distance",
        "absolute_magnitude", "hazardous"
    ])

    # Добавим средний диаметр
    df["avg_diameter"] = (df["est_diameter_min"] + df["est_diameter_max"]) / 2

    # 1. Распределение среднего диаметра
    st.subheader("1. Распределение среднего диаметра астероидов")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["avg_diameter"], kde=True, ax=ax1, color='skyblue')
    ax1.set_xlabel("Средний диаметр (км)")
    ax1.set_yscale("log")
    st.pyplot(fig1)

    # 2. Корреляционная матрица
    st.subheader("2. Корреляционная матрица")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # 3. Boxplot: относительная скорость по классу "опасен"
    st.subheader("3. Boxplot: относительная скорость в зависимости от опасности")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='hazardous', y='relative_velocity', data=df, ax=ax3)
    ax3.set_xlabel("Опасен")
    ax3.set_ylabel("Относительная скорость (км/ч)")
    st.pyplot(fig3)

    # 4. Взаимосвязь скорости и расстояния (цвет — опасен/нет)
    st.subheader("4. Зависимость: скорость vs расстояние (по классам)")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(
        x='relative_velocity',
        y='miss_distance',
        hue='hazardous',
        palette='coolwarm',
        data=df,
        ax=ax4
    )
    ax4.set_xlabel("Относительная скорость (км/ч)")
    ax4.set_ylabel("Расстояние (км)")
    st.pyplot(fig4)

elif menu == "Предсказание":
    st.title("Предсказание опасности астероида")

    mode = st.radio("Выберите режим:", ["Ручной ввод", "Загрузка CSV"])

    # Список всех доступных моделей
    all_models = ["KNN", "Boosting", "CatBoost", "Bagging", "Stacking"]
    # Теперь — мультиселект
    selected_models = st.multiselect("Модели:", all_models)

    # Функция для загрузки одной модели по её названию
    @st.cache_resource
    def load_selected_model(name):
        if name == "KNN":
            return load_model("D:/ML/rgr_ml/models/knn_model.pkl")
        elif name == "Boosting":
            return load_model("D:/ML/rgr_ml/models/boosting.pkl")
        elif name == "CatBoost":
            return load_model("CatBoost")
        elif name == "Bagging":
            return load_model("D:/ML/rgr_ml/models/bagging.pkl")
        elif name == "Stacking":
            return load_model("D:/ML/rgr_ml/models/stacking.pkl")

        else:
            return None  # на случай передачи неизвестного имени

    # Загружаем выбранные модели в словарь
    models_dict = {}
    for name in selected_models:
        model_obj = load_selected_model(name)
        if model_obj is not None:
            models_dict[name] = model_obj

    # Режим «Ручной ввод»
    if mode == "Ручной ввод":
        st.subheader("Введите данные")

        # Поля ввода для каждого признака
        values = {}
        values['est_diameter_min'] = st.number_input("Мин диаметр (км)", min_value=0)
        values['est_diameter_max'] = st.number_input("Макс диаметр (км)", min_value=0)
        values['relative_velocity'] = st.number_input("Скорость")
        values['miss_distance'] = st.number_input("Расстояние до Земли (км)", min_value=0)
        values['absolute_magnitude'] = st.number_input("Яркость")

        # Собираем в DataFrame
        input_df = pd.DataFrame([values])

        if selected_models:
            st.subheader("Результаты предсказания")
            for name, model in models_dict.items():
                # Добавим средний диаметр
                input_df["est_diameter_avg"] = (input_df["est_diameter_min"] + input_df["est_diameter_max"]) / 2

                model_features = {
                    "KNN": ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance',
                            'absolute_magnitude', 'est_diameter_avg'],
                    "Boosting": ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance',
                                 'absolute_magnitude', 'est_diameter_avg'],
                    "CatBoost": ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance',
                                 'absolute_magnitude', 'est_diameter_avg'],
                    "Bagging": ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance',
                                'absolute_magnitude', 'est_diameter_avg'],
                    "Stacking": ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance',
                                 'absolute_magnitude', 'est_diameter_avg']
                }
                # Отберём нужные признаки
                required_features = model_features[name]
                input_model_df = input_df[required_features]

                # Предсказание
                pred = model.predict(input_model_df)[0]

                result_text = "Данный небесный объект опасен" if pred == 1 else "Данный небесный объект неопасен"
                st.write(f"**{name}**: {result_text}")
        else:
            st.warning("Выберите модель для предсказания.")

    # Режим «Загрузка CSV»
    else:
        st.subheader("Загрузите CSV-файл")
        uploaded_file = st.file_uploader("Файл должен содержать все необходимые признаки", type=["csv"])
        if uploaded_file:
            try:
                test_df = pd.read_csv(uploaded_file)
                if selected_models:
                    st.subheader("Результаты предсказания")
                    # Словарь: какие признаки нужны каждой модели

                    for name, model in models_dict.items():
                        preds = model.predict(test_df)
                        test_df[f"Prediction_{name.replace(' ', '_')}"] = preds
                    st.write(test_df.head())
                    st.success("Предсказание выполнено")
                else:
                    st.warning("Выберите модель для предсказания")
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")
