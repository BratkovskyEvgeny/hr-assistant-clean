import streamlit as st


# Настройка страницы
st.set_page_config(
    page_title="🤖 HR Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Стили для красивого отображения
st.markdown(
    """
    <style>
    /* Основные стили */
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Стили для контейнеров */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Анимации */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes scaleIn {
        from { transform: scale(0.95); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Анимированные классы */
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out forwards;
    }
    
    .scale-in {
        animation: scaleIn 0.5s ease-out forwards;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Стили для заголовков */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
        margin-bottom: 1rem;
        opacity: 0;
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Стили для кнопок */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        opacity: 0;
        animation: scaleIn 0.5s ease-out forwards;
    }
    
    .stButton>button:hover {
        background-color: #FF6B6B;
        box-shadow: 0 2px 8px rgba(255, 75, 75, 0.3);
        transform: translateY(-2px);
    }
    
    /* Стили для текстовых полей */
    .stTextArea>div>div>textarea {
        background-color: #262730;
        color: #FFFFFF;
        border: 1px solid #3E3E3E;
        border-radius: 4px;
        transition: all 0.3s ease;
        opacity: 0;
        animation: slideIn 0.5s ease-out forwards;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #FF4B4B;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2);
    }
    
    /* Стили для прогресс-бара */
    .stProgress .st-bo {
        background-color: #FF4B4B;
        opacity: 0;
        animation: scaleIn 0.5s ease-out forwards;
    }
    
    .stProgress .st-bp {
        background-color: #FF6B6B;
        transition: width 0.5s ease-out;
    }
    
    /* Стили для алертов */
    .stAlert {
        background-color: #262730;
        border: 1px solid #3E3E3E;
        border-radius: 4px;
        color: #FFFFFF;
        opacity: 0;
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Стили для вкладок */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #262730;
        border-radius: 4px;
        padding: 0.5rem;
        opacity: 0;
        animation: slideIn 0.5s ease-out forwards;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3E3E3E;
        transform: translateY(-2px);
    }
    
    /* Стили для метрик */
    .stMetric {
        background-color: #262730;
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
        opacity: 0;
        animation: scaleIn 0.5s ease-out forwards;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Стили для списков */
    .skill-item {
        background-color: #262730;
        border-radius: 4px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border: 1px solid #3E3E3E;
        opacity: 0;
        animation: slideIn 0.5s ease-out forwards;
        transition: all 0.3s ease;
    }
    
    .skill-item:hover {
        transform: translateX(10px);
        background-color: #3E3E3E;
        border-color: #FF4B4B;
    }
    
    /* Стили для файлового загрузчика */
    .stFileUploader>div {
        background-color: #262730;
        border: 1px solid #3E3E3E;
        border-radius: 4px;
        padding: 1rem;
        opacity: 0;
        animation: fadeIn 0.5s ease-out forwards;
        transition: all 0.3s ease;
    }
    
    .stFileUploader>div:hover {
        border-color: #FF4B4B;
        transform: translateY(-2px);
    }
    
    /* Стили для разделителей */
    hr {
        border-color: #3E3E3E;
        margin: 2rem 0;
        opacity: 0;
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Анимация для заголовка */
    .header-animation {
        opacity: 0;
        animation: fadeIn 1s ease-out forwards;
    }
    
    .header-animation h1 {
        animation: pulse 2s infinite;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Заголовок
st.markdown(
    """
    <div class='header-animation' style='text-align: center; margin-bottom: 3rem;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>
            🤖 HR Assistant
        </h1>
        <p style='color: #9CA3AF; font-size: 1.2rem;'>
            Оценка соответствия резюме требованиям вакансии
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# Основной контент
st.markdown("### 📋 Описание вакансии")
job_description = st.text_area(
    "Введите описание вакансии",
    height=200,
    help="Опишите требования к вакансии, необходимые навыки и опыт",
    placeholder="Вставьте текст описания вакансии здесь...",
)

st.markdown("### 📄 Загрузка резюме")
uploaded_file = st.file_uploader(
    "Загрузите резюме (PDF или DOCX)",
    type=["pdf", "docx"],
    help="Поддерживаются файлы в форматах PDF и DOCX",
)

if uploaded_file is not None and job_description:
    # Показываем прогресс анализа
    with st.spinner("Подготовка к анализу..."):
        # Извлекаем текст из резюме
        resume_text = extract_text_from_file(uploaded_file)
        st.success("✅ Текст из резюме успешно извлечен")

    with st.spinner("Анализируем соответствие резюме требованиям..."):
        # Анализируем соответствие
        analysis_results = analyze_skills(job_description, resume_text)
        st.success("✅ Анализ соответствия завершен")

    st.markdown("### 📊 Результаты анализа")

    # Отображаем результаты анализа
    if analysis_results:
        # Отображаем схожесть стеков
        stack_similarity = analysis_results.get("similarity", 0.0)
        st.metric(
            "Схожесть технологического стека",
            f"{stack_similarity:.1%}",
            delta=f"{stack_similarity - 0.5:.1%}",
            delta_color="normal" if stack_similarity >= 0.5 else "inverse",
        )

        # Отображаем отсутствующие и лишние навыки
        missing_skills = analysis_results.get("missing_skills", set())
        extra_skills = analysis_results.get("extra_skills", set())

        if missing_skills or extra_skills:
            col1, col2 = st.columns(2)

            with col1:
                if missing_skills:
                    st.warning("**Отсутствующие технологии:**")
                    for skill in sorted(missing_skills):
                        st.write(f"- {skill}")
                else:
                    st.success("**Все требуемые технологии присутствуют в резюме**")

            with col2:
                if extra_skills:
                    st.info("**Дополнительные технологии в резюме:**")
                    for skill in sorted(extra_skills):
                        st.write(f"- {skill}")
                else:
                    st.info("**Нет дополнительных технологий в резюме**")

    # Кнопка для глубокого LLM-анализа
    if uploaded_file is not None and job_description:
        if st.button("Глубокий LLM-анализ (DistilGPT-2)"):
            with st.spinner("Анализируем с помощью LLM..."):
                prompt = f"""
Описание вакансии:
{job_description}

Текст резюме:
{resume_text}

Проанализируй:
1. Какие требования вакансии не отражены в резюме?
2. Какие сильные стороны есть у кандидата?
3. Какие рекомендации по улучшению резюме?
"""
                result = query_llm(prompt)
                st.markdown("### 📝 Результаты LLM-анализа (DistilGPT-2)")
                st.write(result)
