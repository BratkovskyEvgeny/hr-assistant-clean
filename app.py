import streamlit as st

from utils import (
    analyze_skills,
    calculate_similarity,
    extract_text_from_file,
    get_detailed_analysis,
    query_llm,
)

# Настройка страницы
st.set_page_config(
    page_title="HR Assistant - Оценка резюме",
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
        similarity_score = calculate_similarity(job_description, resume_text)
        analysis_results = analyze_skills(job_description, resume_text)
        st.success("✅ Анализ соответствия завершен")

    with st.spinner("Выполняем детальный анализ..."):
        detailed_analysis = get_detailed_analysis(job_description, resume_text)
        st.success("✅ Детальный анализ завершен")

    st.markdown("### 📊 Результаты анализа")

    # Отображаем результаты анализа
    if analysis_results:
        # Отображаем общий процент соответствия
        overall_match = analysis_results.get("overall_match", 0.0)
        st.metric(
            "Общее соответствие",
            f"{overall_match:.1f}%",
            delta=f"{overall_match - 50:.1f}%",
            delta_color="normal" if overall_match >= 50 else "inverse",
        )

        # Создаем вкладки для разных типов анализа
        tab1, tab2, tab3 = st.tabs(["Опыт работы", "Образование", "Навыки"])

        with tab1:
            if "experience" in analysis_results:
                exp_data = analysis_results["experience"]
                st.subheader("Опыт работы")
                st.write(f"**Соответствие:** {exp_data['relevance']:.1f}%")
                if exp_data["text"].strip():
                    st.text_area(
                        "Текст секции", exp_data["text"], height=120, disabled=True
                    )
                else:
                    st.warning("Секция 'Опыт работы' не найдена или пуста.")
                if exp_data["responsibilities"]:
                    st.write("**Обязанности:**")
                    for resp in exp_data["responsibilities"]:
                        st.write(f"- {resp}")
                if exp_data["skills"]:
                    st.write("**Приобретенные навыки:**")
                    for skill in exp_data["skills"]:
                        st.write(f"- {skill}")

        with tab2:
            if "education" in analysis_results:
                edu_data = analysis_results["education"]
                st.subheader("Образование")
                st.write(f"**Соответствие:** {edu_data['relevance']:.1f}%")
                if edu_data["text"].strip():
                    st.text_area(
                        "Текст секции", edu_data["text"], height=120, disabled=True
                    )
                else:
                    st.warning("Секция 'Образование' не найдена или пуста.")
                if edu_data["skills"]:
                    st.write("**Приобретенные навыки:**")
                    for skill in edu_data["skills"]:
                        st.write(f"- {skill}")

        with tab3:
            if "skills" in analysis_results:
                skills_data = analysis_results["skills"]
                st.subheader("Навыки")
                st.write(f"**Соответствие:** {skills_data['relevance']:.1f}%")
                if skills_data["text"].strip():
                    st.text_area(
                        "Текст секции", skills_data["text"], height=120, disabled=True
                    )
                else:
                    st.warning("Секция 'Навыки' не найдена или пуста.")
                if skills_data["skills"]:
                    st.write("**Технические навыки:**")
                    for skill in skills_data["skills"]:
                        st.write(f"- {skill}")

        # Отображаем отсутствующие навыки и опыт
        missing_skills = analysis_results.get("missing_skills", [])
        missing_experience = analysis_results.get("missing_experience", [])
        if missing_skills or missing_experience:
            st.warning("**Отсутствующие навыки и опыт:**")
            if missing_skills:
                st.write("**Отсутствующие навыки:**")
                for skill in missing_skills:
                    st.write(f"- {skill}")
            if missing_experience:
                st.write("**Отсутствующий опыт:**")
                for exp in missing_experience:
                    st.write(f"- {exp}")

        # Выводим весь текст резюме для диагностики
        st.markdown("#### 📝 Весь текст резюме (отладка)")
        st.text_area("Весь текст резюме", resume_text, height=200, disabled=True)

        # Отображаем отладочную информацию о найденных заголовках и тексте между ними
        debug_headers = analysis_results.get("_debug_headers", [])
        if debug_headers:
            st.markdown("#### 🐞 Найденные заголовки и их позиции (отладка)")
            for i, h in enumerate(debug_headers):
                section = h["section"]
                start = h["end"]
                end = (
                    debug_headers[i + 1]["start"]
                    if i + 1 < len(debug_headers)
                    else len(resume_text)
                )
                section_text = resume_text[start:end].strip()
                st.write(
                    f"Секция: {section}, Заголовок: '{h['keyword']}', Позиция: {h['start']}-{h['end']}"
                )
                st.write(f"Текст секции (первые 100 символов): {section_text[:100]}")

    # Кнопка для глубокого LLM-анализа
    if uploaded_file is not None and job_description:
        if st.button("Глубокий LLM-анализ (Pythia-70M)"):
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
                st.markdown("### 📝 Результаты LLM-анализа (Pythia-70M)")
                st.write(result)
