import streamlit as st

from utils import analyze_skills, extract_text_from_file, generate_text

st.set_page_config(page_title="🤖 HR Assistant", page_icon="🤖", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)

# Заголовок приложения
st.title("🤖 HR Assistant")
st.markdown("### Анализ резюме и рекомендации")

# Боковая панель с инструкциями
with st.sidebar:
    st.markdown("### 📝 Инструкции")
    st.markdown("""
    1. Введите описание вакансии
    2. Загрузите резюме (PDF или DOCX)
    3. Нажмите кнопку "Анализировать"
    4. Получите результаты анализа и рекомендации
    """)

# Основной контент
job_description = st.text_area(
    "Описание вакансии",
    height=200,
    placeholder="Введите описание вакансии здесь...",
)

uploaded_file = st.file_uploader(
    "Загрузите резюме",
    type=["pdf", "docx"],
    help="Поддерживаются файлы PDF и DOCX",
)

if uploaded_file and job_description:
    # Извлекаем текст из резюме
    resume_text = extract_text_from_file(uploaded_file)

    if resume_text:
        # Анализируем навыки
        skills_analysis = analyze_skills(job_description, resume_text)

        # Отображаем результаты анализа
        st.markdown("### 📊 Результаты анализа")

        # Метрики
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Соответствие навыков", f"{skills_analysis['similarity']:.0%}")
        with col2:
            st.metric("Отсутствующие навыки", len(skills_analysis["missing_tech"]))
        with col3:
            st.metric("Дополнительные навыки", len(skills_analysis["extra_tech"]))

        # Детальный анализ
        st.markdown("#### 🔍 Детальный анализ")

        # Отсутствующие навыки
        if skills_analysis["missing_tech"]:
            st.markdown("##### ❌ Отсутствующие технические навыки")
            for skill in skills_analysis["missing_tech"]:
                st.markdown(
                    f'<div class="skill-item">{skill}</div>', unsafe_allow_html=True
                )

        if skills_analysis["missing_other"]:
            st.markdown("##### ❌ Отсутствующие другие навыки")
            for skill in skills_analysis["missing_other"]:
                st.markdown(
                    f'<div class="skill-item">{skill}</div>', unsafe_allow_html=True
                )

        # Дополнительные навыки
        if skills_analysis["extra_tech"]:
            st.markdown("##### ✅ Дополнительные технические навыки")
            for skill in skills_analysis["extra_tech"]:
                st.markdown(
                    f'<div class="skill-item">{skill}</div>', unsafe_allow_html=True
                )

        if skills_analysis["extra_other"]:
            st.markdown("##### ✅ Дополнительные другие навыки")
            for skill in skills_analysis["extra_other"]:
                st.markdown(
                    f'<div class="skill-item">{skill}</div>', unsafe_allow_html=True
                )

        # Генерация рекомендаций
        st.markdown("### 💡 Рекомендации")

        try:
            # Формируем промпт для генерации рекомендаций
            prompt = f"""
            На основе анализа резюме и описания вакансии, предоставь рекомендации по улучшению резюме.
            
            Описание вакансии:
            {job_description}
            
            Анализ навыков:
            - Отсутствующие технические навыки: {', '.join(skills_analysis['missing_tech'])}
            - Отсутствующие другие навыки: {', '.join(skills_analysis['missing_other'])}
            - Дополнительные технические навыки: {', '.join(skills_analysis['extra_tech'])}
            - Дополнительные другие навыки: {', '.join(skills_analysis['extra_other'])}
            
            Предоставь конкретные рекомендации по:
            1. Как подчеркнуть имеющиеся навыки
            2. Какие навыки стоит развить
            3. Как лучше представить опыт
            """

            with st.spinner("Генерируем рекомендации..."):
                recommendations = generate_text(prompt)
                st.markdown(recommendations)

        except Exception as e:
            st.error(f"Ошибка при генерации рекомендаций: {str(e)}")
    else:
        st.error("Не удалось извлечь текст из файла")
