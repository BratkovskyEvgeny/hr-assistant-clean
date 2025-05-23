import base64
import json

import requests
import streamlit as st

from src.utils import analyze_skills, extract_text_from_file

st.set_page_config(
    page_title="HR Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
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
    </style>
    """,
    unsafe_allow_html=True,
)

# Заголовок приложения
st.title("🤖 HR Assistant")
st.markdown("""
Этот инструмент поможет вам:
- Проанализировать соответствие навыков кандидата требованиям вакансии
- Сгенерировать рекомендации по улучшению резюме
- Получить советы по подготовке к собеседованию
""")

# Создаем две колонки
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Описание вакансии")
    job_description = st.text_area(
        "Введите описание вакансии",
        height=300,
        placeholder="Вставьте сюда текст описания вакансии...",
    )

with col2:
    st.subheader("📄 Резюме кандидата")
    resume_file = st.file_uploader(
        "Загрузите резюме (PDF или DOCX)", type=["pdf", "docx"]
    )

    if resume_file:
        resume_text = extract_text_from_file(resume_file)
        st.text_area("Текст резюме", value=resume_text, height=300, disabled=True)
    else:
        resume_text = ""

# Создаем секцию для логов
st.markdown("### Логи API запроса")
log_container = st.empty()

# Кнопка анализа
if st.button("🔍 Анализировать", type="primary"):
    if not job_description or not resume_text:
        st.error("Пожалуйста, заполните все поля")
    else:
        # Анализ навыков
        analysis = analyze_skills(job_description, resume_text)

        # Отображаем результаты анализа
        st.subheader("📊 Результаты анализа")

        # Создаем три колонки для метрик
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Схожесть навыков", f"{analysis['similarity']:.1%}")

        with metric_col2:
            st.metric(
                "Отсутствующие навыки",
                len(analysis["missing_tech"] | analysis["missing_other"]),
            )

        with metric_col3:
            st.metric(
                "Лишние навыки", len(analysis["extra_tech"] | analysis["extra_other"])
            )

        # Отображаем детали анализа
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🚫 Отсутствующие навыки")
            if analysis["missing_tech"]:
                st.markdown("**Технические навыки:**")
                for skill in sorted(analysis["missing_tech"]):
                    st.markdown(f"- {skill}")
            if analysis["missing_other"]:
                st.markdown("**Другие навыки:**")
                for skill in sorted(analysis["missing_other"]):
                    st.markdown(f"- {skill}")

        with col2:
            st.markdown("#### ✅ Лишние навыки")
            if analysis["extra_tech"]:
                st.markdown("**Технические навыки:**")
                for skill in sorted(analysis["extra_tech"]):
                    st.markdown(f"- {skill}")
            if analysis["extra_other"]:
                st.markdown("**Другие навыки:**")
                for skill in sorted(analysis["extra_other"]):
                    st.markdown(f"- {skill}")

        # Генерация рекомендаций
        st.subheader("💡 Рекомендации")

        # Подготавливаем данные для запроса
        prompt = f"""
        Проанализируй резюме кандидата и описание вакансии.
        
        Описание вакансии:
        {job_description}
        
        Резюме кандидата:
        {resume_text}
        
        Отсутствующие навыки:
        {', '.join(analysis['missing_tech'] | analysis['missing_other'])}
        
        Лишние навыки:
        {', '.join(analysis['extra_tech'] | analysis['extra_other'])}
        
        Предоставь рекомендации по улучшению резюме и подготовке к собеседованию.
        """

        try:
            # Получаем URL и учетные данные из конфигурации
            api_url = st.secrets["api"]["kaggle_url"]
            username = st.secrets["kaggle"]["username"]
            key = st.secrets["kaggle"]["key"]

            # Подготавливаем данные для запроса
            payload = {
                "input": {"prompt": prompt, "max_tokens": 1000, "temperature": 0.7}
            }

            # Формируем заголовки авторизации
            auth = f"{username}:{key}"
            auth_bytes = auth.encode("ascii")
            base64_auth = base64.b64encode(auth_bytes).decode("ascii")

            # Логируем детали запроса
            log_text = []
            log_text.append("=== ДЕТАЛИ ЗАПРОСА ===")
            log_text.append(f"URL запроса: {api_url}")
            log_text.append(
                f"Payload запроса: {json.dumps(payload, indent=2, ensure_ascii=False)}"
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Basic {base64_auth}",
                "Accept": "application/json",
            }
            log_text.append(
                f"Заголовки запроса: {json.dumps({k: v if k != 'Authorization' else '***' for k, v in headers.items()}, indent=2)}"
            )

            # Обновляем контейнер с логами
            log_container.code("\n".join(log_text), language="text")

            # Отправляем запрос
            try:
                log_text.append("Отправка запроса...")
                log_container.code("\n".join(log_text), language="text")

                response = requests.post(
                    api_url, json=payload, headers=headers, timeout=30, verify=True
                )
                log_text.append("Запрос отправлен успешно")
                log_container.code("\n".join(log_text), language="text")
            except requests.exceptions.SSLError as e:
                log_text.append(f"Ошибка SSL: {str(e)}")
                log_container.code("\n".join(log_text), language="text")
                raise Exception("Ошибка SSL при подключении к API")
            except requests.exceptions.ConnectionError as e:
                log_text.append(f"Ошибка подключения: {str(e)}")
                log_container.code("\n".join(log_text), language="text")
                raise Exception("Не удалось подключиться к API")
            except requests.exceptions.Timeout as e:
                log_text.append(f"Таймаут: {str(e)}")
                log_container.code("\n".join(log_text), language="text")
                raise Exception("Превышено время ожидания ответа от API")

            # Логируем детали ответа
            log_text.append("=== ДЕТАЛИ ОТВЕТА ===")
            log_text.append(f"Статус код: {response.status_code}")
            log_text.append(
                f"Заголовки ответа: {json.dumps(dict(response.headers), indent=2)}"
            )

            try:
                response_json = response.json()
                log_text.append(
                    f"Тело ответа: {json.dumps(response_json, indent=2, ensure_ascii=False)}"
                )
            except:
                log_text.append(f"Тело ответа: {response.text}")

            # Обновляем контейнер с логами
            log_container.code("\n".join(log_text), language="text")

            # Проверяем статус ответа
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "output" in result and "text" in result["output"]:
                        recommendations = result["output"]["text"]
                    elif "generated_text" in result:
                        recommendations = result["generated_text"]
                    elif "text" in result:
                        recommendations = result["text"]
                    else:
                        error_msg = result.get("message", "Неизвестная ошибка")
                        log_text.append(f"Неожиданный формат ответа: {result}")
                        log_container.code("\n".join(log_text), language="text")
                        raise Exception(f"Ошибка в ответе API: {error_msg}")
                except json.JSONDecodeError as e:
                    log_text.append(f"Ошибка при разборе JSON: {str(e)}")
                    log_text.append(f"Полученный текст: {response.text}")
                    log_container.code("\n".join(log_text), language="text")
                    raise Exception("Неверный формат ответа от API")
            else:
                error_msg = f"Ошибка API: {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f" - {error_details}"
                except:
                    error_msg += f" - {response.text}"
                log_text.append(f"Ошибка API: {error_msg}")
                log_container.code("\n".join(log_text), language="text")
                raise Exception(error_msg)

            # Отображаем рекомендации
            st.markdown(recommendations)

        except Exception as e:
            st.error(f"Ошибка при генерации рекомендаций: {str(e)}")
            log_text.append(f"Ошибка при генерации текста: {str(e)}")
            log_container.code("\n".join(log_text), language="text")

# Футер
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p>Создано с ❤️ для HR-специалистов</p>
    <p>Версия 1.0.0</p>
</div>
""",
    unsafe_allow_html=True,
)
