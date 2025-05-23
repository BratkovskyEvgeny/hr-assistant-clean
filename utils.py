import os
import re

import PyPDF2
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# Путь для кэширования модели
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

# Инициализация модели для многоязычного анализа
model = None

# API URL для Hugging Face
API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}


@st.cache_resource
def get_model() -> SentenceTransformer:
    """Получает модель с кэшированием"""
    try:
        with st.spinner("Загрузка модели для анализа текста..."):
            return SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                cache_folder=CACHE_DIR,
                use_auth_token=True,
            )
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        return None


def extract_skills(text):
    """Извлекает навыки из текста"""
    # Разбиваем текст на слова
    words = re.findall(r"\b\w+\b", text.lower())

    # Фильтруем короткие слова и стоп-слова
    stop_words = {
        "и",
        "в",
        "во",
        "не",
        "что",
        "с",
        "со",
        "как",
        "а",
        "то",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "по",
        "только",
        "ее",
        "мне",
        "было",
        "вот",
        "от",
        "меня",
        "еще",
        "нет",
        "о",
        "из",
        "ему",
        "теперь",
        "когда",
        "даже",
        "ну",
        "вдруг",
        "ли",
        "если",
        "уже",
        "или",
        "ни",
        "быть",
        "был",
        "него",
        "до",
        "вас",
        "нибудь",
        "опять",
        "уж",
        "вам",
        "ведь",
        "там",
        "потом",
        "себя",
        "ничего",
        "ей",
        "может",
        "они",
        "тут",
        "где",
        "есть",
        "надо",
        "ней",
        "для",
        "мы",
        "тебя",
        "их",
        "чем",
        "была",
        "сам",
        "чтоб",
        "без",
        "будто",
        "чего",
        "раз",
        "тоже",
        "себе",
        "под",
        "будет",
        "ж",
        "тогда",
        "кто",
        "этот",
        "того",
        "потому",
        "этого",
        "какой",
        "совсем",
        "ним",
        "здесь",
        "этом",
        "один",
        "почти",
        "мой",
        "тем",
        "чтобы",
        "нее",
        "сейчас",
        "были",
        "куда",
        "зачем",
        "всех",
        "никогда",
        "можно",
        "при",
        "на",
        "об",
        "я",
        "а",
        "б",
        "в",
        "г",
        "д",
        "е",
        "ж",
        "з",
        "и",
        "й",
        "к",
        "л",
        "м",
        "н",
        "о",
        "п",
        "р",
        "с",
        "т",
        "у",
        "ф",
        "х",
        "ц",
        "ч",
        "ш",
        "щ",
        "ъ",
        "ы",
        "ь",
        "э",
        "ю",
        "я",
    }

    # Фильтруем слова
    skills = {word for word in words if len(word) > 2 and word not in stop_words}

    return skills


def extract_text_from_file(file):
    """Извлекает текст из загруженного файла (PDF или DOCX)"""
    try:
        if file.type == "application/pdf":
            # Читаем PDF
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif (
            file.type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            # Читаем DOCX
            doc = docx.Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        else:
            st.error("Неподдерживаемый формат файла")
            return ""
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {str(e)}")
        return ""


def query_llm(prompt):
    """Отправляет запрос к LLM через Hugging Face API"""
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt, "parameters": {"return_full_text": False}},
        )
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    except Exception as e:
        st.error(f"Ошибка при запросе к LLM: {str(e)}")
        return "Не удалось получить ответ от модели"


@st.cache_data
def analyze_skills(job_description, resume_text):
    """Анализирует соответствие навыков"""
    try:
        # Извлекаем навыки из описания вакансии и резюме
        job_skills = extract_skills(job_description)
        resume_skills = extract_skills(resume_text)

        # Находим отсутствующие и лишние навыки
        missing_skills = job_skills - resume_skills
        extra_skills = resume_skills - job_skills

        # Вычисляем схожесть
        similarity = (
            len(job_skills & resume_skills) / len(job_skills) if job_skills else 0.0
        )

        # Отладочная информация
        st.write("### Отладочная информация")
        st.write("#### Навыки из вакансии:")
        for skill in sorted(job_skills):
            st.write(f"- {skill}")
        st.write("#### Навыки из резюме:")
        for skill in sorted(resume_skills):
            st.write(f"- {skill}")
        st.write(f"#### Схожесть навыков: {similarity:.2%}")

        return {
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "similarity": similarity,
            "job_skills": job_skills,
            "resume_skills": resume_skills,
        }

    except Exception as e:
        st.error(f"Ошибка при анализе навыков: {str(e)}")
        return {
            "missing_skills": set(),
            "extra_skills": set(),
            "similarity": 0.0,
            "job_skills": set(),
            "resume_skills": set(),
        }
