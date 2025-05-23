import os
import re


import docx
import PyPDF2
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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


def simple_tokenize(text):
    """Простая токенизация текста на предложения и слова"""
    # Разбиваем на предложения по знакам препинания
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Разбиваем на слова
    words = []
    for sentence in sentences:
        # Разбиваем по пробелам и знакам препинания
        sentence_words = re.findall(r"\b\w+\b", sentence.lower())
        words.extend(sentence_words)

    return sentences, words


def extract_stack_from_text(text):
    """Извлекает стек технологий из текста"""
    try:
        # Разбиваем текст на предложения и слова
        sentences, words = simple_tokenize(text.lower())

        # Ищем упоминания стека
        stack_indicators = [
            "стек",
            "stack",
            "технологии",
            "technologies",
            "инструменты",
            "tools",
            "используем",
            "используем:",
            "работаем с",
            "работаем с:",
            "требования",
            "requirements",
            "требования:",
            "requirements:",
            "навыки",
            "skills",
            "навыки:",
            "skills:",
        ]

        stack_sentences = []
        for sentence in sentences:
            if any(indicator in sentence for indicator in stack_indicators):
                stack_sentences.append(sentence)

        # Если не нашли явных указаний на стек, берем все предложения
        if not stack_sentences:
            stack_sentences = sentences

        # Объединяем все предложения со стеком
        stack_text = " ".join(stack_sentences)

        # Очищаем слова
        words = [w for w in words if len(w) > 2]  # Убираем короткие слова

        return stack_text, words
    except Exception as e:
        st.error(f"Ошибка при извлечении стека: {str(e)}")
        return text, []


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
    """Анализирует соответствие стека технологий"""
    try:
        # Извлекаем стек из описания вакансии и резюме
        job_stack_text, job_words = extract_stack_from_text(job_description)
        resume_stack_text, resume_words = extract_stack_from_text(resume_text)

        # Получаем модель для анализа
        model = get_model()
        if model is None:
            return {
                "missing_skills": set(),
                "extra_skills": set(),
                "similarity": 0.0,
                "job_stack": job_stack_text,
                "resume_stack": resume_stack_text,
            }

        # Получаем эмбеддинги для текстов стека
        job_embedding = model.encode(job_stack_text)
        resume_embedding = model.encode(resume_stack_text)

        # Вычисляем схожесть стеков
        similarity = cosine_similarity(
            job_embedding.reshape(1, -1), resume_embedding.reshape(1, -1)
        )[0][0]

        # Находим уникальные слова в каждом стеке
        job_unique = set(job_words) - set(resume_words)
        resume_unique = set(resume_words) - set(job_words)

        # Отладочная информация
        st.write("### Отладочная информация")
        st.write("#### Стек из вакансии:")
        st.write(job_stack_text)
        st.write("#### Стек из резюме:")
        st.write(resume_stack_text)
        st.write(f"#### Схожесть стеков: {similarity:.2%}")

        if job_unique:
            st.write("#### Уникальные технологии в вакансии:")
            for tech in sorted(job_unique):
                st.write(f"- {tech}")

        if resume_unique:
            st.write("#### Уникальные технологии в резюме:")
            for tech in sorted(resume_unique):
                st.write(f"- {tech}")

        return {
            "missing_skills": job_unique,
            "extra_skills": resume_unique,
            "similarity": similarity,
            "job_stack": job_stack_text,
            "resume_stack": resume_stack_text,
        }

    except Exception as e:
        st.error(f"Ошибка при анализе навыков: {str(e)}")
        return {
            "missing_skills": set(),
            "extra_skills": set(),
            "similarity": 0.0,
            "job_stack": "",
            "resume_stack": "",
        }
