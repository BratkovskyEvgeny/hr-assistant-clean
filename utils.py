import os
import re

import nltk
import numpy as np
import PyPDF2
import streamlit as st
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging, pipeline

# Отключаем предупреждения transformers
logging.set_verbosity_error()

# Загружаем необходимые ресурсы NLTK
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Путь для кэширования модели
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

# Инициализация модели для многоязычного анализа
model = None


# Инициализация LLM pipeline
@st.cache_resource
def get_llm_pipeline():
    """Получает LLM pipeline с кэшированием"""
    try:
        with st.spinner("Загрузка LLM модели..."):
            return pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.3",
                device_map="auto",
                torch_dtype="auto",
            )
    except Exception as e:
        st.error(f"Ошибка при загрузке LLM модели: {str(e)}")
        return None


API_URL = "https://api-inference.huggingface.co/models/gpt2-medium"
headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}


@st.cache_resource
def get_model() -> SentenceTransformer:
    """Получает модель с кэшированием"""
    try:
        with st.spinner("Загрузка модели для анализа текста..."):
            # Проверяем наличие кэшированной модели
            if os.path.exists(CACHE_DIR):
                return SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    cache_folder=CACHE_DIR,
                    use_auth_token=True,
                )

            # Если кэша нет, создаем директорию и загружаем модель
            os.makedirs(CACHE_DIR, exist_ok=True)
            return SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                cache_folder=CACHE_DIR,
                use_auth_token=True,
            )
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        return None


# Словари технических навыков
TECH_SKILLS = {
    # Языки программирования
    "languages": {
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "php",
        "ruby",
        "go",
        "rust",
        "swift",
        "kotlin",
        "scala",
        "r",
        "matlab",
    },
    # Фреймворки и библиотеки
    "frameworks": {
        "django",
        "flask",
        "fastapi",
        "spring",
        "laravel",
        "express",
        "asp.net",
        "rails",
        "react",
        "angular",
        "vue",
        "node.js",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "scikit-learn",
        "keras",
        "spark",
        "hadoop",
        "langchain",
        "chromadb",
        "transformers",
        "huggingface",
        "streamlit",
        "gradio",
    },
    # Базы данных
    "databases": {
        "sql",
        "nosql",
        "mongodb",
        "postgresql",
        "mysql",
        "oracle",
        "redis",
        "elasticsearch",
        "cassandra",
        "neo4j",
        "dynamodb",
        "pinecone",
        "weaviate",
        "qdrant",
    },
    # DevOps и инструменты
    "devops": {
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "linux",
        "unix",
        "git",
        "jenkins",
        "gitlab",
        "jira",
        "confluence",
        "ansible",
        "terraform",
        "prometheus",
        "grafana",
    },
    # Методологии
    "methodologies": {
        "agile",
        "scrum",
        "kanban",
        "waterfall",
        "devops",
        "ci/cd",
    },
    # AI/ML инструменты
    "ai_ml": {
        "langchain",
        "chromadb",
        "transformers",
        "huggingface",
        "pinecone",
        "weaviate",
        "qdrant",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "keras",
    },
}

# Ключевые слова для определения обязанностей
RESPONSIBILITY_KEYWORDS = {
    "разработка",
    "development",
    "разработать",
    "develop",
    "создание",
    "creation",
    "создать",
    "create",
    "внедрение",
    "implementation",
    "внедрить",
    "implement",
    "оптимизация",
    "optimization",
    "оптимизировать",
    "optimize",
    "поддержка",
    "maintenance",
    "поддерживать",
    "maintain",
    "тестирование",
    "testing",
    "тестировать",
    "test",
    "анализ",
    "analysis",
    "анализировать",
    "analyze",
    "управление",
    "management",
    "управлять",
    "manage",
    "координация",
    "coordination",
    "координировать",
    "coordinate",
}


def extract_text_from_file(file):
    """Извлекает текст из PDF или DOCX файла"""
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        raise ValueError("Неподдерживаемый формат файла")


def extract_text_from_pdf(file):
    """Извлекает текст из PDF файла"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    """Извлекает текст из DOCX файла"""
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def preprocess_text(text):
    """Предобработка текста"""
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление специальных символов
    text = re.sub(r"[^\w\s]", " ", text)

    # Токенизация
    tokens = word_tokenize(text)

    # Удаление стоп-слов
    stop_words = set(stopwords.words("russian") + stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    return " ".join(tokens)


def calculate_similarity(text1: str, text2: str) -> float:
    """Вычисляет семантическую схожесть между двумя текстами"""
    try:
        model = get_model()
        if model is None:
            return 0.0

        # Разбиваем тексты на предложения
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)

        # Получаем эмбеддинги
        embeddings1 = model.encode(sentences1)
        embeddings2 = model.encode(sentences2)

        # Вычисляем косинусное сходство
        similarity = np.mean(
            [
                np.max(
                    [
                        np.dot(emb1, emb2)
                        / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        for emb2 in embeddings2
                    ]
                )
                for emb1 in embeddings1
            ]
        )

        # Преобразуем в проценты
        return float(similarity * 100)
    except Exception as e:
        print(f"Ошибка при вычислении схожести: {str(e)}")
        return 0.0


def extract_skills(text):
    """Извлекает навыки из текста (поиск по подстроке, регистронезависимо)"""
    # Разбиваем текст на предложения
    sentences = sent_tokenize(text.lower())
    skills = set()
    stop_words = {
        "опыт",
        "experience",
        "работа",
        "work",
        "разработка",
        "development",
        "создание",
        "creation",
        "внедрение",
        "implementation",
        "использование",
        "using",
        "знание",
        "knowledge",
        "умение",
        "ability",
        "навык",
        "skill",
        "требование",
        "requirement",
        "обязанность",
        "responsibility",
        "задача",
        "task",
        "проект",
        "project",
        "верим",
        "делать",
        "жизнь",
        "инструмент",
        "интеллект",
        "который",
        "легче",
        "лучше",
        "нам",
        "наш",
        "помогает",
        "усиливает",
        "что",
        "это",
        "агентов",
        "баз",
        "будут",
        "вакансии",
        "валюта",
        "векторизованных",
        "владельцев",
        "внедрить",
        "выполнять",
        "достаточно",
        "драгоценные",
        "задач",
        "задачу",
        "знаний",
        "конкретного",
        "которые",
        "лет",
        "металлы",
        "название",
        "обязательно",
        "опыта",
        "организация",
        "пайплайна",
        "перед",
        "полученного",
        "продуктов",
        "промышленное",
        "процесса",
        "процессе",
        "работу",
        "ранжирование",
        "реализация",
        "роли",
        "сильных",
        "слабых",
        "собой",
        "создания",
        "создать",
        "ставим",
        "сторон",
        "требования",
        "уровне",
        "цепочек",
        "часть",
        "эффективные",
    }
    # Собираем все ключевые слова из TECH_SKILLS в один плоский список
    all_tech_keywords = set()
    for group in TECH_SKILLS.values():
        all_tech_keywords.update([kw.lower() for kw in group])
    for sentence in sentences:
        for tech in all_tech_keywords:
            if tech in sentence and tech not in stop_words:
                skills.add(tech)
    return skills


def extract_responsibilities(text):
    """Извлекает обязанности из текста"""
    responsibilities = []
    sentences = sent_tokenize(text.lower())
    model = get_model()

    if model is None:
        print("Ошибка: модель не была загружена")
        return responsibilities

    try:
        for sentence in sentences:
            # Проверяем, содержит ли предложение ключевые слова обязанностей
            if any(keyword in sentence for keyword in RESPONSIBILITY_KEYWORDS):
                # Получаем эмбеддинг предложения
                sentence_embedding = model.encode(sentence)

                # Проверяем, не является ли это дубликатом
                is_duplicate = False
                for existing_resp in responsibilities:
                    existing_embedding = model.encode(existing_resp)
                    similarity = cosine_similarity(
                        sentence_embedding.reshape(1, -1),
                        existing_embedding.reshape(1, -1),
                    )[0][0]
                    if similarity > 0.8:  # Порог схожести
                        is_duplicate = True
                        break

                if not is_duplicate:
                    responsibilities.append(sentence)
    except Exception as e:
        print(f"Ошибка при извлечении обязанностей: {str(e)}")
        return responsibilities

    return responsibilities


def analyze_skills(job_description, resume_text):
    """Анализирует отсутствующие навыки и опыт"""
    # Извлекаем навыки из описания вакансии и резюме
    job_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume_text)

    # Извлекаем обязанности
    job_responsibilities = extract_responsibilities(job_description)
    resume_responsibilities = extract_responsibilities(resume_text)

    # Анализируем отсутствующие навыки
    missing_skills = job_skills - resume_skills

    # Анализируем отсутствующий опыт
    missing_experience = []
    model = get_model()

    if model is None:
        print("Ошибка: модель не была загружена")
        return {
            "missing_skills": missing_skills,
            "missing_experience": missing_experience,
        }

    try:
        for job_resp in job_responsibilities:
            job_resp_embedding = model.encode(job_resp)
            max_similarity = 0
            for resume_resp in resume_responsibilities:
                resume_resp_embedding = model.encode(resume_resp)
                similarity = cosine_similarity(
                    job_resp_embedding.reshape(1, -1),
                    resume_resp_embedding.reshape(1, -1),
                )[0][0]
                max_similarity = max(max_similarity, similarity)

            if max_similarity < 0.5:  # Порог схожести
                missing_experience.append(job_resp)
    except Exception as e:
        print(f"Ошибка при анализе опыта: {str(e)}")
        return {
            "missing_skills": missing_skills,
            "missing_experience": missing_experience,
        }

    return {"missing_skills": missing_skills, "missing_experience": missing_experience}


def get_detailed_analysis(job_description, resume_text):
    """Получает детальный анализ резюме и возвращает найденные заголовки для отладки"""
    # Ключевые слова для поиска секций (регистронезависимо, с учетом спецсимволов)
    sections = {
        "experience": ["опыт работы", "experience", "work experience"],
        "education": ["образование", "education"],
        "skills": [
            "навыки",
            "skills",
            "технические навыки",
            "знания и навыки",
            "основной стек",
        ],
    }

    analysis = {}
    model = get_model()
    if model is None:
        print("Ошибка: модель не была загружена")
        return analysis

    resume_text_lower = resume_text.lower()
    # Собираем все заголовки и их позиции с учетом запятых, пробелов и переносов
    found_headers = []
    for section, keywords in sections.items():
        for keyword in keywords:
            # Регулярка: ищет заголовок как отдельное слово, с возможными пробелами, запятыми, переносами
            pattern = r"[\s,\n\r]*" + re.escape(keyword.lower()) + r"[\s,\n\r]*"
            for match in re.finditer(pattern, resume_text_lower):
                found_headers.append(
                    {
                        "section": section,
                        "keyword": keyword,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
    # Сортируем по позиции
    found_headers = sorted(found_headers, key=lambda x: x["start"])

    # Извлекаем текст между заголовками
    section_texts = {k: "" for k in sections.keys()}
    for i, header in enumerate(found_headers):
        section = header["section"]
        start = header["end"]
        end = (
            found_headers[i + 1]["start"]
            if i + 1 < len(found_headers)
            else len(resume_text)
        )
        section_texts[section] += resume_text[start:end].strip() + "\n"

    # Анализируем каждую секцию
    try:
        job_embedding = model.encode(job_description)
        for section in sections.keys():
            section_text = section_texts[section]
            if section_text.strip():
                section_embedding = model.encode(section_text)
                similarity = cosine_similarity(
                    section_embedding.reshape(1, -1), job_embedding.reshape(1, -1)
                )[0][0]
                section_skills = extract_skills(section_text)
                section_responsibilities = extract_responsibilities(section_text)
                analysis[section] = {
                    "text": section_text,
                    "relevance": float(similarity * 100),
                    "skills": list(section_skills),
                    "responsibilities": section_responsibilities,
                }
            else:
                analysis[section] = {
                    "text": "",
                    "relevance": 0.0,
                    "skills": [],
                    "responsibilities": [],
                }
        # Общий процент соответствия
        if analysis:
            total_relevance = sum(section["relevance"] for section in analysis.values())
            average_relevance = total_relevance / len(analysis)
            analysis["overall_match"] = float(average_relevance)
        # Для отладки: возвращаем найденные заголовки
        analysis["_debug_headers"] = found_headers
    except Exception as e:
        print(f"Ошибка при детальном анализе: {str(e)}")
        return analysis
    return analysis


def query_llm(prompt):
    """Отправляет запрос к LLM модели"""
    try:
        pipe = get_llm_pipeline()
        if pipe is None:
            return "Не удалось загрузить LLM модель. Пожалуйста, попробуйте позже."

        # Форматируем промпт для Mistral
        formatted_prompt = f"""<s>[INST] Ты — HR-ассистент. Проанализируй следующую информацию:

{prompt}

Ответь структурировано. [/INST]</s>"""

        # Генерируем ответ
        response = pipe(
            formatted_prompt,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        # Извлекаем и очищаем результат
        result = response[0]["generated_text"]
        result = result.replace(formatted_prompt, "").strip()
        return result

    except Exception as e:
        st.error(f"Ошибка при обращении к LLM: {str(e)}")
        return "Произошла ошибка при анализе. Пожалуйста, попробуйте позже."
