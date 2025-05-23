import os
import re

import nltk
import numpy as np
import PyPDF2
import requests
import streamlit as st
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, pipeline

# Отключаем предупреждения transformers
logging.set_verbosity_error()

# Загружаем необходимые ресурсы NLTK
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("wordnet", quiet=True)

# Путь для кэширования модели
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

# Инициализация модели для многоязычного анализа
model = None

# API URL для Hugging Face
API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}


# Инициализация LLM pipeline
@st.cache_resource
def get_llm_pipeline():
    """Получает LLM pipeline с кэшированием"""
    try:
        with st.spinner("Загрузка LLM модели..."):
            # Загружаем модель и токенизатор
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

            # Загружаем модель с оптимизацией памяти
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                device_map="auto",
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "2GB"},  # Ограничиваем использование памяти
            )

            # Создаем pipeline с оптимизацией
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                max_memory={0: "2GB"},
            )
    except Exception as e:
        st.error(f"Ошибка при загрузке LLM модели: {str(e)}")
        return None


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
        "python3",
        "java script",
        "type script",
        "c plus plus",
        "c sharp",
        "golang",
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
        "asp net",
        "rails",
        "react",
        "angular",
        "vue",
        "node.js",
        "node js",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "scikit-learn",
        "scikit learn",
        "keras",
        "spark",
        "hadoop",
        "langchain",
        "chromadb",
        "transformers",
        "huggingface",
        "hugging face",
        "streamlit",
        "gradio",
        "bootstrap",
        "jquery",
        "redux",
        "vue.js",
        "vue js",
        "next.js",
        "next js",
        "nuxt.js",
        "nuxt js",
    },
    # Базы данных
    "databases": {
        "sql",
        "nosql",
        "mongodb",
        "postgresql",
        "postgres",
        "mysql",
        "oracle",
        "redis",
        "elasticsearch",
        "elastic search",
        "cassandra",
        "neo4j",
        "dynamodb",
        "dynamo db",
        "pinecone",
        "weaviate",
        "qdrant",
        "mssql",
        "ms sql",
        "sqlite",
        "sql server",
    },
    # DevOps и инструменты
    "devops": {
        "docker",
        "kubernetes",
        "k8s",
        "aws",
        "azure",
        "gcp",
        "google cloud",
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
        "ci/cd",
        "cicd",
        "continuous integration",
        "continuous deployment",
        "github",
        "bitbucket",
        "svn",
        "mercurial",
    },
    # Методологии
    "methodologies": {
        "agile",
        "scrum",
        "kanban",
        "waterfall",
        "devops",
        "ci/cd",
        "cicd",
        "continuous integration",
        "continuous deployment",
        "tdd",
        "test driven development",
        "bdd",
        "behavior driven development",
        "xp",
        "extreme programming",
        "lean",
        "six sigma",
    },
    # AI/ML инструменты
    "ai_ml": {
        "langchain",
        "chromadb",
        "transformers",
        "huggingface",
        "hugging face",
        "pinecone",
        "weaviate",
        "qdrant",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "scikit learn",
        "keras",
        "opencv",
        "open cv",
        "nltk",
        "spacy",
        "gensim",
        "fastai",
        "fast ai",
        "xgboost",
        "lightgbm",
        "catboost",
        "pytorch lightning",
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


@st.cache_data
def extract_text_from_file(file):
    """Извлекает текст из PDF или DOCX файла"""
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        raise ValueError("Неподдерживаемый формат файла")


@st.cache_data
def extract_text_from_pdf(file):
    """Извлекает текст из PDF файла"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


@st.cache_data
def extract_text_from_docx(file):
    """Извлекает текст из DOCX файла"""
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


@st.cache_data
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


@st.cache_data
def calculate_similarity(text1: str, text2: str) -> float:
    """Вычисляет семантическую схожесть между двумя текстами"""
    try:
        model = get_model()
        if model is None:
            return 0.0

        # Разбиваем тексты на предложения и берем только первые 10 для ускорения
        sentences1 = sent_tokenize(text1)[:10]
        sentences2 = sent_tokenize(text2)[:10]

        # Получаем эмбеддинги для всех предложений сразу
        embeddings1 = model.encode(sentences1, show_progress_bar=False)
        embeddings2 = model.encode(sentences2, show_progress_bar=False)

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

        return float(similarity * 100)
    except Exception as e:
        print(f"Ошибка при вычислении схожести: {str(e)}")
        return 0.0


@st.cache_data
def extract_skills(text):
    """Извлекает навыки из текста"""
    # Разбиваем текст на предложения
    sentences = sent_tokenize(text.lower())
    skills = set()

    # Собираем все ключевые слова из TECH_SKILLS в один плоский список
    all_tech_keywords = set()
    for group in TECH_SKILLS.values():
        all_tech_keywords.update([kw.lower() for kw in group])

    # Добавляем отладочную информацию
    st.write("### Отладочная информация")
    st.write(f"Текст для анализа (первые 200 символов): {text[:200]}...")
    st.write(f"Количество предложений: {len(sentences)}")
    st.write(f"Количество навыков для поиска: {len(all_tech_keywords)}")

    found_skills = []
    for sentence in sentences:
        for tech in all_tech_keywords:
            if tech in sentence:
                skills.add(tech)
                found_skills.append((tech, sentence))

    if found_skills:
        st.write("Найденные навыки:")
        for skill, sentence in found_skills:
            st.write(f"- {skill} (в предложении: {sentence[:100]}...)")
    else:
        st.write("Навыки не найдены")

    return skills


@st.cache_data
def extract_responsibilities(text):
    """Извлекает обязанности из текста"""
    responsibilities = []
    sentences = sent_tokenize(text.lower())
    model = get_model()

    if model is None:
        print("Ошибка: модель не была загружена")
        return responsibilities

    try:
        # Фильтруем предложения по ключевым словам
        relevant_sentences = [
            sentence
            for sentence in sentences
            if any(keyword in sentence for keyword in RESPONSIBILITY_KEYWORDS)
        ]

        if not relevant_sentences:
            return responsibilities

        # Получаем эмбеддинги для всех предложений сразу
        sentence_embeddings = model.encode(relevant_sentences, show_progress_bar=False)

        # Проверяем дубликаты
        for i, (sentence, embedding) in enumerate(
            zip(relevant_sentences, sentence_embeddings)
        ):
            is_duplicate = False
            for j in range(i):
                similarity = cosine_similarity(
                    embedding.reshape(1, -1), sentence_embeddings[j].reshape(1, -1)
                )[0][0]
                if similarity > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                responsibilities.append(sentence)

    except Exception as e:
        print(f"Ошибка при извлечении обязанностей: {str(e)}")
        return responsibilities

    return responsibilities


def extract_stack_from_text(text):
    """Извлекает стек технологий из текста"""
    try:
        # Разбиваем текст на предложения
        sentences = sent_tokenize(text.lower())

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
            "используем:",
            "используем:",
            "работаем с",
            "работаем с:",
            "работаем с:",
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

        # Разбиваем на слова и очищаем
        words = word_tokenize(stack_text)
        words = [w for w in words if len(w) > 2]  # Убираем короткие слова

        return stack_text, words
    except Exception as e:
        st.error(f"Ошибка при извлечении стека: {str(e)}")
        return text, []


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


@st.cache_data
def get_detailed_analysis(job_description, resume_text):
    """Получает детальный анализ резюме"""
    # Ключевые слова для поиска секций
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
    # Собираем все заголовки и их позиции
    found_headers = []
    for section, keywords in sections.items():
        for keyword in keywords:
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


@st.cache_data
def query_llm(prompt):
    """Отправляет запрос к LLM модели через API"""
    try:
        # Форматируем промпт
        formatted_prompt = f"""HR Analysis Task:
{prompt}

Analysis:"""

        # Проверяем доступность модели
        health_check = requests.get(API_URL, headers=HEADERS)
        if health_check.status_code == 503:
            return (
                "Модель загружается. Пожалуйста, подождите немного и попробуйте снова."
            )

        # Отправляем запрос к API
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "inputs": formatted_prompt,
                "parameters": {
                    "max_length": 150,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                    "return_full_text": False,
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()[0]["generated_text"]
            result = result.replace(formatted_prompt, "").strip()
            return result
        elif response.status_code == 503:
            return (
                "Модель загружается. Пожалуйста, подождите немного и попробуйте снова."
            )
        else:
            st.error(f"Ошибка API: {response.status_code}")
            return "Произошла ошибка при анализе. Пожалуйста, попробуйте позже."

    except requests.exceptions.Timeout:
        return "Превышено время ожидания ответа. Пожалуйста, попробуйте позже."
    except Exception as e:
        st.error(f"Ошибка при обращении к API: {str(e)}")
        return "Произошла ошибка при анализе. Пожалуйста, попробуйте позже."
