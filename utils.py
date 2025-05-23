import re

import PyPDF2
import requests
import streamlit as st
from docx import Document


def extract_skills(text):
    """Извлекает навыки из текста"""
    # Разбиваем текст на слова
    words = re.findall(r"\b\w+\b", text.lower())

    # Фильтруем короткие слова и стоп-слова
    stop_words = {
        # Общие стоп-слова
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
        # ... остальные стоп-слова ...
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
            doc = Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        else:
            st.error("Неподдерживаемый формат файла")
            return ""
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {str(e)}")
        return ""


def generate_text(prompt, max_tokens=1000, temperature=0.7):
    """
    Генерирует текст с помощью Kaggle API
    """
    try:
        # Получаем URL из конфигурации
        base_url = st.secrets["api"]["kaggle_url"]

        # Формируем URL для запроса
        api_url = f"{base_url}/generate"

        # Подготавливаем данные для запроса
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Отправляем запрос
        response = requests.post(
            api_url,
            json=payload,
            timeout=30,  # Увеличиваем таймаут до 30 секунд
        )

        # Проверяем статус ответа
        if response.status_code == 200:
            result = response.json()
            if "generated_text" in result:
                return result["generated_text"]
            else:
                raise Exception("Неверный формат ответа от API")
        else:
            raise Exception(f"Ошибка API: {response.status_code} - {response.text}")

    except requests.exceptions.Timeout:
        raise Exception("Превышено время ожидания ответа от API")
    except requests.exceptions.ConnectionError:
        raise Exception("Ошибка подключения к API")
    except Exception as e:
        raise Exception(f"Ошибка при генерации текста: {str(e)}")


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

        # Группируем навыки по категориям
        missing_tech = {
            skill
            for skill in missing_skills
            if skill
            in {
                "python",
                "numpy",
                "pandas",
                "matplotlib",
                "seaborn",
                "plotly",
                "docker",
                "chromadb",
                "langchain",
                "llm",
                "data",
                "scientist",
            }
        }
        missing_other = missing_skills - missing_tech

        extra_tech = {
            skill
            for skill in extra_skills
            if skill
            in {
                "python",
                "numpy",
                "pandas",
                "matplotlib",
                "seaborn",
                "plotly",
                "docker",
                "chromadb",
                "langchain",
                "llm",
                "data",
                "scientist",
            }
        }
        extra_other = extra_skills - extra_tech

        return {
            "missing_tech": missing_tech,
            "missing_other": missing_other,
            "extra_tech": extra_tech,
            "extra_other": extra_other,
            "similarity": similarity,
        }

    except Exception as e:
        st.error(f"Ошибка при анализе навыков: {str(e)}")
        return {
            "missing_tech": set(),
            "missing_other": set(),
            "extra_tech": set(),
            "extra_other": set(),
            "similarity": 0.0,
        }
