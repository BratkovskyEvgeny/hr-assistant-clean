import base64
import json

import PyPDF2
import requests
import streamlit as st
from docx import Document


def extract_skills(text):
    """Извлекает навыки из текста"""
    # Список общих слов для исключения
    common_words = {
        "организация",
        "организации",
        "организовать",
        "организованный",
        "управление",
        "управлять",
        "управляющий",
        "управляемый",
        "работа",
        "работать",
        "рабочий",
        "работающий",
        "процесс",
        "процессы",
        "процессный",
        "процессный",
        "система",
        "системы",
        "системный",
        "системный",
        "разработка",
        "разрабатывать",
        "разработанный",
        "внедрение",
        "внедрять",
        "внедренный",
        "реализация",
        "реализовывать",
        "реализованный",
        "опыт",
        "опытный",
        "опыт работы",
        "знание",
        "знания",
        "знать",
        "знающий",
        "навык",
        "навыки",
        "навычный",
        "умение",
        "умения",
        "уметь",
        "умеющий",
    }

    # Список технических навыков и фраз
    tech_skills = {
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
        "sql",
        "postgresql",
        "clickhouse",
        "greenplum",
        "mariadb",
        "hadoop",
        "hive",
        "spark",
        "kafka",
        "nifi",
        "hdfs",
        "aws",
        "git",
        "jira",
        "confluence",
        "grafana",
        "redis",
        "django",
        "flask",
        "fastapi",
        "rest",
        "soap",
        "api",
        "scikit-learn",
        "pytorch",
        "tensorflow",
        "keras",
        "devops",
        "ci/cd",
        "jenkins",
        "kubernetes",
        "agile",
        "scrum",
        "kanban",
        "power bi",
        "tableau",
        "superset",
        "datalens",
        "machine learning",
        "deep learning",
        "nlp",
        "computer vision",
        "data engineering",
        "data analysis",
        "data science",
        "business intelligence",
        "etl",
        "data warehouse",
        "big data",
        "cloud computing",
        "microservices",
    }

    # Список общих навыков и фраз
    soft_skills = {
        # Коммуникационные навыки
        "деловая переписка",
        "публичные выступления",
        "презентация",
        "ведение переговоров",
        "межличностное общение",
        "навыки убеждения",
        "активное слушание",
        # Управленческие навыки
        "управление проектами",
        "управление командой",
        "делегирование задач",
        "стратегическое планирование",
        "управление рисками",
        "управление изменениями",
        # Аналитические навыки
        "аналитическое мышление",
        "критическое мышление",
        "решение проблем",
        "принятие решений",
        "системное мышление",
        "стратегическое мышление",
        # Личностные качества
        "самоорганизация",
        "самообучение",
        "адаптивность",
        "стрессоустойчивость",
        "инициативность",
        "ответственность",
        "креативность",
        # Профессиональные навыки
        "менторство",
        "коучинг",
        "фасилитация",
        "модерация",
        "управление конфликтами",
        "управление временем",
        "управление качеством",
    }

    # Приводим текст к нижнему регистру и удаляем лишние пробелы
    text = " ".join(text.lower().split())

    # Находим технические навыки
    found_tech_skills = set()
    for skill in tech_skills:
        if skill in text:
            found_tech_skills.add(skill)

    # Находим общие навыки
    found_soft_skills = set()
    for skill in soft_skills:
        if skill in text:
            found_soft_skills.add(skill)

    # Объединяем все найденные навыки
    all_skills = found_tech_skills | found_soft_skills

    # Удаляем общие слова из результата
    all_skills = {skill for skill in all_skills if skill not in common_words}

    return all_skills


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
        # Получаем URL и учетные данные из конфигурации
        api_url = st.secrets["api"]["kaggle_url"]
        username = st.secrets["kaggle"]["username"]
        key = st.secrets["kaggle"]["key"]

        # Подготавливаем данные для запроса
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model": "distilgpt2",  # Указываем модель из ноутбука
            }
        }

        # Формируем заголовки авторизации
        auth = f"{username}:{key}"
        auth_bytes = auth.encode("ascii")
        base64_auth = base64.b64encode(auth_bytes).decode("ascii")

        # Логируем запрос (без учетных данных)
        st.write("Отправляем запрос к API...")
        st.write(f"URL: {api_url}")
        st.write(f"Payload: {payload}")
        st.write(
            "Headers: {'Content-Type': 'application/json', 'Authorization': 'Basic ***'}"
        )

        # Отправляем запрос
        try:
            response = requests.post(
                api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Basic {base64_auth}",
                    "Accept": "application/json",
                },
                timeout=30,
                verify=True,  # Проверяем SSL сертификат
            )
        except requests.exceptions.SSLError as e:
            st.error(f"Ошибка SSL: {str(e)}")
            raise Exception("Ошибка SSL при подключении к API")
        except requests.exceptions.ConnectionError as e:
            st.error(f"Ошибка подключения: {str(e)}")
            raise Exception("Не удалось подключиться к API")
        except requests.exceptions.Timeout as e:
            st.error(f"Таймаут: {str(e)}")
            raise Exception("Превышено время ожидания ответа от API")

        # Логируем ответ
        st.write(f"Статус ответа: {response.status_code}")
        st.write(f"Заголовки ответа: {dict(response.headers)}")
        st.write(f"Текст ответа: {response.text}")

        # Проверяем статус ответа
        if response.status_code == 200:
            try:
                result = response.json()
                if "output" in result and "text" in result["output"]:
                    return result["output"]["text"]
                elif "generated_text" in result:
                    return result["generated_text"]
                elif "text" in result:
                    return result["text"]
                else:
                    error_msg = result.get("message", "Неизвестная ошибка")
                    raise Exception(f"Ошибка в ответе API: {error_msg}")
            except json.JSONDecodeError as e:
                st.error(f"Ошибка при разборе JSON: {str(e)}")
                raise Exception("Неверный формат ответа от API")
        else:
            error_msg = f"Ошибка API: {response.status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details}"
            except:
                error_msg += f" - {response.text}"
            raise Exception(error_msg)

    except Exception as e:
        st.error(f"Ошибка при генерации текста: {str(e)}")
        raise


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
        tech_skills = {
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
            "sql",
            "postgresql",
            "clickhouse",
            "greenplum",
            "mariadb",
            "hadoop",
            "hive",
            "spark",
            "kafka",
            "nifi",
            "hdfs",
            "aws",
            "git",
            "jira",
            "confluence",
            "grafana",
            "redis",
            "django",
            "flask",
            "fastapi",
            "rest",
            "soap",
            "api",
            "scikit-learn",
            "pytorch",
            "tensorflow",
            "keras",
            "devops",
            "ci/cd",
            "jenkins",
            "kubernetes",
            "agile",
            "scrum",
            "kanban",
            "power bi",
            "tableau",
            "superset",
            "datalens",
            "machine learning",
            "deep learning",
            "nlp",
            "computer vision",
            "data engineering",
            "data analysis",
            "data science",
            "business intelligence",
            "etl",
            "data warehouse",
            "big data",
            "cloud computing",
            "microservices",
        }

        # Разделяем навыки на технические и общие
        missing_tech = {skill for skill in missing_skills if skill in tech_skills}
        missing_other = missing_skills - missing_tech

        extra_tech = {skill for skill in extra_skills if skill in tech_skills}
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
