FROM python:3.9

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения
COPY . .

# Открытие порта
EXPOSE 7860

# Запуск приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"] 