import os

from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем токен из переменных окружения
HF_TOKEN = os.getenv("HF_TOKEN")

# URL модели
API_URL = "https://api-inference.huggingface.co/models/facebook/opt-350m"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def test_model_access():
    """Тестирует доступ к модели через Hugging Face API"""
    try:
        # Проверяем наличие токена
        if not HF_TOKEN:
            print(
                "❌ Токен не найден. Пожалуйста, установите переменную окружения HF_TOKEN"
            )
            return

        print("🔍 Проверяем доступ к модели...")

        # Тестовый запрос
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "inputs": "Hello! How are you?",
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "return_full_text": False,
                },
            },
            timeout=30,
        )

        # Проверяем статус ответа
        if response.status_code == 404:
            print("❌ Модель не найдена. Пожалуйста, проверьте URL модели")
            return

        response.raise_for_status()

        # Выводим результат
        result = response.json()[0]["generated_text"]
        print("\n✅ Успешное подключение к модели!")
        print("\n📝 Ответ модели:")
        print(result)

    except requests.exceptions.Timeout:
        print("⏰ Превышено время ожидания ответа от модели")
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при запросе к API: {str(e)}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {str(e)}")


if __name__ == "__main__":
    test_model_access()
