import os


def test_connection():
    print("Тестируем подключение к Kaggle API...")

    try:
        # Проверяем наличие переменных окружения
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")

        if not username or not key:
            print(
                "Ошибка: Не найдены переменные окружения KAGGLE_USERNAME или KAGGLE_KEY"
            )
            return

        # Создаем временный файл конфигурации
        config_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(config_dir, exist_ok=True)

        config_path = os.path.join(config_dir, "kaggle.json")
        with open(config_path, "w") as f:
            f.write(f'{{"username": "{username}", "key": "{key}"}}')

        # Инициализация API
        api = KaggleApi()
        api.authenticate()

        # Получаем информацию о пользователе
        user_info = api.get_user_info()
        print("\nУспешное подключение!")
        print(f"Имя пользователя: {user_info['username']}")
        print(f"Email: {user_info['email']}")

        # Получаем список ноутбуков
        notebooks = api.notebooks_list()
        print("\nДоступные ноутбуки:")
        for notebook in notebooks:
            print(f"- {notebook['ref']}")

    except Exception as e:
        print(f"\nОшибка при подключении: {str(e)}")
    finally:
        # Удаляем временный файл конфигурации
        if os.path.exists(config_path):
            os.remove(config_path)


if __name__ == "__main__":
    test_connection()
