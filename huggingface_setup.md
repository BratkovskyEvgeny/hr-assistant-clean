# Настройка Hugging Face API

1. Зарегистрируйтесь на Hugging Face (https://huggingface.co/join)

2. Получите токен доступа:
   - Перейдите в настройки профиля (https://huggingface.co/settings/tokens)
   - Нажмите "New token"
   - Выберите "read" права
   - Скопируйте токен

3. Обновите файл `.streamlit/secrets.toml`:
```toml
[huggingface]
token = "ваш_токен_здесь"
```

4. Перезапустите приложение

Примечания:
- Hugging Face предоставляет бесплатный доступ к API
- Есть ограничения на количество запросов
- Можно использовать разные модели, например:
  - gpt2 (базовая)
  - mistralai/Mistral-7B-Instruct-v0.2 (более продвинутая)
  - facebook/opt-350m (оптимизированная) 