# FastAPI_ML 😇

В этом репозитории собраны файлы для создания веб-приложения на FastAPI для машинного обучения.

**Использованные технологии: FastAPI, Docker, Hugging Face**

## Local development (Windows)

```

# (создание виртуальной среды)
pip py3.11 -m venv env

# активация виртуальной среды из корневой папки проекта
env\Scripts\activate

# запуск тестов
pytest tests/test_ml.py

pytest tests/test_app.py

# запуск приложения
uvicorn app.app:app --host 0.0.0.0 --port 8080

```

## Создание и запуск docker-контейнера

```

# создание
docker build -t ml-app .

# запуск
docker -run -p 80:80 ml-app

```