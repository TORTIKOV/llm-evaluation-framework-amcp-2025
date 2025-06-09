## 📁 Структура проекта
```
llm-evaluation-framework/
├── data\_loaders/
│   ├── loader\_summarization.py
│   ├── loader\_dialogue.py
│   └── loader\_narrativeqa.py
│
├── experiments/
│   ├── summarization/
│   │   ├── common.py
│   │   ├── run\_yandexgpt.py
│   │   └── run\_gigachat.py
│   ├── dialogue/
│   │   ├── common.py
│   │   ├── run\_yandexgpt.py
│   │   └── run\_gigachat.py
│   └── qa/
│       ├── common.py
│       ├── run\_yandexgpt.py
│       └── run\_gigachat.py
│
├── evaluation/
│   └── evaluate.py
│   └── sci.py
│
├── models/
│   ├── base\_model.py
│   ├── yandexgpt
│   │   ├── yandexgpt\_api.py
│   │   └── yandexgpt\_client.py
│   └── gigachat
│       ├── gigachat\_api.py
│       └── gigachat\_client.py
│
├── utils/
│   ├── translator.py
│   ├── save\_results.py
│   ├── timer.py
│   └── censorship\_filter.py
│
├── config/
│   └── config.py
```

---

## ⚙️ Инициализация

```bash
git clone https://github.com/your-username/llm-evaluation-framework-amcp-2025.git
cd llm-evaluation-framework-amcp-2025
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** Tестировал на macOS + Python 3.9. Убедитесь, что ресурсы nltk (например, punkt) загружены.

---

## 🧪 Запуск экспериментов

### 1. Summarization (XSum)

```bash
python experiments/summarization/run_yandexgpt.py
python experiments/summarization/run_gigachat.py
```

### 2. Dialogue (DailyDialog)

```bash
python experiments/dialogue/run_yandexgpt.py
python experiments/dialogue/run_gigachat.py
```

### 3. Question Answering (NarrativeQA)

```bash
python experiments/qa/run_yandexgpt.py
python experiments/qa/run_gigachat.py
```

> Все результаты будут сохранены в файлах формата `.xlsx` в папке `outputs/` 

---

## 📏 Метрики

* **ROUGE-1**
* **ROUGE-L**
* **METEOR**
* **BERTScore**
* **SCI** (Semantic Compression Index), with breakdown:

  * `SCI_Overlap` — factual consistency
  * `SCI_Compression` — brevity penalty
  * `SCI_IntentScore` — semantic similarity (SBERT)
  * `SCI_Delta`, `SCI_Penalty`, and weighted variants

---

## 🔐 Конфигурация

Edit `config/config.py` to set API keys and URLs:

```python
FOLDER_ID=your_yandex_folder_id
API_KEY=your_yandex_api_key
SBER_AUTH=your_sber_auth_login
SBER_ID=your_sber_client_id
SBER_SECRET=your_sber_client_secret
```
