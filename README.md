# MLOps Лабораторна №3: Моніторинг ML API та детекція проблем

**Автор:** Дмитро Рєзєнков
**Дата:** 2026-05-06

> **Примітка щодо варіанту:** методичка ЛР3 не містить індивідуальних варіантів — для всіх студентів використовується класифікатор Iris, побудований у ЛР2. Ця робота розширює його шаром моніторингу.

## Опис системи

ML API з ЛР2 розширено до повноцінно спостережуваного (observable) сервісу:

- **Prometheus-метрики** інструментують код (Counter, Histogram, Gauge), endpoint `GET /metrics` віддає їх у Prometheus exposition format.
- **Prometheus** піднімається локально через `docker-compose` і кожні 10 секунд скрейпить ML API.
- **Drift detector** (`DriftDetector`) зберігає reference-вибірку (X_train) і виконує KS-тест Колмогорова–Смирнова для кожної ознаки проти live-вибірки. Endpoint `POST /check-drift` приймає батч і повертає p-value та рішення `drift_detected` для кожної ознаки.
- **Structured JSON logging** (`python-json-logger`): кожна подія (`startup`, `prediction`, `drift_check`, помилка) пишеться як одна JSON-лінія у stdout.
- **Evidently HTML-звіт** (опціональне завдання): візуальне порівняння reference та live з усіма стат-тестами.

Стек: Python 3.11, FastAPI 0.115, Prometheus client 0.21, scipy 1.14 (KS-тест), python-json-logger 2.0, Prometheus 2.55, Evidently 0.4.40, Docker Compose.

## Структура репозиторію

```
ml-api-lab3/
├── .github/workflows/ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py                  # /, /health, /metrics, /predict, /check-drift + middleware
│   ├── schemas.py               # IrisFeatures, PredictionResponse, DriftRequest/Response
│   ├── metrics.py               # Counter / Histogram / Gauge у власному CollectorRegistry
│   ├── drift.py                 # DriftDetector з KS-тестом
│   └── logging_config.py        # JSON-формат логів
├── ml/
│   ├── __init__.py
│   └── train.py                 # тренує модель + зберігає reference_stats.joblib
├── monitoring/
│   ├── prometheus.yml           # scrape config (ml-api:8000 кожні 10с)
│   └── docker-compose.monitoring.yml
├── scripts/
│   └── evidently_report.py      # bonus: HTML-звіт про drift
├── tests/
│   ├── conftest.py              # client fixture з lifespan
│   ├── test_api.py              # 6 тестів API (root/health/predict/422)
│   ├── test_model.py            # 3 тести train + reference
│   ├── test_metrics.py          # 3 тести /metrics і Counter-інкременту
│   └── test_drift.py            # 6 тестів DriftDetector (KS-тест на синтетиці)
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## Реалізовані метрики

| Метрика | Тип | Labels | Призначення |
|---|---|---|---|
| `ml_predictions_total` | Counter | `class_name`, `status` | Скільки прогнозів зроблено за класом і статусом (success/error) |
| `ml_prediction_latency_seconds` | Histogram (buckets 5ms…5s) | — | Розподіл часу інференсу `/predict`; для p95/p99 |
| `ml_prediction_confidence` | Histogram (buckets 0.1…1.0) | — | Розподіл `predict_proba` обраного класу — індикатор впевненості |
| `ml_errors_total` | Counter | `error_type` | Помилки: `model_not_loaded`, `inference_error`, `drift_detector_not_ready` |
| `ml_model_loaded` | Gauge | — | 1 = модель завантажилась, 0 = ні (виставляється у lifespan) |
| `ml_drift_checks_total` | Counter | — | Кількість викликів `/check-drift` |
| `ml_drift_detected_total` | Counter | `feature` | Скільки разів конкретна ознака була визнана drifted |

Усі метрики реєструються у власному `CollectorRegistry()`, а не глобальному — це чистіше для тестів і ізолює нас від метрик самого FastAPI.

### Приклад виводу `/metrics`

```text
# HELP ml_predictions_total Total number of model predictions
# TYPE ml_predictions_total counter
ml_predictions_total{class_name="setosa",status="success"} 30.0
ml_predictions_total{class_name="virginica",status="success"} 15.0

# HELP ml_prediction_latency_seconds Inference latency in seconds
# TYPE ml_prediction_latency_seconds histogram
ml_prediction_latency_seconds_bucket{le="0.005"} 44.0
ml_prediction_latency_seconds_bucket{le="0.01"} 45.0
...
ml_prediction_latency_seconds_count 45.0

ml_model_loaded 1.0
ml_drift_checks_total 2.0
ml_drift_detected_total{feature="sepal_length"} 1.0
```

## Drift detection

`DriftDetector` (`app/drift.py`) — простий, прозорий статистичний детектор для числових ознак:

1. Конструктор приймає reference-вибірку (X_train у формі numpy 2D) і список feature_names.
2. Метод `detect(current, alpha)` для кожної ознаки викликає `scipy.stats.ks_2samp(ref_col, cur_col)` і повертає `(D, p_value)`.
3. Якщо `p_value < alpha` — ознака помічається як drifted; загальний прапорець `drift_detected` = OR за всіма ознаками.

Альфа-поріг (типово 0.05) задається у запиті — це дозволяє регулювати чутливість без зміни коду.

### Endpoint `/check-drift`

| Поле | Тип | Опис |
|---|---|---|
| `samples` | `List[List[float]]` (≥10 рядків × рівно 4 числа) | Live-батч |
| `alpha` | `float ∈ [0.001, 0.5]` (default 0.05) | Поріг значущості KS |

Відповідь:

| Поле | Опис |
|---|---|
| `drift_detected` | Загальний прапорець (OR за ознаками) |
| `n_drifted_features` | Скільки ознак drifted |
| `drifted_features` | Список їхніх імен |
| `per_feature` | Детальні `{statistic, p_value, drift_detected}` для кожної ознаки |
| `n_samples`, `alpha` | Echo вхідних параметрів |

## Приклади запитів

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

```json
{"class_id":0,"class_name":"setosa","probability":0.9784}
```

### `POST /check-drift` — здоровий батч

```bash
curl -X POST http://localhost:8000/check-drift \
  -H "Content-Type: application/json" \
  -d '{"samples":[[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],
                  [5.4,3.9,1.7,0.4],[5.0,3.6,1.4,0.2],[5.5,2.5,4.0,1.3],
                  [6.1,2.9,4.7,1.4],[6.0,3.0,4.8,1.8],[6.3,2.5,5.0,1.9],
                  [6.5,3.0,5.2,2.0]],"alpha":0.05}'
```

→ `{"drift_detected": false, ...}` (значення близькі до тренувальних).

### `POST /check-drift` — drifted батч

```bash
curl -X POST http://localhost:8000/check-drift \
  -H "Content-Type: application/json" \
  -d '{"samples":[[9.0,8.0,8.0,5.0],[9.5,7.5,8.5,5.5],[8.5,8.5,7.5,4.5],
                  [9.2,8.2,8.2,5.2],[9.8,7.8,8.8,5.8],[8.8,8.8,7.8,4.8],
                  [9.4,8.4,8.4,5.4],[9.6,7.6,8.6,5.6],[8.6,8.6,7.6,4.6],
                  [9.1,8.1,8.1,5.1],[9.3,8.3,8.3,5.3],[9.7,7.7,8.7,5.7]],
       "alpha":0.05}'
```

→ `{"drift_detected": true, "n_drifted_features": 4, "drifted_features": ["sepal_length","sepal_width","petal_length","petal_width"], ...}` — для кожної ознаки `p_value ≈ 0`.

## Логування

Усі ключові події пишуться як **structured JSON** в stdout (формат стандартного `python-json-logger` з полями `timestamp`, `level`, `logger`, `event`, плюс контекст конкретної події).

Приклади з реального запуску:

```json
{"event":"startup","model_loaded":true,"drift_detector_ready":true,"timestamp":"2026-05-06 15:42:06,241","level":"INFO","logger":"ml-api"}
{"event":"prediction","class_id":0,"class_name":"setosa","probability":0.9784,"features":{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},"timestamp":"...","level":"INFO","logger":"ml-api"}
{"event":"drift_check","n_samples":12,"alpha":0.05,"drift_detected":true,"drifted_features":["sepal_length","sepal_width","petal_length","petal_width"],"timestamp":"...","level":"INFO","logger":"ml-api"}
{"event":"inference_error","timestamp":"...","level":"ERROR","logger":"ml-api"}
```

Такі логи легко агрегуються Loki/ELK і дають другий «вимір» спостережуваності — на доповнення до агрегованих метрик Prometheus.

## Як запустити моніторинг

### 1. Локальний ML API

```bash
git clone <repo-url>
cd ml-api-lab3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.train                 # створює model.joblib + reference_stats.joblib
uvicorn app.main:app --reload      # http://localhost:8000
```

Перевірка: `http://localhost:8000/docs` (Swagger UI), `http://localhost:8000/metrics`.

### 2. Prometheus + ML API через docker-compose

```bash
cd monitoring
docker compose -f docker-compose.monitoring.yml up --build
```

- ML API → <http://localhost:8000> (Swagger на `/docs`)
- Prometheus UI → <http://localhost:9090>
- Targets → <http://localhost:9090/targets> (ml-api має бути `UP`)

### 3. Корисні PromQL-запити

| Питання | PromQL |
|---|---|
| Швидкість прогнозів за останню хвилину | `rate(ml_predictions_total[1m])` |
| Розподіл прогнозів за класами | `sum by (class_name) (ml_predictions_total)` |
| 95-й перцентиль latency | `histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))` |
| Середня впевненість моделі | `histogram_quantile(0.5, rate(ml_prediction_confidence_bucket[5m]))` |
| Частка помилок | `sum(rate(ml_errors_total[5m])) / sum(rate(ml_predictions_total[5m]))` |
| Drift у `petal_length` | `ml_drift_detected_total{feature="petal_length"}` |

### 4. Тести

```bash
pytest -q
```

Очікувано: `18 passed`. Розподіл:
- `test_model.py` — 3 (тренування створює обидва артефакти, reference shape, predict повертає 0/1/2)
- `test_api.py` — 6 (root, health з drift_detector_ready, predict setosa/virginica, 422 на невалідні)
- `test_metrics.py` — 3 (`/metrics` віддає 200, `predict` інкрементує counter, `check-drift` інкрементує лічильник)
- `test_drift.py` — 6 (no-drift на однаковому розподілі, drift на зсунутому, частковий drift, валідація розміру)

### 5. Evidently HTML-звіт (опціонально)

```bash
python scripts/evidently_report.py
# відкрити drift_report.html у браузері
```

Звіт містить інтерактивні графіки: розподіл reference vs current, KS-статистики, p-values, і загальний `dataset drift` summary.

## Висновки

- ML-сервіс отримав три «опори» observability: **метрики** (Prometheus), **логи** (structured JSON), і базу для **трейсінгу** (через middleware можна додати OpenTelemetry без змін у бізнес-логіці).
- **KS-тест** виявився достатнім для tabular Iris-датасету (всі ознаки числові). На зсуві `loc 5 → 8` за всіма 4 ознаками детектор спрацював 4/4 з p_value ≈ 0; на однорідному батчі — 0/4 з p_value > 0.05.
- **Власний `CollectorRegistry`** замість глобального — найважливіше архітектурне рішення тут: воно ізолює наші метрики від default-метрик FastAPI/uvicorn і робить тести детермінованими.
- **Histogram з low buckets (5ms…5s)** свідомо асиметричні — зважено на типові інференс-часи легких моделей (логрег на 4 фічах виконується за <5 ms, тому майже всі обсервації потрапляють у перший bucket; для важчих моделей buckets треба зсунути правіше).
- **Pull-модель Prometheus** автоматично детектить недоступність цілі — на `/targets` ціль помічається `DOWN` без жодних змін у коді ML API. Це краще, ніж push-моделі, які при недоступності metric-server тихо втрачають дані.

## Контрольні питання

### 1. Поясніть різницю між моніторингом класичного веб-сервісу та моніторингом ML-сервісу. Які метрики потрібно додатково збирати у ML-системі і чому стандартних метрик типу latency / error rate недостатньо для виявлення проблем із моделлю?

**Класичний сервіс** має чітко визначений контракт: «отримай запит → виконай детермінований код → поверни відповідь». Поломки помітні: 500-ка, timeout, exception, чи різке зростання p99 latency. Якщо ці чотири індикатори зелені — сервіс «працює».

**ML-сервіс** має додатковий шар недетермінізму: одна й та сама `/predict` з тим самим payload завжди повертає 200 OK з валідним JSON, але **зміст** прогнозу залежить від статистичних властивостей вхідних даних. Сервіс може «тихо» деградувати: latency 12 ms, error rate 0 %, але accuracy впала з 94 % до 67 %, бо у проді почав надходити інший розподіл. Стандартний моніторинг цього не побачить.

**Що додатково моніториться в ML:**

| Категорія | Метрики | Що ловить |
|---|---|---|
| **Розподіл вхідних ознак** | гістограма кожної ознаки (live vs reference), KS-статистика, PSI | Data drift, помилки інтеграції, нові сегменти користувачів |
| **Розподіл прогнозів** | `predictions_total{class}` — частка кожного класу | Concept drift, перекіс трафіку, поломки upstream |
| **Впевненість моделі** | гістограма `predict_proba` обраного класу | Модель «вагається» (середня впевненість падає з 0.95 до 0.55) — сильний сигнал |
| **Якість на ground truth** | accuracy / F1 на шматку реальних розмічених даних (з затримкою) | Найважливіша, але доступна тільки коли є labels |
| **Бізнес-метрики** | conversion, кількість fallback-ів, дохід на запит | Кінцевий критерій: чи модель виконує свою роль |
| **Latency / Error rate** | p50/p95/p99, частка 5xx | Технічна справність — те ж саме, що й у класичному сервісі |

Без цих ML-специфічних метрик ви дізнаєтесь про проблему тільки коли вона дійде до бізнес-метрик (тобто пізно — після того як користувачі вже отримали поганий досвід). У моніторингу ML головне правило: **ніколи не довіряй 200 OK** — модель може повертати ввічливо невірні відповіді з ідеальною технічної здоров'ям.

### 2. Поясніть pull-модель Prometheus. Що відбувається, коли ML API повертає 500 на запит до /metrics або взагалі стає недоступний? Чи помітить це Prometheus, і у який спосіб?

**Pull-модель** означає, що Prometheus сам ініціює збір даних: за заданим у `prometheus.yml` інтервалом (у нас 10 с) робить HTTP-запит `GET <target>/metrics`, парсить text exposition format і записує всі семпли у власну time-series базу. Клієнтський код (ML API) **нічого не надсилає** — він тільки відкриває endpoint і чекає, коли Prometheus прийде.

Це принципово відрізняє Prometheus від push-моделей (StatsD, OpenTelemetry OTLP), де клієнти самі шлють метрики у колектор.

**Що буде, якщо ML API падає:**

| Сценарій | Що бачить Prometheus | Метрика для алерту |
|---|---|---|
| ML API повертає 500 на `/metrics` | Scrape failed → ціль `health: down` у `/targets`; `up{job="ml-api"} = 0` | `up == 0 for 1m` |
| ML API не відповідає (timeout) | Scrape failed → `up = 0`, у логах Prometheus `context deadline exceeded` | Те саме |
| ML API повертає 200, але з неповним або битим текстом | Scrape failed з parse error | `up = 0` плюс `scrape_samples_post_metric_relabeling = 0` |
| ML API повертає 200 з нормальним текстом | `up = 1`, всі семпли записані з timestamp scrape-у | — |

**Ключова метрика — `up{job="..."}`**: вона існує **завжди** для кожного scrape target, незалежно від того, що віддає сам сервіс. `up = 1` означає «scrape вдався», `up = 0` — «не вдався». Це автоматичний liveness probe сервісу — нічого додатково писати не треба.

**Переваги pull-моделі для ML:**
1. Кожен запит scrape — атомарний знімок стану; немає черги push-ів, які можна втратити.
2. Prometheus сам бачить недоступних — не треба окремої системи heartbeat.
3. Можна тестувати вручну: відкрити `<service>/metrics` у браузері і подивитись, що він віддає (як я робив у smoke-тесті).
4. Легко контролювати навантаження: один централізований Prometheus робить запит у тому ритмі, у якому хоче.

**Недоліки:** не підходить для коротко-живущих jobs (Lambda, cron-task), які можуть закінчити роботу до наступного scrape — для них Prometheus має окремий компонент Pushgateway, до якого ці jobs пушать перед exit-ом.

### 3. У чому принципова різниця між метриками типу Counter, Gauge та Histogram?

Усі три — типи семплів у time-series базі Prometheus, але вони описують **різні види даних**.

| Тип | Формальне правило | Що вимірює | Приклад у нашому коді | PromQL для роботи з нею |
|---|---|---|---|---|
| **Counter** | Монотонно зростає від старту процесу; обнуляється тільки при рестарті | Кількість подій з моменту старту | `ml_predictions_total`, `ml_errors_total` | Сире значення малоінформативне; завжди `rate(metric[window])` для швидкості або `increase(metric[window])` для приросту |
| **Gauge** | Може зростати і спадати | Миттєвий стан системи | `ml_model_loaded` (0 або 1) | Можна агрегувати безпосередньо: `avg`, `max`, `sum` |
| **Histogram** | Збирає **розподіл** значень за заздалегідь визначеними buckets | Latency, розміри payload-ів, predict_proba | `ml_prediction_latency_seconds`, `ml_prediction_confidence` | `histogram_quantile(0.95, rate(metric_bucket[5m]))` для перцентилів |

**Чому це важливо:**

- **Counter ≠ Gauge.** Counter «1247 запитів з моменту старту» сам по собі непотрібний — потрібна **швидкість** (запитів/сек), яку дає `rate()`. Якщо помилково використати Gauge для лічильника, при рестарті процесу значення скине у 0, і `rate()` побачить це як гігантський негативний стрибок (rate Counter-а ігнорує decrease, гарантуючи коректний результат).
- **Gauge ≠ Counter.** Кількість активних з'єднань або поточна температура CPU природно коливаються — там немає сенсу в `rate()`.
- **Histogram ≠ середнє.** Середній latency — це майже безкорисна метрика: 99 запитів по 10 ms + 1 запит на 10 секунд дають середнє 110 ms, яке приховує катастрофічну затримку 1 % користувачів. Histogram зберігає buckets (`le="0.05"`, `le="0.1"`, ...) і дозволяє обчислити будь-який перцентиль (p50, p95, p99) на стороні Prometheus.

**Внутрішнє влаштування Histogram у нас:**

```
ml_prediction_latency_seconds_bucket{le="0.005"} 1100   # ≤5 ms
ml_prediction_latency_seconds_bucket{le="0.01"}  1200   # ≤10 ms
...
ml_prediction_latency_seconds_bucket{le="+Inf"}  1247   # усі
ml_prediction_latency_seconds_sum                38.7   # сума всіх вимірів
ml_prediction_latency_seconds_count              1247   # стільки спостережень
```

З цих сирих даних PromQL обчислює перцентиль через `histogram_quantile()`. Третій тип, **Summary**, обчислює перцентилі на стороні клієнта — він простіший, але не агрегується через кілька реплік сервісу, тому в сучасних практиках надають перевагу Histogram.

### 4. Поясніть, що таке data drift і чому він є серйозною загрозою для ML-моделей у продакшені, навіть коли сервіс не повертає жодної помилки. Назвіть і поясніть три можливі причини виникнення drift на конкретному прикладі ML-сервісу — оцінки кредитного ризику.

**Data drift** — статистично значуща зміна розподілу вхідних даних, які надходять до моделі у проді, відносно даних, на яких модель навчалась. Формально: розподіл `P(X)` під час інференсу більше не дорівнює тому самому розподілу під час тренування. Існують також **label drift** (зміна `P(Y)`) і **concept drift** (зміна `P(Y|X)` — той самий вхід тепер означає інше).

**Чому drift небезпечний попри відсутність помилок:**

ML-модель — це функція `f(X) → Y`, навчена на конкретній області у просторі ознак. Поза цією областю модель **екстраполює** — і ніколи правильно. Але:
- На рівні HTTP — все ок: 200 OK, валідний JSON, latency у межах SLA.
- На рівні Pydantic — все ок: типи правильні, межі дотримані.
- На рівні моделі — `predict_proba` повертає валідні ймовірності у [0,1].
- **Кінцева якість прогнозу — катастрофічна,** але дізнатися про це можна тільки після того, як отримаєте labels з затримкою (тижні-місяці), або через зростання частки відмов / падіння бізнес-метрик.

Це називається **silent failure** — найгірший вид збою, бо він не запускає жодних звичних алертів.

**Три причини drift на прикладі сервісу оцінки кредитного ризику:**

1. **Зміни середовища (макроекономічний шок).** Модель навчена на даних 2019 р., коли середня зарплата у вибірці = 15 000 грн і безробіття 8 %. У 2024 р. через інфляцію середня зарплата у запитах = 25 000 грн, а через війну зросла частка переселенців з нестабільним доходом. Модель бачить «25 000 → це багатий клієнт → низький ризик» — але насправді ці 25 000 у нових цінах = ті самі 15 000 у 2019. Результат: масові видачі кредитів неплатоспроможним.

2. **Зміни джерела даних (поломка інтеграції).** Партнер-банк, що передає кредитну історію, оновив API: поле `credit_score` тепер у діапазоні 0–100 замість 300–850. Pydantic пропустить (поле float), але модель бачить «score = 75 → значить дуже низький ризик» (бо у тренуванні 75 = неможливе значення, ближче до нижньої межі 300). Drift детектор побачить колосальний зсув розподілу `credit_score` за день і спрацює — без нього банк дізнається про поломку через тижні після десятків мільйонів збитків.

3. **Зміни поведінки користувачів (нова географія/сегмент).** Сервіс розширився на онлайн-канал з мобільним застосунком, який залучив новий сегмент: молодь 18–22 роки, яка раніше не подавала заяви. У тренувальній вибірці середній вік був 38, а у нових запитах — 21. Модель не бачила такого сегмента і не вміє його оцінювати; ймовірно, видає неадекватно високий ризик (бо корелює молодий вік з відсутністю кредитної історії = «підозрілий клієнт»). Бізнес втрачає якраз найперспективніших нових клієнтів.

**Як ми це детектимо у нашій ЛР:** `/check-drift` отримує батч живих запитів і викликає KS-тест проти `reference_stats` (X_train). У продакшені цей батч зазвичай збирається з логів за останній період (день, годину). Метрика `ml_drift_detected_total{feature="..."}` росте — Prometheus тригерить алерт — інженер дивиться у Grafana чи Evidently, який саме розподіл «поїхав», — і вирішує: переобучити модель або заблокувати трафік до фіксу інтеграції.

### 5. Поясніть, як працює KS-тест Колмогорова-Смирнова та що означає p-value у його результаті.

**KS-тест (двовибірковий)** — непараметричний статистичний тест, який перевіряє нульову гіпотезу `H0`: «дві вибірки походять з одного й того ж розподілу». «Непараметричний» означає, що тест не вимагає припущень про конкретну форму розподілу (нормальний, експоненційний тощо) — він працює з будь-якими неперервними числовими даними. Тому KS добре пасує для перевірки drift, де ми не знаємо нічого про реальний розподіл живих даних.

**Як він працює, інтуїтивно:**

1. Для кожної з двох вибірок будується **емпірична кумулятивна функція розподілу** (CDF) `F1(x)` і `F2(x)`. CDF у точці `x` — це частка спостережень, що ≤ x; вона має форму сходинок, що зростають від 0 до 1.
2. Тест вимірює **максимальну вертикальну відстань** між цими сходинками: `D = max_x |F1(x) − F2(x)|`. Якщо вибірки походять з одного розподілу, їхні CDF мають збігатися, і `D ≈ 0`. Чим сильніше розходяться вибірки — тим більше `D`.
3. На основі `D` і розмірів вибірок (`n1`, `n2`) тест обчислює **p-value** — ймовірність побачити настільки велике (або більше) `D`, **якщо насправді H0 правдива** (тобто розподіли однакові і вся різниця — випадковість).

**Що означає p-value (і чого воно НЕ означає):**

- p-value **не** є «ймовірністю того, що дані відрізняються» (популярна помилкова інтерпретація).
- p-value — це **ймовірність побачити такі ж або ще екстремальніші розбіжності за припущення, що насправді розподіли однакові**.

**Інтерпретація на пальцях:**
- **Маленьке p-value** (наприклад, 0.001) означає: «отримані розбіжності було б дуже дивно побачити випадково» → випадковість як пояснення відкидається → відкидаємо `H0` → робимо висновок: розподіли значуще відрізняються → є drift.
- **Велике p-value** (наприклад, 0.4) означає: «розбіжності цілком пояснюються випадковістю» → підстав відкидати `H0` немає → drift статистично не виявлено.

**Поріг α (alpha):**
- За домовленістю беруть α = 0.05 (інколи 0.01) — це **компроміс між чутливістю до справжнього drift і частотою помилкових тривог**.
- Правило: `p_value < α → drift; p_value ≥ α → no drift`.
- При α = 0.05 ми очікуємо помилково оголосити drift приблизно у 5 % випадків, коли його насправді немає (false positive rate). У сервісах з високим трафіком навіть невелика реальна різниця стає статистично значущою при великих вибірках — тому додатково дивляться і на сам **D** (величина практичної різниці), а не тільки на p-value.

**Підтвердження у наших тестах:**
- `test_no_drift_on_same_distribution` — обидві вибірки з `N(5, 1)` → p_value > 0.05 → `drift_detected = False` ✓
- `test_drift_on_shifted_distribution` — друга вибірка з `N(8, 1)` (зсув на 3σ) → p_value ≪ 0.001 для всіх 4 ознак → `drift_detected = True` ✓

**Обмеження KS:** працює тільки з неперервними (числовими) ознаками. Для категоріальних застосовують χ²-тест або PSI (Population Stability Index). У нашому Iris-датасеті всі 4 ознаки числові, тому KS — природний вибір.
