# WrestleStat NCAA Division 1 Wrestling Data Scraper

This project is a Python-based web scraping tool that compiles comprehensive NCAA Division I wrestling data from [WrestleStat.com](https://www.wrestlestat.com/). It retrieves detailed match histories, team rosters, and season-level results across all D1 wrestling programs from the 2013–2014 season through the 2025–2026 season.

## 🎯 Goals

This project is the foundation for a long-term initiative to:

- 📂 **Build a clean, publicly available, and easy-to-use dataset** of NCAA Division I wrestling matches from 2013–2026.
- 🤖 **Develop machine learning models** to predict future match outcomes using historical performance data.
- 🌐 **Scale into a full-stack web application** for fans, analysts, and recruiters to interactively explore wrestlers, teams, trends, and predictions.

## 📍 In Progress

Currently working on:

- 🔬 Exploratory data analysis to reveal trends
- 🛠️ Feature engineering to prepare for predictive modeling

## 📌 Features

- 🔐 User authentication to access full match data
- 🏫 Team and wrestler scraping for all NCAA D1 programs (active + inactive)
- 📄 Season and full-dataset CSV generation
- 🔁 Multi-season scraping with parallel processing
- 🗃️ Activity map support for defunct and newly added programs
- 🧹 Cleaning pipeline to standardize data and prepare for future use

## 📁 Folder Structure

```Folder Structure
├── data/
│   ├── clean/
│   │   ├── d1_results_clean.csv
│   │   └── d1_results_unique.csv
│   └── raw/
│       ├── team_results/
│       │   ├── penn_state/
│       │   │   └── 2025_penn-state.csv
│       │   └── ...
│       ├── year_results/
│       │   ├── 2014_matches.csv
│       │   ├── 2015_matches.csv
│       │   └── ...
│       └── d1_results_raw.csv
├── notebooks/
│   ├── cleaning.ipynb
│   └── feature_engineering.ipynb
├── scripts/
│   ├── cleaning/
│   │   └── cleaning.py
│   └── scraping/
│       └── scraper.py
├── .env
├── README.md
└── requirements.txt
```

## 🛠 Requirements

- Python 3.8+
- [Playwright](https://playwright.dev/python/)
- [BeautifulSoup4](https://beautiful-soup-4.readthedocs.io/en/latest/)
- [pandas](https://pandas.pydata.org/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [tqdm](https://tqdm.github.io/)

Install dependencies with:

```bash
pip install -r requirements.txt
```

Initialize Playwright:

```bash
playwright install
```

## 🔐 Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
WRESTLESTAT_EMAIL=your_email@example.com
WRESTLESTAT_PASSWORD=your_password
```

These credentials are required to authenticate and access full wrestler match histories.

## ⏱ Runtime Note

Full dataset scraping across all seasons may take **several hours**. It is recommended to:

- Use a stable internet connection
- Avoid power-intensive activities while scraping (gaming, video uploading, etc.)

## 📊 Example Output (Raw Data)

A single match row includes:

| season | date  | event                  | is_dual_meet | weight_class | result | result_type | score  | opponent        | opponent_id | opponent_school | wrestler        | wrestler_id | wrestler_school |
| ------ | ----- | ---------------------- | ------------ | ------------ | ------ | ----------- | ------ | --------------- | ----------- | --------------- | --------------- | ----------- | --------------- |
| 2024   | 02/09 | Penn State - Iowa Dual | True         | 174          | W      | MD          | 13 - 5 | Patrick Kennedy | 56760       | Iowa            | Carter Starocci | 58819       | Penn State      |

## ⚠️ Known Issues

- Occasional site structure changes may break scraping—update selectors accordingly.
- Wrestlers with no matches or malformed data may be skipped silently.

## 🧪 Future Improvements

- Modularize scraper into a CLI-first package
- Engineer additional features to improve ML model performance
- Implement support for scraping newly added matches without full re-scraping
- Build machine learning pipelines for outcome prediction
- Develop front-end dashboard for interactive data exploration
- Integrate cloud database support (e.g., Firebase, Supabase, or S3) for data storage and queries

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Acknowledgments

Data source and project inspiration from [WrestleStat.com](https://www.wrestlestat.com/), an invaluable resource for NCAA wrestling fans and analysts.
