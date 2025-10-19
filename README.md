# WrestleStat NCAA Division 1 Wrestling Data Scraper/Predictor

This project is a Python-based web scraping tool that compiles comprehensive NCAA Division I wrestling data from [WrestleStat.com](https://www.wrestlestat.com/). It retrieves detailed match histories, team rosters, and season-level results across all D1 wrestling programs from the 2013–2014 season through the 2025–2026 season. It also contains multiple machine learning models trained to predict the results of hypothetical matches based on previous data.

## 📌 Features

- 🔐 User authentication to access full match data
- 🏫 Team and wrestler scraping for all NCAA D1 programs (active + inactive)
- 📄 Season and full-dataset CSV generation
- 🔁 Multi-season scraping with parallel processing
- 🗃️ Activity map support for defunct and newly added programs
- 🧹 Cleaning pipeline to standardize data and prepare for future use
- 🤖 Machine learning models capable of predicting matches with 77% accuracy

## 📊 Exploratory Analysis Highlights

After scraping and cleaning over 441,000 matches across 12 seasons (2013–2025), EDA has uncovered some key trends:

### 🕒 Time-Period Analysis

- 🎯 Decisions dominate: Over 50% of matches end in a decision (point differential <8)
- ⏱ Overtime is rare: Fewer than 5% of matches (~12,000 total) went into OT

### 📅 Season-by-Season Trends

- 🔄 Steady match volume, except for a COVID-19 dip in 2020–21
- 📊 Post-2023 result shifts: Technical falls increased following the 3-point takedown rule change

### 🏟️ Team-Level Insights

- 🏆 Bonus rate trends highlight aggressive programs like Penn State and Oklahoma State
- 👑 Penn State’s dominance is clear — with the highest win rates and bonus point percentages in recent seasons

### 💪 Wrestler-Level Performance

Used **Bayesian-adjusted metrics** to compare wrestlers with different match counts:

- 🧠 Win Rate Leaders: Yianni Diakomihalis, Zain Retherford
- 🧨 Bonus Point Machines: Jason Nolf, Zahid Valencia
- 🏋️ Workhorses: Parker Keckeisen — high consistency across 156 matches

## 🛠 Scraping Requirements

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

## 📊 Example Output (Raw Data)

A single match row includes:

| season | date  | event                  | is_dual_meet | weight_class | result | result_type | score  | opponent        | opponent_id | opponent_school | wrestler        | wrestler_id | wrestler_school |
| ------ | ----- | ---------------------- | ------------ | ------------ | ------ | ----------- | ------ | --------------- | ----------- | --------------- | --------------- | ----------- | --------------- |
| 2024   | 02/09 | Penn State - Iowa Dual | True         | 174          | W      | MD          | 13 - 5 | Patrick Kennedy | 56760       | Iowa            | Carter Starocci | 58819       | Penn State      |

## Future Development

At the moment I am currently developing a full-stack web application to create an interactive site that users can generate predictions through. This development will remain private for the moment, but this repo will be updated accordingly to reflect progress as the project continues.

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Acknowledgments

Data source and project inspiration from [WrestleStat.com](https://www.wrestlestat.com/), an invaluable resource for NCAA wrestling fans and analysts.
