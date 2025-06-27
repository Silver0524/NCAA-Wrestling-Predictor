# WrestleStat NCAA Division 1 Wrestling Data Scraper

This project is a Python-based web scraping tool that compiles comprehensive NCAA Division I wrestling data from [WrestleStat.com](https://www.wrestlestat.com/). It retrieves detailed match histories, team rosters, and season-level results across all D1 wrestling programs from the 2013–2014 season through the 2025–2026 season.

## 🎯 Goals

This project is the foundation for a long-term initiative to:

- 📂 **Build a clean, publicly available, and easy-to-use dataset** of NCAA Division I wrestling matches from 2013–2026.
- 🤖 **Develop machine learning models** to predict future match outcomes using historical performance data.
- 🌐 **Scale into a full-stack web application** for fans, analysts, and recruiters to interactively explore wrestlers, teams, trends, and predictions.

## 📍 In Progress

Currently working on:

- 🛠 Rescraping full dataset and cleaning data
- 📒 Updating README to reflect new folder structure

## 📌 Features

- 🔐 User authentication to access full match data
- 🏫 Team and wrestler scraping for all NCAA D1 programs (active + inactive)
- 📄 Season and full-dataset CSV generation
- 🔁 Multi-season scraping with parallel processing
- 🗃️ Activity map support for defunct and newly added programs

## 🧠 Key Functions

| Function                  | Description                                                       |
| ------------------------- | ----------------------------------------------------------------- |
| `login`                   | Authenticates a user on WrestleStat                               |
| `get_current_d1_teams`    | Retrieves active D1 team IDs and URL slugs                        |
| `get_team_roster`         | Fetches a specific team’s roster for a given season               |
| `scrape_wrestler_matches` | Scrapes an individual wrestler’s match history                    |
| `scrape_team_matches`     | Scrapes all wrestler matches for a single team and season         |
| `scrape_team_for_season`  | Wrapper function for scraping a team in a single browser instance |
| `scrape_all_d1_teams`     | Multithreaded scraper across all teams and seasons (2014–2026)    |

## 📁 Folder Structure

```Folder Structure
├── Team Results/
│   ├── Penn State/
│   │   └── 2025_penn-state.csv
│   └── ...
├── Year Results/
│   ├── 2014_matches.csv
│   ├── 2015_matches.csv
│   └── ...
├── d1_all_match_results.csv
├── scraper.py
├── .env
└── README.md
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

## 🚀 Usage

Run the scraper from the command line:

```bash
python scraper.py
```

This will:

- Log into WrestleStat
- Scrape every season from 2014 to 2026
- Export team-level and season-level CSVs
- Compile everything into `d1_all_match_results.csv`

You can also modify scraper.py or use CLI options (**in development**) to:

- Target specific seasons or teams
- Control multithreading parameters for faster scraping

## ⏱ Runtime Note

Full dataset scraping across all seasons may take **several hours**. It is recommended to:

- Use a stable internet connection
- Avoid power-intensive activities while scraping (gaming, video uploading, etc.)

## 📊 Example Output

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
