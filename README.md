# WrestleStat NCAA Division 1 Wrestling Data Scraper

This project is a Python-based web scraping tool that compiles comprehensive NCAA Division I wrestling data from [WrestleStat.com](https://www.wrestlestat.com/). It retrieves detailed match histories, team rosters, and season-level results across all D1 wrestling programs from the 2013–2014 season through the 2025–2026 season.

## 🎯 Goals

This project is the foundation for a long-term initiative to:

- 📂 **Build a clean, publicly available, and easy-to-use dataset** of NCAA Division I wrestling matches from 2013–2026.
- 🤖 **Develop machine learning models** to predict future match outcomes using historical performance data.
- 🌐 **Scale into a full-stack web application** for fans, analysts, and recruiters to interactively explore wrestlers, teams, trends, and predictions.

## 📍 In Progress

Currently working on:

- ⚡ Adding multithreading to accelerate season scraping performance
- ⚙️ Implementing CLI options for selecting specific seasons and teams

## 📌 Features

- **User authentication** to access complete match data
- **Team and wrestler data scraping** for every NCAA D1 program
- **Seasonal and full dataset generation** in CSV format
- **Automated coverage of inactive and newly added programs**

## 🧠 Key Functions

| Function | Description |
|---------|-------------|
| `login` | Authenticates the user on WrestleStat |
| `get_current_d1_teams` | Retrieves active D1 team IDs and slugs |
| `get_team_roster` | Retrieves wrestler info from team profile pages |
| `scrape_wrestler_matches` | Extracts individual match data for a specific wrestler |
| `scrape_team_matches` | Scrapes all wrestlers' matches for a given team |
| `scrape_all_d1_teams` | Automates scraping across all teams and seasons (2014–2026) |

## 📁 Folder Structure
```
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
- BeautifulSoup4
- pandas
- python-dotenv
- tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
````

Initialize Playwright:

```bash
playwright install
```

## 🔐 Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
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

* Log into WrestleStat
* Scrape every season from 2014 to 2026
* Export team-level and season-level CSVs
* Compile everything into `d1_all_match_results.csv`

You can also modify `scraper.py` to run only specific teams or seasons if needed.

## ⏱ Runtime Note

Full dataset scraping across all seasons may take **several hours**. It is recommended to:

* Use a stable internet connection
* Run in a terminal with logging enabled
* (Optional) Schedule with **Task Scheduler** (Windows) or **cron** (Unix)

## 📊 Example Output

A single match row includes:

| Season | Date       | Event                  | Weight Class | Result | Score | Opponent    | Wrestler        | School     |
| ------ | ---------- | ---------------------- | ------------ | ------ | ----- | ----------- | --------------- | ---------- |
| 2024   | 02/16/2025 | Penn State vs Illinois | 174          | W      | 6-1   | Edmond Ruth | Carter Starocci | Penn State |

## ⚠️ Known Issues

* Occasional site structure changes may break scraping—update selectors accordingly.
* Wrestlers with no matches or malformed data may be skipped silently.
* Long sessions may trigger login timeouts; re-authenticate if needed.

## 🧪 Future Improvements

* Add async/multithreaded scraping for faster performance
* Modularize scraper into a CLI-first package
* Build ML prediction pipelines based on historical results
* Add capability to scrape new matches in future seasons
* Develop front-end dashboard for interactive exploration
* Integrate cloud database support for data storage and queries

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Acknowledgments

Data source and project inspiration from [WrestleStat.com](https://www.wrestlestat.com/), an invaluable resource for NCAA wrestling fans and analysts.

