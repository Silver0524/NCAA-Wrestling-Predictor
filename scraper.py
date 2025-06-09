# import packages
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import pandas as pd
import time
from tqdm import tqdm
import re
from dotenv import load_dotenv
import os


# WrestleStat requires an account to view wrestler's full match histories
def login(page, email, password):
    page.goto("https://www.wrestlestat.com/account/login")
    page.fill("input[name='Username']", email)
    page.fill("input[name='Password']", password)
    page.click("button[type='submit']")
    page.wait_for_url("https://www.wrestlestat.com/", timeout=10000)

# Returns a list of all Division 1 wrestling teams along with associated team ids
def get_all_d1_teams(page):
    url = "https://www.wrestlestat.com/d1/rankings/dual"
    page.goto(url)
    soup = BeautifulSoup(page.content(), "html.parser")

    teams = []
    for link in soup.select('td a[href^="/team/"]', class_='table table-tight'):
        href = link['href']
        if '/profile' in href:
            parts = href.split('/')
            team_id = int(parts[2])
            team_name = parts[3]
            teams.append((team_id, team_name))
    return list(set(teams))

# Returns a list of all wrestlers on a team's roster
def get_team_roster(page, team_id, team_slug):
    url = f"https://www.wrestlestat.com/team/{team_id}/{team_slug}/profile"
    page.goto(url)
    soup = BeautifulSoup(page.content(), 'html.parser')
    roster = []

    rows = soup.select("table.table tbody tr")
    for row in rows[2:]:
        cols = row.find_all("td")
        if not cols:
            continue

        name_cell = cols[0].find("a", href=True)
        if not name_cell:
            continue

        wrestler_url = name_cell['href']
        try:
            parts = wrestler_url.strip('/').split('/')
            wrestler_id = int(parts[1])
            wrestler_slug = parts[2]
            raw_name = name_cell.text

            match = re.match(r"#\d+\s+([^,]+),\s*(.+)", raw_name)
            if match:
                last_name, first_name = match.groups()
                wrestler_name = f"{first_name.strip()} {last_name.strip()}"

            if "(" not in raw_name:
                roster.append((wrestler_id, wrestler_name, wrestler_slug))
        except:
            continue

    return roster

# Returns every match from a wrestler's collegiate career as a dataframe
def scrape_wrestler_matches(page, wrestler_id, wrestler_name, wrestler_slug):
    url = f"https://www.wrestlestat.com/wrestler/{wrestler_id}/{wrestler_slug}/profile"
    page.goto(url)
    soup = BeautifulSoup(page.content(), 'html.parser')
    all_matches = []

    season_blocks = soup.select("div.row.mt-1")

    for season_block in season_blocks:
        # Get the season year
        h2 = season_block.find("h2")
        if not h2:
            continue
        season = h2.text.strip().split(' ')[0]

        # The next sibling with class="row" contains the match table
        table_div = season_block.find_next_sibling("div", class_="row")
        if not table_div:
            continue

        table = table_div.find("table", class_="table")
        if not table:
            continue

        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) != 9:
                continue

            opponent_a = cols[1].find("a")
            if not opponent_a:
                continue

            try:
                opponent_raw_name = opponent_a.text.strip()
                opponent_record = cols[1].find("small") 
                opponent_record = opponent_record.text.strip(" ()") if opponent_record else "Unlisted"

                name_no_rank = re.sub(r"^#\d+\s*", "", opponent_raw_name)
                if "," in name_no_rank:
                    last, first = [part.strip() for part in name_no_rank.split(",", 1)]
                    opponent_name_clean = f"{first} {last}"
                else:
                    opponent_name_clean = name_no_rank 

                raw_school = cols[2].text.strip()
                cleaned_school = re.sub(r"\(.*?\)|#\d+\s*", "", raw_school).strip()

                match = {
                    "Season": season,
                    "Date": cols[3].text.strip(),
                    "Event": cols[4].text.strip(),
                    "Weight Class": cols[5].text.strip(),
                    "Result": cols[6].text.strip(),
                    "Result Type": cols[7].text.strip(),
                    "Score": cols[8].text.strip(),
                    "Opponent": opponent_name_clean,
                    "Opponent Record": opponent_record,
                    "Opponent School": cleaned_school,
                    "Wrestler": wrestler_name,
                    "Wrestler ID": wrestler_id
                }
                all_matches.append(match)
            except Exception as e:
                print(f"⚠️ Error parsing match row: {e}")
                continue
    
    df = pd.DataFrame(all_matches)
    df = df.drop_duplicates()

    df.replace("", pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    return df

# Returns a df containing all the matches from every wrestler on a team's roster and saves it to a csv file
def scrape_team_matches(page, team_id, team_slug, delay=1.0):
    roster = get_team_roster(page, team_id, team_slug)
    print(f"Found {len(roster)} wrestlers for {team_slug}...")

    all_matches = []

    for wrestler_id, wrestler_name, wrestler_slug in tqdm(roster, desc=f"Scraping {team_slug.title()}"):
        df = scrape_wrestler_matches(page, wrestler_id, wrestler_name, wrestler_slug)
        if df is not None and not df.empty:
            all_matches.append(df)
        time.sleep(delay)

    if all_matches:
        full_df = pd.concat(all_matches, ignore_index=True)
        full_df.to_csv(f"Team Results/{team_slug}_match_results.csv", index=False)
        print(f"Saved {len(full_df)} matches to {team_slug}_match_results.csv")
        return full_df
    else:
        print(f"No match data found for team {team_slug}.")
        return None

# Runs full scraping script that saves every match into a single csv file
def scrape_all_d1_teams():
    load_dotenv()
    with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            login(page, os.getenv('WRESTLESTAT_EMAIL'), os.getenv('WRESTLESTAT_PASSWORD'))

            all_data = []
            teams = get_all_d1_teams(page)
            
            """for team_id, team_slug in tqdm(teams, desc=f"Scraping {team_slug.title()}"):
                try:
                    df = scrape_team_matches(page, team_id, team_slug)
                    if df is not None:
                        all_data.append(df)
                except Exception as e:
                    print(f"Error scraping team {team_slug} (ID: {team_id}): {e}")
                time.sleep(2)

            if all_data:
                full_df = pd.concat(all_data, ignore_index=True)
                full_df.to_csv("d1_all_match_results.csv", index=False)
                print(f"Saved full dataset with {len(full_df)} total matches to d1_all_match_results.csv")
            browser.close()"""

            scrape_wrestler_matches(page, 78997, 'Connor Pierce', 'pierce-connor').to_csv("example.csv", index=False)
            #scrape_team_matches(page, 47, 'nc-state')

scrape_all_d1_teams()

        