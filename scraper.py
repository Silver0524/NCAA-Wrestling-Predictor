"""WrestleStat NCAA Division 1 Wrestling Data Scraper.

This module provides functions to scrape and compile NCAA Division 1 wrestling data from WrestleStat,
including team rosters, individual wrestler match histories, and aggregated team match results. It
handles authentication, data parsing, and exports match data to CSV files for analysis.

Key functions:
- login: Authenticates a user on WrestleStat.
- get_all_d1_teams: Retrieves all active D1 wrestling teams.
- get_team_roster: Fetches the roster for a specified team.
- scrape_wrestler_matches: Scrapes all matches for an individual wrestler.
- scrape_team_matches: Compiles match data for all wrestlers on a team.
- scrape_all_d1_teams: Scrapes and compiles match data for all D1 teams.

Usage example:

    scrape_all_d1_teams()
"""

# import packages
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import pandas as pd
import time
from tqdm import tqdm
import re
from dotenv import load_dotenv
import os


def login(page, email, password):
    """Logs in to the WrestleStat website.

    Logs in to the WrestleStat website using an email and password provided by the user.
    Navigates to the login page, fills in credentials, submits the form, and waits until the 
    homepage loads successfully.

    Args:
        page: A Page instance created by a Playwright Browser
        email: The user's email address that will be used to log in
        password: The user's password that will be used to log in

    Raises:
        May raise a timeout or navigation error if the login page fails to load, the credentials 
        are incorrect, or the site structure has changed. These issues may cause the login to fail 
        silently or result in an exception from Playwright.
    """
    
    page.goto("https://www.wrestlestat.com/account/login")
    page.fill("input[name='Username']", email)
    page.fill("input[name='Password']", password)
    page.click("button[type='submit']")
    page.wait_for_url("https://www.wrestlestat.com/", timeout=10000)

def get_all_d1_teams(page):
    """Scrapes the WrestleStat team rankings for a complete list of all NCAA D1 wrestling programs

    Navigates to WrestleStat's team rankings page and accesses a table containing links for all
    team pages on WrestleStat. Parses the links into tuples containing a unique team ID and team
    slug. Returns a list containing tuples for all active D1 wrestling programs.
    
    Args:
        page: A Page instance created by a Playwright Browser

    Returns:
        A list containing tuples representing all active NCAA D1 wrestling programs. Each
        tuple contains an integer representing a unique team ID and a string representing the team 
        slug (a URL-friendly name of the team), which are used to access the team's home page. 
        For example:

        [(47, 'nc-state'),
         (60, 'penn-state'),
         (34, 'iowa'),
         (57, 'oklahoma-state')]

    Raises:
        Prints warnings or fails silently if the WrestleStat rankings page fails to load properly or if 
        the HTML structure changes in a way that prevents team links from being parsed correctly.
        In such cases, an incomplete or empty list may be returned.
    """

    url = "https://www.wrestlestat.com/d1/rankings/dual"
    page.goto(url)
    soup = BeautifulSoup(page.content(), "html.parser")

    teams = []
    for link in soup.select('td a[href^="/team/"]', class_='table table-tight'):
        href = link['href']
        if '/profile' in href:
            parts = href.split('/')
            team_id = int(parts[2])
            team_slug = parts[3]
            teams.append((team_id, team_slug))
    return list(set(teams))

def get_team_roster(page, team_id, team_slug):
    """Scrapes the WrestleStat roster page for a given NCAA D1 wrestling team

    Navigates to a specific team's profile page on WrestleStat using the team ID and slug.
    Parses the roster table to extract individual wrestler information. For each valid
    wrestler entry, collects the wrestler's unique ID, name, and URL slug. Returns a list
    of tuples containing this information for all wrestlers currently listed on the team’s
    page.

    Args:
        page: A Page instance created by a Playwright Browser
        team_id: An integer representing the team’s unique ID on WrestleStat
        team_slug: A string representing the URL-friendly name of the team (e.g., 'penn-state')

    Returns:
        A list of tuples containing wrestler information for the specified team. Each tuple
        includes an integer ID for the wrestler, a string for their full name in "First Last"
        format, and a string for their slug used in URLs. For example:

        [(131567, 'Carter Starocci', 'starocci-carter'),
         (131570, 'Aaron Brooks', 'brooks-aaron')]
    """

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

def scrape_wrestler_matches(page, wrestler_id, wrestler_name, wrestler_slug):
    """Scrapes the match history for a specific NCAA D1 wrestler from WrestleStat

    Navigates to an individual wrestler's profile page on WrestleStat using their unique ID
    and URL slug. Parses all season blocks and corresponding match tables to extract detailed
    match information, including opponent data, match result, event, weight class, and score.
    Filters out malformed rows and incomplete data entries. Cleans opponent names and schools,
    and attaches metadata such as wrestler name and ID to each match.

    Args:
        page: A Page instance created by a Playwright Browser
        wrestler_id: An integer representing the wrestler’s unique ID on WrestleStat
        wrestler_name: A string containing the full name of the wrestler (used for tagging matches)
        wrestler_slug: A URL-friendly string representing the wrestler’s name (e.g., 'starocci-carter')

    Returns:
        A pandas DataFrame containing all parsed and cleaned match data for the given wrestler.
        Each row in the DataFrame represents a match and includes fields such as:

        - Season
        - Date
        - Event
        - Weight Class
        - Result
        - Result Type
        - Score
        - Opponent
        - Opponent ID
        - Opponent Record
        - Opponent School
        - Wrestler
        - Wrestler ID

    Raises:
        Prints a warning message for any row that cannot be parsed due to unexpected structure
        or data formatting issues. These rows are skipped, and scraping continues without interruption.
    """

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

        # Access all the columns for each match in the match table
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) != 9:
                continue

            opponent_a = cols[1].find("a")
            if not opponent_a:
                continue

            # Attempt to parse columns into clean formatting
            try:
                opponent_raw_name = opponent_a.text.strip()
                opponent_url = opponent_a['href']
                opponent_id = int(opponent_url.strip('/').split('/')[1])
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
                    "Opponent ID": opponent_id,
                    "Opponent Record": opponent_record,
                    "Opponent School": cleaned_school,
                    "Wrestler": wrestler_name,
                    "Wrestler ID": wrestler_id
                }
                all_matches.append(match)
            except Exception as e:
                print(f"⚠️ Error parsing match row: {e}")
                continue
    
    # Convert to DataFrame and handle duplicates and missing values
    df = pd.DataFrame(all_matches)
    df = df.drop_duplicates()
    df.replace("", pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    return df


def scrape_team_matches(page, team_id, team_slug, delay=1.0):
    """Scrapes and compiles all match data for a specific NCAA D1 wrestling team from WrestleStat

    Retrieves the full active roster for a given team using its team ID and slug, then iteratively
    scrapes each wrestler’s individual match history. Filters out empty or missing match data, compiles
    all valid match DataFrames, and saves the combined results as a CSV file in the 'Team Results' directory.

    Args:
        page: A Page instance created by a Playwright Browser
        team_id: An integer representing the team’s unique ID on WrestleStat
        team_slug: A string representing the URL-friendly name of the team (e.g., 'penn-state')
        delay: A float specifying the number of seconds to wait between scraping each wrestler (default is 1.0)

    Returns:
        A pandas DataFrame containing all match data for valid wrestlers on the team.
        Each row represents an individual match and includes fields such as:

        - Season
        - Date
        - Event
        - Weight Class
        - Result
        - Result Type
        - Score
        - Opponent
        - Opponent ID
        - Opponent Record
        - Opponent School
        - Wrestler
        - Wrestler ID

        Returns `None` if no valid match data could be retrieved.

    Raises:
        Prints warning messages when individual wrestler pages cannot be parsed correctly or yield no results.
        Match DataFrames that are empty or contain only incomplete rows are skipped.
        No exceptions are raised directly, allowing scraping to continue for other wrestlers.
    """

    roster = get_team_roster(page, team_id, team_slug)
    print(f"Found {len(roster)} wrestlers for {team_slug}...")

    all_matches = []

    # Scrape individual wrestlers from a team's roster
    for wrestler_id, wrestler_name, wrestler_slug in tqdm(roster, desc=f"Scraping {team_slug.title()}"):
        df = scrape_wrestler_matches(page, wrestler_id, wrestler_name, wrestler_slug)
        if df is not None and not df.empty:
            all_matches.append(df)
        time.sleep(delay)

    # Convert scraped matches into a DataFrame and save as a CSV file
    if all_matches:
        full_df = pd.concat(all_matches, ignore_index=True)
        full_df.to_csv(f"Team Results/{team_slug}_match_results.csv", index=False)
        print(f"Saved {len(full_df)} matches to {team_slug}_match_results.csv")
        return full_df
    else:
        print(f"No match data found for team {team_slug}.")
        return None

def scrape_all_d1_teams():
    """Scrapes match data for all NCAA D1 wrestling programs from WrestleStat

    Authenticates into WrestleStat using credentials stored in environment variables, launches a
    browser instance, and retrieves the list of all active Division 1 wrestling teams. Iteratively
    scrapes each team's full match data by calling `scrape_team_matches`. Filters out teams that fail
    or return no match data. Compiles all valid data into a single DataFrame and exports the full dataset
    to 'd1_all_match_results.csv'.

    This function is designed to be run as a top-level batch scraper and will open a visible browser
    window for interaction.

    Args:
        None

    Returns:
        None. Writes the compiled match data for all teams to a CSV file in the working directory.

    Raises:
        Prints error messages when individual teams fail to scrape due to unexpected issues such as 
        page load failures, missing data, or parsing errors. These errors are caught and logged, allowing 
        the scraping process to continue uninterrupted for other teams.
    """

    load_dotenv()

    with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            # WrestleStat requires a logged in account to view wrestler's full match histories
            login(page, os.getenv('WRESTLESTAT_EMAIL'), os.getenv('WRESTLESTAT_PASSWORD'))

            all_data = []
            teams = get_all_d1_teams(page)
            
            # Scrape individual teams from list of all teams
            for team_id, team_slug in tqdm(teams, desc=f"Scraping teams"):
                try:
                    df = scrape_team_matches(page, team_id, team_slug)
                    if df is not None:
                        all_data.append(df)
                except Exception as e:
                    print(f"Error scraping team {team_slug} (ID: {team_id}): {e}")
                time.sleep(2)

            # Convert full match list to a DataFrame and save as a CSV file
            if all_data:
                full_df = pd.concat(all_data, ignore_index=True)
                full_df.to_csv("d1_all_match_results.csv", index=False)
                print(f"Saved full dataset with {len(full_df)} total matches to d1_all_match_results.csv")
            browser.close()

scrape_all_d1_teams()
