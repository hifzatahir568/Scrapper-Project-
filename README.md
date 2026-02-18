Web Content Scraper (Google-based Project Scraper)
Overview:

This project is a Python-based web scraping tool that automates Google search and extracts structured textual and media content from multiple web pages related to specific projects.

It is designed to:

1)Search project-related information on Google

2)Visit relevant web pages automatically

3)Extract clean, organized content

4)Save the output in structured JSON format

The scraper is suitable for research, data analysis, content aggregation, and downstream processing (APIs, AI pipelines, or databases).

✨ Features
🔍 Automated Google Search

1)Builds Google search queries dynamically

2)Collects multiple result URLs per query

3)Skips unwanted or irrelevant domains

4)Handles Google consent screens

🌐 Selenium-Based Scraping

Uses real Chrome browser automation:

Supports non-headless browsing for stability

Opens links in new tabs to preserve sessions

Scrolls pages to load dynamic content

🧠 Content Extraction

Extracts:

1)Headings

2)Paragraph text

3)Lists

4)Tables

5)Organizes content into logical sections

6)Removes noisy or irrelevant elements

🖼️ Media Extraction

1)Extracts images from:

<img> tags

2)Lazy-loaded sources

3)Background styles

4)Captures image URLs and related metadata

5)Extracts linked PDF documents when available

📦 Structured Output

Outputs data in JSON format

Groups extracted data by:

Project

Source URL

Content sections

Automatically cleans empty or invalid fields

🛡️ Stability & Safety

Random delays to reduce blocking

Graceful handling of timeouts and failures

CAPTCHA detection support

Detailed logging for debugging

📁 Project Structure
scrapper project/
│
├── scrapper.py          # Main scraper script
├── .gitignore           # Ignored files (data, venv, cache)
├── README.md            # Project documentation


Note:
Generated data files (.json, .csv, .txt), virtual environments, and cache files are intentionally ignored.

⚙️ How It Works

Reads project-related input (configured inside the script)

Performs Google search queries

Collects relevant URLs

Visits each page using Selenium

Extracts structured text and media

Saves results into JSON files

🧰 Tech Stack

*Python 3

*Selenium

*BeautifulSoup

*Pandas

*webdriver-manager

*ChromeDriver

▶️ How to Run

Create and activate a virtual environment (recommended)

Install required dependencies

Run the scraper:

python scrapper.py


Make sure Google Chrome is installed on your system.
