# Web Performance Dataset

This project is designed to collect, store, and analyze web performance data from a list of URLs. It includes a Python scraper and supports data storage in both CSV and JSON formats.

## Project Structure

- `scraper.py` — Python script to scrape web performance data from URLs.
- `urls.txt` — List of URLs to be scraped (one per line).
- `performance_data.csv` — Collected performance data in CSV format.
- `performance_data.json` — Collected performance data in JSON format.

## Usage

1. **Prepare URLs**
   - Add the URLs you want to analyze to `urls.txt`, one per line.

2. **Run the Scraper**
   - Make sure you have Python 3 installed.
   - Install any required dependencies (see below).
   - Run the scraper:
     ```bash
     python scraper.py
     ```
   - The script will read URLs from `urls.txt` and output results to `performance_data.csv` and/or `performance_data.json`.

3. **View Results**
   - Open `performance_data.csv` or `performance_data.json` to view the collected data.

## Dependencies

- Python 3.x
- (Add any additional dependencies here, e.g., `requests`, `beautifulsoup4`, etc. If not sure, check `scraper.py` for imports.)

Install dependencies with:
```bash
pip install -r requirements.txt
```
(If a `requirements.txt` file is not present, install packages manually.)

## License

This project is open source and available under the MIT License.
