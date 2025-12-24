import requests
import json
import time
import csv
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("PAGESPEED_API_KEY")

# Config
STRATEGY = "desktop"   # or "mobile"
API_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
DELAY_SECONDS = 2      # rate limiting

def fetch_pagespeed_metrics(url, retries=3, backoff_factor=1.0):
    params = {
        "url": url,
        "key": API_KEY,
        "strategy": STRATEGY,
        "category": "performance"
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            # Treat 5xx as retriable
            if response.status_code >= 500:
                response.raise_for_status()

            response.raise_for_status()
            data = response.json()

            lr = data.get("lighthouseResult", {})
            audits = lr.get("audits", {})

            def num(audit_key):
                v = audits.get(audit_key, {})
                if isinstance(v, dict):
                    return v.get("numericValue")
                return None

            def safe_list_count(audit_key):
                details = audits.get(audit_key, {}).get("details", {})
                items = details.get("items") or []
                return len(items)

            metrics = {
                "url": lr.get("finalUrl") or url,

                # Network
                "TTFB_ms": num("server-response-time"),

                # Rendering
                "FCP_ms": num("first-contentful-paint"),
                "LCP_ms": num("largest-contentful-paint"),
                "SpeedIndex_ms": num("speed-index"),

                # Execution
                "TBT_ms": num("total-blocking-time"),
                "TTI_ms": num("interactive"),

                # Resources
                "noOfRequests": safe_list_count("network-requests"),
                "pageSize_kb": (num("total-byte-weight") / 1024) if num("total-byte-weight") is not None else None,

                # JS execution cost (proxy)
                "javascriptExecution_ms": num("bootup-time"),

                # Overall load time
                "loadTime_ms": lr.get("timing", {}).get("total")
            }

            return metrics

        except requests.exceptions.HTTPError as e:
            status = None
            try:
                status = e.response.status_code
            except Exception:
                pass
            # Don't retry on 4xx errors (client errors)
            if status and 400 <= status < 500:
                print(f"Non-retriable HTTP error for {url}: {e}")
                raise

            if attempt < retries:
                wait = backoff_factor * (2 ** (attempt - 1))
                print(f"Request failed (attempt {attempt}/{retries}) for {url}: {e} — retrying in {wait}s")
                time.sleep(wait)
                continue
            raise

        except (requests.exceptions.RequestException, ValueError) as e:
            # Network errors, timeouts, or JSON decode errors
            if attempt < retries:
                wait = backoff_factor * (2 ** (attempt - 1))
                print(f"Request failed (attempt {attempt}/{retries}) for {url}: {e} — retrying in {wait}s")
                time.sleep(wait)
                continue
            raise


def load_urls(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def save_json(data, filename="performance_data.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def save_csv(data, filename="performance_data.csv"):
    if not data:
        return

    # Build a consistent set of fieldnames across all rows
    fieldnames = set()
    for row in data:
        fieldnames.update(row.keys())
    fieldnames = list(fieldnames)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def load_json_file(filename="performance_data.json"):
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: couldn't load existing JSON file '{filename}': {e}")
        return []


from urllib.parse import urlparse

def normalize_url(u):
    """Normalize URLs for matching: lower-case host, strip leading www., remove trailing slash and drop query/fragment."""
    if not u:
        return u
    try:
        if '://' not in u:
            u = 'http://' + u
        p = urlparse(u)
        host = p.netloc.lower()
        if host.startswith('www.'):
            host = host[4:]
        path = (p.path or '').rstrip('/')
        return host + path
    except Exception:
        return u.rstrip('/').lower()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch PageSpeed Insights metrics for a list of URLs")
    parser.add_argument("--urls-file", default="urls.txt", help="Path to the URLs file")
    parser.add_argument("--force", action="store_true", help="Force re-fetching even if results already exist")
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS, help="Delay between requests in seconds")
    parser.add_argument("--check", action="store_true", help="Only show how many would be fetched/skipped without performing requests")
    parser.add_argument("--no-resume", action="store_true", help="Disable resuming from the last recorded item in performance_data.json")
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("PAGESPEED_API_KEY is not set. Please set it in your environment or .env file.")

    urls = load_urls(args.urls_file)

    # Load existing results (preserve order)
    existing_list = load_json_file()

    # Build mapping of normalized keys to items (for both requested_url and final url)
    existing_map = {}
    for item in existing_list:
        for field in ("requested_url", "url"):
            val = item.get(field)
            if val:
                existing_map[normalize_url(val)] = item

    # Determine resume index from the last recorded item
    if getattr(args, "no_resume", False):
        start_idx = 0
        print("Resuming disabled by --no-resume; starting from index 0")
    elif existing_list:
        last_item = existing_list[-1]
        last_key = normalize_url(last_item.get("requested_url") or last_item.get("url"))
        found_idx = next((i for i, u in enumerate(urls) if normalize_url(u) == last_key), None)
        if found_idx is None:
            print(f"Warning: last URL '{last_key}' not found in '{args.urls_file}'; starting from 0")
            start_idx = 0
        else:
            start_idx = found_idx + 1
            print(f"Resuming from index {start_idx} (next URL: {urls[start_idx] if start_idx < len(urls) else 'end'})")
    else:
        start_idx = 0

    tail_urls = urls[start_idx:]

    to_fetch = []
    skipped = []
    for url in tail_urls:
        n = normalize_url(url)
        if not args.force and n in existing_map:
            skipped.append(url)
        else:
            to_fetch.append(url)

    print(f"Found {len(existing_list)} existing records. Starting at index {start_idx}. Skipping {len(skipped)} URLs in tail; {len(to_fetch)} to fetch.")

    # If user only wants to check what would be done, exit now
    if args.check:
        print("Check mode - exiting without fetching.")
        return

    # Process and append/replace entries in existing_list so order is preserved
    for i, url in enumerate(to_fetch, start=1):
        print(f"[{i}/{len(to_fetch)}] Fetching metrics for {url}")
        try:
            metrics = fetch_pagespeed_metrics(url)
            # Record the original requested URL so we can match next runs
            metrics["requested_url"] = url

            # Normalize key based on requested URL
            key_req = normalize_url(metrics.get("requested_url"))

            if key_req in existing_map:
                # Replace existing entry in the list
                for idx, item in enumerate(existing_list):
                    if normalize_url(item.get("requested_url") or item.get("url")) == key_req:
                        existing_list[idx] = metrics
                        break
            else:
                existing_list.append(metrics)

            # Update the mapping for both requested and final URL
            existing_map[key_req] = metrics
            if metrics.get("url"):
                existing_map[normalize_url(metrics.get("url"))] = metrics

            # Save progress after each successful fetch
            save_json(existing_list)
            save_csv(existing_list)
            print(f"Saved progress ({len(existing_list)} records).")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
        time.sleep(args.delay)

    save_json(existing_list)
    save_csv(existing_list)
    print(f"Done. Total records: {len(existing_list)} (skipped {len(skipped)}).")


if __name__ == "__main__":
    main()

