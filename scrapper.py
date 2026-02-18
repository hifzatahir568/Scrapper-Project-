
import os
import re
import json
import time
import random
import urllib.parse
import datetime
from typing import List, Optional, Dict, Tuple, Any

import pandas as pd
from bs4 import BeautifulSoup, Tag
import tldextract
from dotenv import load_dotenv

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


CSV_PATH = r"D:\xosoft\Off Load_Projects_minus_processed_part2.csv"
ENV_PATH = r"D:\xosoft\venv\.env"
OUTPUT_JSON_PATH = r"D:\xosoft\Off Load_Projects_minus_processed_part12.json"
LOG_FILE_PATH = r"D:\xosoft\Off Load_Projects_minus_processed_part12_{}.txt".format(
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)


MAX_PAGES_TO_PROCESS = 10
SERP_CANDIDATE_LIMIT = 50

HEADLESS = False

PAGE_LOAD_TIMEOUT = 60

WAIT_SECONDS_BETWEEN_URLS = (6.0, 12.0)
WAIT_SECONDS_BETWEEN_PROJECTS = (12.0, 25.0)
WAIT_SECONDS_AFTER_SERP = (6.0, 16.0)
WAIT_SECONDS_BEFORE_NEW_TAB = (1.2, 3.2)
WAIT_SECONDS_AFTER_TAB_OPEN = (1.2, 3.5)
WAIT_SECONDS_AFTER_SCROLL = (0.8, 2.4)
SCROLL_PAUSE = (0.8, 1.8)

CAPTCHA_COOLDOWN_SECONDS = (120, 240)
CAPTCHA_MAX_RETRIES_PER_PROJECT = 2
CAPTCHA_MANUAL_SOLVE_WAIT_SECONDS = 180

DROP_QUERY_KEYS = {"sa", "ved", "usg", "opi", "source", "ei", "oq", "gs_lcp", "gs_lp", "sclient", "rlz"}

SKIP_DOMAINS = {
    "bayut.com", "propertyfinder.ae", "dubizzle.com", "dubbizle.com", "houza.com",
    "betterhomes.ae", "metropolitan.realestate", "allproperties.ae", "zoomproperty.com",
    "realestate.finder", "gilfnews.com",
    "realtor.com", "zillow.com", "trulia.com",
    "facebook.com", "instagram.com", "linkedin.com", "tiktok.com", "youtube.com", "pinterest.com",
    "wikipedia.org", "maps.google.com", "goo.gl", "google.com", "support.google.com",
    "yellowpages.ae",
}

LIMITS = {
    "max_sections": 900,
    "max_blocks_total": 3000,
    "max_block_text_chars": 2200,
    "max_section_blocks": 350,

    "max_images_total": 600,
    "max_images_per_section": 120,

    "max_pdfs_total": 160,
    "max_pdfs_per_section": 60,

    "max_title_chars": 220,
    "max_nearby_text_chars": 350,
    "max_attr_len": 300,
}


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    print(line.strip())
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(i) for i in x if i is not None)
    return str(x)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def domain_of(url: str) -> str:
    ext = tldextract.extract(url)
    if not ext.domain and not ext.suffix:
        return ""
    return f"{ext.domain}.{ext.suffix}".lower()


def should_skip_url(url: str) -> bool:
    d = domain_of(url)
    if not d:
        return True
    return d in SKIP_DOMAINS


def is_pdf_url(url: str) -> bool:
    u = (url or "").lower().split("?")[0].split("#")[0]
    return u.endswith(".pdf")


def absolutize_url(base_url: str, link: str) -> str:
    return urllib.parse.urljoin(base_url, link)


def normalize_google_url(href: str) -> Optional[str]:
    if not href:
        return None
    if href.startswith("/url?"):
        qs = urllib.parse.urlparse(href).query
        params = urllib.parse.parse_qs(qs)
        if "q" in params and params["q"]:
            return params["q"][0]
        return None
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return None


def polite_sleep(rng: Tuple[float, float]):
    time.sleep(random.uniform(rng[0], rng[1]))


def strip_text_fragment(url: str) -> str:
    if not url:
        return url
    return url.split("#:~:text=")[0]


def canonicalize_for_dedupe(url: str) -> str:
    try:
        url = strip_text_fragment(url)
        parsed = urllib.parse.urlsplit(url)
    except Exception:
        return url
    fragment = ""
    q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    q2 = [(k, v) for (k, v) in q if k.lower() not in DROP_QUERY_KEYS]
    query = urllib.parse.urlencode(q2, doseq=True)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, fragment))


def dedupe_keep_longest(urls: List[str]) -> List[str]:
    best: Dict[str, str] = {}
    for u in urls:
        key = canonicalize_for_dedupe(u)
        if key not in best or len(u) > len(best[key]):
            best[key] = u
    return list(best.values())


def prune_empty(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            cleaned = prune_empty(v)
            if cleaned is None:
                continue
            if cleaned == "":
                continue
            if isinstance(cleaned, (dict, list)) and not cleaned:
                continue
            new_dict[k] = cleaned
        return new_dict

    if isinstance(obj, list):
        new_list = []
        for v in obj:
            cleaned = prune_empty(v)
            if cleaned is None:
                continue
            if cleaned == "":
                continue
            if isinstance(cleaned, (dict, list)) and not cleaned:
                continue
            new_list.append(cleaned)
        return new_list

    return obj


def is_google_sorry_url(url: str) -> bool:
    u = (url or "").lower()
    return "google.com/sorry" in u or "/sorry/" in u


def captcha_cooldown(reason: str):
    secs = random.randint(CAPTCHA_COOLDOWN_SECONDS[0], CAPTCHA_COOLDOWN_SECONDS[1])
    log(f"CAPTCHA detected ({reason}). Cooling down for {secs} seconds.")
    time.sleep(secs)


def wait_for_manual_solve_if_possible(driver: webdriver.Chrome, reason: str) -> bool:
    if HEADLESS:
        return False
    log(f"CAPTCHA detected ({reason}). Solve in browser if shown. Waiting up to {CAPTCHA_MANUAL_SOLVE_WAIT_SECONDS}s.")
    deadline = time.time() + CAPTCHA_MANUAL_SOLVE_WAIT_SECONDS
    while time.time() < deadline:
        time.sleep(2.5)
        try:
            cur_url = driver.current_url or ""
            if not is_google_sorry_url(cur_url):
                log("CAPTCHA cleared.")
                return True
        except Exception:
            pass
    log("CAPTCHA still present after wait.")
    return False


def make_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--lang=en-US,en")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.page_load_strategy = "eager"

    user_data_dir = os.getenv("CHROME_USER_DATA_DIR", "").strip()
    if not user_data_dir:
        user_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_chrome_user_data")
    os.makedirs(user_data_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument("--profile-directory=Default")

    if HEADLESS:
        options.add_argument("--headless=new")

    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()),
        options=options
    )
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def google_search_collect_urls(driver: webdriver.Chrome, query: str, max_urls: int) -> List[str]:
    search_url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    driver.get(search_url)

    time.sleep(random.uniform(2.5, 5.5))
    try:
        driver.execute_script("window.scrollBy(0, arguments[0]);", random.randint(200, 900))
        time.sleep(random.uniform(0.8, 2.0))
        driver.execute_script("window.scrollBy(0, arguments[0]);", random.randint(200, 900))
        time.sleep(random.uniform(0.8, 2.0))
    except Exception:
        pass

    try:
        WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        for xpath in [
            "//button//div[contains(text(),'Accept all')]",
            "//button[contains(., 'Accept all')]",
            "//button[contains(., 'I agree')]",
            "//button[contains(., 'Agree')]",
        ]:
            btns = driver.find_elements(By.XPATH, xpath)
            if btns:
                btns[0].click()
                time.sleep(random.uniform(1.0, 2.0))
                break
    except Exception:
        pass

    try:
        cur_url = driver.current_url or ""
        if is_google_sorry_url(cur_url):
            cleared = wait_for_manual_solve_if_possible(driver, "google_main_tab")
            if not cleared:
                captcha_cooldown("google_main_tab")
                return []
    except Exception:
        pass

    urls: List[str] = []
    anchors = driver.find_elements(By.CSS_SELECTOR, "a")
    for a in anchors:
        href = a.get_attribute("href")
        if not href:
            continue
        normalized = normalize_google_url(href) or href
        if not normalized.startswith(("http://", "https://")):
            continue
        if "google.com" in domain_of(normalized):
            continue
        urls.append(normalized)
        if len(urls) >= max_urls:
            break

    return dedupe_keep_longest(urls)


def fetch_page_source_in_new_tab(driver: webdriver.Chrome, url: str, google_handle: str) -> Tuple[str, str]:
    url = strip_text_fragment(url)
    log(f"Opening new tab for URL: {url}")
    polite_sleep(WAIT_SECONDS_BEFORE_NEW_TAB)

    driver.execute_script("window.open(arguments[0], '_blank');", url)
    new_handle = driver.window_handles[-1]
    driver.switch_to.window(new_handle)

    polite_sleep(WAIT_SECONDS_AFTER_TAB_OPEN)

    final_url = url
    html = ""

    try:
        try:
            WebDriverWait(driver, 18).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        except Exception:
            pass

        time.sleep(random.uniform(1.0, 3.0))

        for _ in range(6):
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight/5);")
            polite_sleep(SCROLL_PAUSE)

        polite_sleep(WAIT_SECONDS_AFTER_SCROLL)

        final_url = driver.current_url or url
        html = driver.page_source or ""

    except TimeoutException as e:
        log(f"Timeout while loading. window.stop(). Error: {e}")
        try:
            driver.execute_script("window.stop();")
            time.sleep(random.uniform(0.8, 1.6))
        except Exception:
            pass
        try:
            final_url = driver.current_url or url
            html = driver.page_source or ""
        except Exception:
            html = ""

    except WebDriverException as e:
        log(f"WebDriver error during page fetch: {e}")
        try:
            final_url = driver.current_url or url
        except Exception:
            final_url = url
        try:
            html = driver.page_source or ""
        except Exception:
            html = ""

    finally:
        log(f"Closing tab: {url}")
        try:
            driver.close()
        except Exception:
            pass
        driver.switch_to.window(google_handle)

    return final_url, html


HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def guess_canonical_url(soup: BeautifulSoup) -> str:
    link = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
    if link and link.get("href"):
        return safe_text(link.get("href")).strip()
    return ""


def choose_content_root(soup: BeautifulSoup) -> Tag:
    selectors = [
        "main",
        "article",
        "[role='main']",
        "#content",
        "#main",
        ".content",
        ".main-content",
        ".post-content",
        ".entry-content",
        ".container",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if isinstance(el, Tag):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 200:
                return el
    return soup.body if soup.body else soup


def clean_soup_for_content(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    for tname in ["button", "input", "select", "option", "textarea"]:
        for el in soup.find_all(tname):
            el.decompose()

    for el in soup.find_all(True, attrs={"id": re.compile(r"(cookie|consent)", re.I)}):
        el.decompose()
    for el in soup.find_all(True, attrs={"class": re.compile(r"(cookie|consent|gdpr|overlay|modal)", re.I)}):
        txt = el.get_text(" ", strip=True)
        if txt and len(txt) < 800:
            el.decompose()


def text_is_junk(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    if len(s) <= 2:
        return True
    low = s.lower()
    if low in {"prev", "next", "share", "home", "menu"}:
        return True
    return False


def looks_like_country_code_spam(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    return bool(re.search(r"\+\d{1,4}\b", s)) and len(s) < 60


def get_visible_text(el: Tag) -> str:
    return el.get_text(" ", strip=True)


def extract_urls_from_srcset(srcset: str) -> List[str]:
    out = []
    if not srcset:
        return out
    parts = [p.strip() for p in srcset.split(",")]
    for p in parts:
        if not p:
            continue
        url = p.split()[0].strip()
        if url:
            out.append(url)
    return out


_URL_IN_STYLE_RE = re.compile(r'url\(([^)]+)\)', re.I)


def normalize_media_url(base_url: str, raw: str) -> Optional[str]:
    raw = (raw or "").strip().strip('"').strip("'")
    if not raw:
        return None
    full = absolutize_url(base_url, raw)
    if not full.lower().startswith(("http://", "https://")):
        return None
    return full


def image_record(
    url: str,
    source: str,
    alt: Optional[str],
    title: Optional[str],
    nearby_text: Optional[str],
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    rec = {
        "url": url,
        "source": source,
        "alt": alt or None,
        "title": title or None,
        "nearby_text": (nearby_text[:LIMITS["max_nearby_text_chars"]] if nearby_text else None),
    }
    if extra:
        rec.update(extra)
    return rec


def extract_images_nearby_context(base_url: str, root: Tag) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    def add(url: str, source: str, img_tag: Optional[Tag], nearby: Optional[str]):
        if not url:
            return
        if url in seen:
            return
        seen.add(url)

        alt = None
        title = None
        extra = {}

        if img_tag is not None:
            alt = safe_text(img_tag.get("alt")).strip() or None
            title = safe_text(img_tag.get("title")).strip() or None
            extra = {
                "width": safe_text(img_tag.get("width")).strip()[:LIMITS["max_attr_len"]] or None,
                "height": safe_text(img_tag.get("height")).strip()[:LIMITS["max_attr_len"]] or None,
                "class": safe_text(img_tag.get("class")).strip()[:LIMITS["max_attr_len"]] or None,
            }

        out.append(image_record(url=url, source=source, alt=alt, title=title, nearby_text=nearby, extra=extra))

    for img in root.find_all("img"):
        nearby = None

        fig = img.find_parent("figure")
        if fig:
            cap = fig.find("figcaption")
            if cap:
                nearby = get_visible_text(cap)

        if not nearby:
            parent = img.parent if isinstance(img.parent, Tag) else None
            if parent:
                nearby = get_visible_text(parent)

        candidates: List[Tuple[str, str]] = []

        src = safe_text(img.get("src")).strip()
        if src:
            candidates.append((src, "src"))

        for k in ["data-src", "data-lazy-src", "data-original", "data-url", "data-img", "data-image"]:
            v = safe_text(img.get(k)).strip()
            if v:
                candidates.append((v, k))

        srcset = safe_text(img.get("srcset")).strip()
        if srcset:
            for u in extract_urls_from_srcset(srcset):
                candidates.append((u, "srcset"))

        dsrcset = safe_text(img.get("data-srcset")).strip()
        if dsrcset:
            for u in extract_urls_from_srcset(dsrcset):
                candidates.append((u, "data-srcset"))

        for raw, source in candidates:
            full = normalize_media_url(base_url, raw)
            if full:
                add(full, source, img, nearby)

        if len(out) >= LIMITS["max_images_total"]:
            break

    if len(out) < LIMITS["max_images_total"]:
        for pic in root.find_all("picture"):
            nearby = get_visible_text(pic)[:LIMITS["max_nearby_text_chars"]] or None
            for src_el in pic.find_all("source"):
                ss = safe_text(src_el.get("srcset")).strip()
                if ss:
                    for u in extract_urls_from_srcset(ss):
                        full = normalize_media_url(base_url, u)
                        if full:
                            add(full, "picture/source-srcset", None, nearby)
                if len(out) >= LIMITS["max_images_total"]:
                    break
            if len(out) >= LIMITS["max_images_total"]:
                break

    if len(out) < LIMITS["max_images_total"]:
        for el in root.find_all(True):
            style = safe_text(el.get("style")).strip()
            if style:
                for m in _URL_IN_STYLE_RE.findall(style):
                    full = normalize_media_url(base_url, m)
                    if full:
                        nearby = get_visible_text(el)[:LIMITS["max_nearby_text_chars"]] or None
                        add(full, "style-url()", None, nearby)

            for k in ["data-bg", "data-background", "data-bg-image", "data-background-image"]:
                v = safe_text(el.get(k)).strip()
                if v:
                    full = normalize_media_url(base_url, v)
                    if full:
                        nearby = get_visible_text(el)[:LIMITS["max_nearby_text_chars"]] or None
                        add(full, k, None, nearby)

            if len(out) >= LIMITS["max_images_total"]:
                break

    return out


def extract_pdfs_with_labels(base_url: str, root: Tag) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for a in root.find_all("a", href=True):
        href = safe_text(a.get("href")).strip()
        if not href:
            continue
        full = absolutize_url(base_url, href)
        if not full.lower().startswith(("http://", "https://")):
            continue
        if not (is_pdf_url(full) or "pdf" in full.lower()):
            continue
        if full in seen:
            continue
        seen.add(full)
        label = (a.get_text(" ", strip=True) or "")[:200] or None
        out.append({"url": full, "label": label})
        if len(out) >= LIMITS["max_pdfs_total"]:
            break
    return out


def bucket_images(images: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {
        "floorplan_candidates": [],
        "master_plan_candidates": [],
        "location_map_candidates": [],
        "gallery_candidates": [],
        "other_images": [],
    }

    for it in images:
        url = (it.get("url") or "").lower()
        ctx = " ".join([
            (it.get("alt") or ""),
            (it.get("title") or ""),
            (it.get("nearby_text") or ""),
            url
        ]).lower()

        if any(k in ctx for k in ["floor plan", "floorplan", "floor-plans", "layout", "unit plan"]):
            buckets["floorplan_candidates"].append(it)
        elif any(k in ctx for k in ["master plan", "masterplan"]):
            buckets["master_plan_candidates"].append(it)
        elif any(k in ctx for k in ["location map", "google map", "directions", "map"]):
            buckets["location_map_candidates"].append(it)
        elif any(k in ctx for k in ["gallery", "carousel", "slider"]):
            buckets["gallery_candidates"].append(it)
        else:
            buckets["other_images"].append(it)

    return buckets


def element_signature_text(el: Tag) -> str:
    txt = normalize_ws(el.get_text(" ", strip=True))
    return txt[:800]


def has_meaningful_child(el: Tag) -> bool:
    return bool(el.find(["p", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6"]))


def extract_kv_rows_from_container(el: Tag) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in el.find_all("div", recursive=False):
        spans = row.find_all("span", recursive=True)
        if len(spans) < 2:
            continue

        key = normalize_ws(spans[0].get_text(" ", strip=True))
        val = normalize_ws(spans[-1].get_text(" ", strip=True))

        if not key or not val:
            continue
        if len(key) > 80:
            continue
        if len(val) > 400:
            val = val[:400]

        rows.append({"key": key, "value": val})
        if len(rows) >= 40:
            break
    return rows


def build_sections_tree_with_media(base_url: str, root: Tag) -> Dict[str, Any]:
    tree = {
        "title": "ROOT",
        "level": 0,
        "blocks": [],
        "children": [],
        "images": [],
        "pdfs": [],
    }

    stack = [tree]
    sections_total = 1
    blocks_total = 0

    TAGS_OF_INTEREST = set(list(HEADING_TAGS) + [
        "p", "ul", "ol", "table", "img", "a", "picture", "figure", "div", "section", "article"
    ])

    def find_section_path(stack_: List[Dict[str, Any]]) -> List[str]:
        return [n["title"] for n in stack_[1:] if n.get("title")]

    def add_image_to_current(img_item: Dict[str, Any]):
        if len(stack[-1]["images"]) >= LIMITS["max_images_per_section"]:
            return
        item = dict(img_item)
        item["section_path"] = find_section_path(stack)
        stack[-1]["images"].append(item)

    def add_pdf_to_current(pdf_item: Dict[str, Any]):
        if len(stack[-1]["pdfs"]) >= LIMITS["max_pdfs_per_section"]:
            return
        item = dict(pdf_item)
        item["section_path"] = find_section_path(stack)
        stack[-1]["pdfs"].append(item)

    all_images = extract_images_nearby_context(base_url, root)
    all_pdfs = extract_pdfs_with_labels(base_url, root)

    img_by_url = {x["url"]: x for x in all_images}
    pdf_by_url = {x["url"]: x for x in all_pdfs}

    used_imgs = set()
    used_pdfs = set()

    section_text_fingerprints: Dict[int, set] = {}

    def current_fp_set() -> set:
        sid = id(stack[-1])
        if sid not in section_text_fingerprints:
            section_text_fingerprints[sid] = set()
        return section_text_fingerprints[sid]

    def add_paragraph_block(txt: str):
        nonlocal blocks_total
        txt = normalize_ws(txt)
        if text_is_junk(txt) or looks_like_country_code_spam(txt):
            return
        txt = txt[:LIMITS["max_block_text_chars"]]
        fp = txt[:260]
        fpset = current_fp_set()
        if fp in fpset:
            return
        fpset.add(fp)
        stack[-1]["blocks"].append({"type": "paragraph", "text": txt})
        blocks_total += 1

    def add_kv_block(rows: List[Dict[str, str]], title: Optional[str] = None):
        nonlocal blocks_total
        if not rows:
            return
        stack[-1]["blocks"].append({
            "type": "key_values",
            "title": title or None,
            "rows": rows
        })
        blocks_total += 1

    for el in root.find_all(True, recursive=True):
        name = (el.name or "").lower()
        if name not in TAGS_OF_INTEREST:
            continue

        if name in HEADING_TAGS:
            title = normalize_ws(get_visible_text(el))
            if text_is_junk(title):
                continue

            level = int(name[1])
            node = {
                "title": title[:LIMITS["max_block_text_chars"]],
                "level": level,
                "blocks": [],
                "children": [],
                "images": [],
                "pdfs": [],
            }

            while stack and stack[-1]["level"] >= level:
                stack.pop()

            parent = stack[-1] if stack else tree
            parent["children"].append(node)
            stack.append(node)

            sections_total += 1
            if sections_total >= LIMITS["max_sections"]:
                break
            continue

        if name == "p":
            txt = normalize_ws(get_visible_text(el))
            if text_is_junk(txt) or looks_like_country_code_spam(txt):
                continue
            add_paragraph_block(txt)
            if blocks_total >= LIMITS["max_blocks_total"]:
                break
            continue

        if name in {"ul", "ol"}:
            items = []
            for li in el.find_all("li", recursive=False):
                t = normalize_ws(get_visible_text(li))
                if text_is_junk(t) or looks_like_country_code_spam(t):
                    continue
                items.append(t[:LIMITS["max_block_text_chars"]])
                if len(items) >= 60:
                    break
            if items:
                stack[-1]["blocks"].append({
                    "type": "list",
                    "ordered": (name == "ol"),
                    "items": items
                })
                blocks_total += 1
                if blocks_total >= LIMITS["max_blocks_total"]:
                    break
            continue

        if name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cols = [normalize_ws(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
                cols = [c for c in cols if c]
                if cols:
                    rows.append(cols)
                if len(rows) >= 35:
                    break
            if rows:
                stack[-1]["blocks"].append({"type": "table", "rows": rows})
                blocks_total += 1
                if blocks_total >= LIMITS["max_blocks_total"]:
                    break
            continue

        if name in {"div", "section", "article"}:
            kv_rows = extract_kv_rows_from_container(el)
            if len(kv_rows) >= 3:
                title_el = el.find(["h2", "h3", "h4"])
                kv_title = normalize_ws(title_el.get_text(" ", strip=True)) if title_el else None
                if kv_title and len(kv_title) > 80:
                    kv_title = kv_title[:80]
                add_kv_block(kv_rows, title=kv_title)
                if blocks_total >= LIMITS["max_blocks_total"]:
                    break
                continue

            txt = normalize_ws(el.get_text(" ", strip=True))
            if len(txt) < 220:
                continue

            if has_meaningful_child(el):
                cls = " ".join(el.get("class", [])).lower()
                if not any(k in cls for k in ["prose", "leading", "space-y", "text-", "content"]):
                    continue

            sig = element_signature_text(el)
            fpset = current_fp_set()
            if sig in fpset:
                continue
            fpset.add(sig)

            add_paragraph_block(txt)
            if blocks_total >= LIMITS["max_blocks_total"]:
                break
            continue

        if name == "img":
            candidates = []
            for k in ["src", "data-src", "data-lazy-src", "data-original", "data-url", "data-image"]:
                v = safe_text(el.get(k)).strip()
                if v:
                    candidates.append(v)
            ss = safe_text(el.get("srcset")).strip()
            if ss:
                candidates.extend(extract_urls_from_srcset(ss))

            for raw in candidates:
                full = normalize_media_url(base_url, raw)
                if full and full in img_by_url and full not in used_imgs:
                    add_image_to_current(img_by_url[full])
                    used_imgs.add(full)
            continue

        if name == "a" and el.get("href"):
            href = safe_text(el.get("href")).strip()
            full = absolutize_url(base_url, href)
            if full.lower().startswith(("http://", "https://")) and (is_pdf_url(full) or "pdf" in full.lower()):
                if full in pdf_by_url and full not in used_pdfs:
                    add_pdf_to_current(pdf_by_url[full])
                    used_pdfs.add(full)
            continue

    for img in all_images:
        if img["url"] in used_imgs:
            continue
        tree["images"].append({**img, "section_path": []})
    for pdf in all_pdfs:
        if pdf["url"] in used_pdfs:
            continue
        tree["pdfs"].append({**pdf, "section_path": []})

    tree["_stats"] = {
        "sections_total": sections_total,
        "blocks_total": blocks_total,
        "images_total": len(all_images),
        "pdfs_total": len(all_pdfs),
    }

    tree["_media_buckets_global"] = {
        "images": bucket_images(all_images),
        "pdfs": all_pdfs
    }

    return tree


def extract_content_structured(page_url: str, html: str) -> Dict[str, Any]:
    if not html or not html.strip():
        return {
            "source_url": page_url,
            "title": None,
            "canonical_url": None,
            "sections_tree": None,
            "stats": {"error": "empty_html"},
        }

    soup = BeautifulSoup(html, "html.parser")
    clean_soup_for_content(soup)

    title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:LIMITS["max_title_chars"]]
    canonical = guess_canonical_url(soup)

    content_root = choose_content_root(soup)
    sections_tree = build_sections_tree_with_media(page_url, content_root)

    st = sections_tree.get("_stats", {})
    return {
        "source_url": page_url,
        "title": title or None,
        "canonical_url": canonical or None,
        "content_root_hint": content_root.name if isinstance(content_root, Tag) else None,
        "sections_tree": sections_tree,
        "stats": {
            "sections_total": st.get("sections_total", 0),
            "blocks_total": st.get("blocks_total", 0),
            "images_total": st.get("images_total", 0),
            "pdfs_total": st.get("pdfs_total", 0),
        }
    }


def main():
    load_dotenv(ENV_PATH)

    log("Script started. Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    required_cols = {"project_name_en_resolved", "developer_name_en"}
    if not required_cols.issubset(set(df.columns)):
        log(f"CSV missing required columns: {required_cols}")
        return

    driver = make_driver()
    log("Selenium driver initialized.")

    output: Dict[str, Any] = {
        "meta": {
            "csv_path": CSV_PATH,
            "generated_at_epoch": int(time.time()),
            "output_mode": "content_only_nested_v2_section_media_with_div_text_and_kv",
            "max_pages_to_process_per_project": MAX_PAGES_TO_PROCESS,
            "serp_candidate_limit": SERP_CANDIDATE_LIMIT,
            "skip_domains": sorted(list(SKIP_DOMAINS)),
            "limits": LIMITS,
            "selenium": {
                "page_load_timeout": PAGE_LOAD_TIMEOUT,
                "page_load_strategy": "eager",
                "headless": HEADLESS,
            }
        },
        "projects": []
    }

    def save_output():
        clean_output = prune_empty(output)
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(clean_output, f, ensure_ascii=False, indent=2)

    try:
        for idx, row in df.iterrows():
            project_name = str(row["project_name_en_resolved"]).strip()
            developer_name = str(row["developer_name_en"]).strip()

            if not project_name or project_name.lower() == "nan":
                continue

            query = f"{project_name} {developer_name} Offplan Project Dubai"
            log(f"\n[{idx+1}/{len(df)}] Project: {project_name}")
            log(f"  Developer: {developer_name}")
            log(f"  Google query: {query}")

            urls: List[str] = []
            serp_attempt = 0
            while serp_attempt <= CAPTCHA_MAX_RETRIES_PER_PROJECT:
                serp_attempt += 1
                urls = google_search_collect_urls(driver, query, SERP_CANDIDATE_LIMIT)
                if urls:
                    break
                log(f"  No URLs returned (possible Google captcha). SERP attempt {serp_attempt}/{CAPTCHA_MAX_RETRIES_PER_PROJECT+1}")
                if serp_attempt <= CAPTCHA_MAX_RETRIES_PER_PROJECT:
                    captcha_cooldown("serp_retry")
                    try:
                        driver.get("https://www.google.com/")
                        time.sleep(random.uniform(2.0, 5.0))
                    except Exception:
                        pass

            if not urls:
                log("  Skipping project due to repeated Google captcha/blocks (no URLs).")
                project_record = {
                    "project_name_en_resolved": project_name,
                    "developer_name_en": developer_name,
                    "google_query": query,
                    "candidate_urls": [],
                    "processed_pages_target": MAX_PAGES_TO_PROCESS,
                    "sources": [],
                    "error": "google_main_tab_captcha_no_urls"
                }
                output["projects"] = [p for p in output["projects"] if p["project_name_en_resolved"] != project_name]
                output["projects"].append(project_record)
                save_output()
                polite_sleep(WAIT_SECONDS_BETWEEN_PROJECTS)
                continue

            log(f"  Collected {len(urls)} candidate URLs (deduped)")
            polite_sleep(WAIT_SECONDS_AFTER_SERP)

            google_handle = driver.current_window_handle

            project_record: Dict[str, Any] = {
                "project_name_en_resolved": project_name,
                "developer_name_en": developer_name,
                "google_query": query,
                "candidate_urls": urls,
                "processed_pages_target": MAX_PAGES_TO_PROCESS,
                "sources": [],
            }

            processed_pages = 0

            for u in urls:
                if processed_pages >= MAX_PAGES_TO_PROCESS:
                    break

                u2 = canonicalize_for_dedupe(u)

                if should_skip_url(u2):
                    log(f"  Skipping URL (blacklisted domain): {u2}")
                    continue

                if is_pdf_url(u2):
                    log(f"  Skipping direct PDF: {u2}")
                    continue

                polite_sleep(WAIT_SECONDS_BETWEEN_URLS)
                log(f"  Processing URL ({processed_pages+1}/{MAX_PAGES_TO_PROCESS}): {u2}")

                try:
                    final_url, html = fetch_page_source_in_new_tab(driver, u2, google_handle)

                    if not html.strip():
                        project_record["sources"].append({
                            "source_url": u2,
                            "final_url": final_url,
                            "type": "content_only_nested_v2",
                            "error": "Empty page content"
                        })
                    else:
                        content = extract_content_structured(final_url or u2, html)
                        project_record["sources"].append({
                            "source_url": u2,
                            "final_url": final_url,
                            "type": "content_only_nested_v2",
                            "content": content,
                        })

                        st = content.get("stats", {})
                        log(
                            f"  extracted content "
                            f"(sections={st.get('sections_total')}, blocks={st.get('blocks_total')}, "
                            f"imgs={st.get('images_total')}, pdfs={st.get('pdfs_total')}, root={content.get('content_root_hint')})"
                        )

                    processed_pages += 1

                    log(f"  Saving JSON after URL (sources so far: {len(project_record['sources'])})")
                    output["projects"] = [p for p in output["projects"] if p["project_name_en_resolved"] != project_name]
                    output["projects"].append(project_record)
                    save_output()

                except Exception as e:
                    log(f"  Error on URL {u2}: {e}")
                    project_record["sources"].append({
                        "source_url": u2,
                        "final_url": None,
                        "type": "content_only_nested_v2",
                        "error": str(e),
                    })

                    output["projects"] = [p for p in output["projects"] if p["project_name_en_resolved"] != project_name]
                    output["projects"].append(project_record)
                    save_output()

            log(f"Project completed: {project_name} — {processed_pages} pages processed")
            polite_sleep(WAIT_SECONDS_BETWEEN_PROJECTS)

    finally:
        log("Closing Selenium driver...")
        try:
            driver.quit()
        except Exception:
            pass

    log(f"Script finished. Output: {OUTPUT_JSON_PATH}")
    log(f"Log file: {LOG_FILE_PATH}")


if __name__ == "__main__":
    main()
