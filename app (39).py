#!/usr/bin/env python3
# app.py — Metflic unified backend (Flask + Postgres + optional Pyrogram)
import os
import time
import json
import logging
import threading
import re
from datetime import datetime
from urllib.parse import quote_plus

import requests
from flask import Flask, request, jsonify, render_template, abort

# Optional imports
try:
    from sqlalchemy import create_engine, text
    SQLA = True
except Exception:
    SQLA = False 

try:
    from pyrogram import Client as PyroClient
    PYRO = True
except Exception:
    PYRO = False

# --- logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("metflic")

# --- config (env) ---
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
WEBSITE_URL = os.getenv("WEBSITE_URL", "").rstrip("/") if os.getenv("WEBSITE_URL") else ""
DB_CHANNEL = os.getenv("DB_CHANNEL", "").strip()  # -100... or @channelname
REQUIRED_CHANNELS = [c.strip() for c in os.getenv("REQUIRED_CHANNELS", "").split(",") if c.strip()]
API_ID = os.getenv("API_ID", "")
API_HASH = os.getenv("API_HASH", "")
PYRO_SESSION = os.getenv("PYRO_SESSION", "")
TMDB_KEY = os.getenv("TMDB_KEY", "")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")
PORT = int(os.getenv("PORT", "10000"))
DELETE_AFTER_SECONDS = int(os.getenv("DELETE_AFTER_SECONDS", "300"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))
BACKUP_HOURS = int(os.getenv("BACKUP_HOURS", "12"))
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
INDEX_CHANNEL = os.getenv("INDEX_CHANNEL", "").strip()  # -100...
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
BOT_OPTION_PAGE_SIZE = 8

logger.info("SUPABASE_URL present: %s", bool(SUPABASE_URL))
logger.info("SUPABASE_KEY present: %s", bool(SUPABASE_SERVICE_KEY))

# =========================================================
# 📁 JSON STORAGE (OPTION 3 — FINAL & STABLE)
# =========================================================

import os
import json
import time
import threading

# ================= JSON STORAGE PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MOVIE_DIR = os.path.join(DATA_DIR, "movies")

os.makedirs(MOVIE_DIR, exist_ok=True)

DELETE_QUEUE_FILE = os.path.join(DATA_DIR, "delete_queue.json")

# ================= SHARDS =================
SHARDS = ["ae", "fj", "ko", "pt", "uz", "09"]

# ================= INIT JSON FILES =================
def ensure_json_files():
    for shard in SHARDS:
        path = os.path.join(MOVIE_DIR, f"{shard}.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)

    if not os.path.exists(DELETE_QUEUE_FILE):
        with open(DELETE_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

    logger.info("✅ JSON storage initialized")


# ================= SHARD SELECTOR =================
def get_shard_from_title(title: str) -> str:
    if not title:
        return "09"

    c = title[0].lower()

    if c.isdigit():
        return "09"
    if 'a' <= c <= 'e':
        return "ae"
    if 'f' <= c <= 'j':
        return "fj"
    if 'k' <= c <= 'o':
        return "ko"
    if 'p' <= c <= 't':
        return "pt"
    if 'u' <= c <= 'z':
        return "uz"

    return "09"


# ================= THREAD SAFE JSON INSERT =================
_json_lock = threading.Lock()

def json_insert_movie(movie: dict):
    shard = get_shard_from_title(movie.get("title", ""))
    path = os.path.join(MOVIE_DIR, f"{shard}.json")

    with _json_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []

        # 🚫 Duplicate check (channel_id + message_id)
        for m in data:
            if (
                m.get("message_id") == movie.get("message_id")
                and m.get("channel_id") == movie.get("channel_id")
            ):
                logger.info("⚠️ Duplicate movie skipped")
                return

        data.append(movie)

        # ✅ Atomic write (no corruption)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(tmp, path)

    logger.info(f"🎬 JSON INSERTED → {movie.get('title')}")

# =================================================
# 🔍 STRONG DETECT HELPERS
# =================================================
def detect_quality_strong(text: str):
    t = text.lower()
    if "2160p" in t or "4k" in t or "uhd" in t:
        return "2160p"
    if "1440p" in t or "2k" in t or "qhd" in t:
        return "1440p"
    if "1080p" in t or "full hd" in t or "fhd" in t:
        return "1080p"
    if "720p" in t or " hd " in t:
        return "720p"
    if "480p" in t:
        return "480p"
    if "360p" in t:
        return "360p"
    return None


def detect_language_strong(text: str):
    t = text.lower()
    langs = []

    if any(x in t for x in ["hindi", " hin "]):
        langs.append("Hindi")
    if any(x in t for x in ["english", " eng "]):
        langs.append("English")
    if any(x in t for x in ["tamil", " tam "]):
        langs.append("Tamil")
    if any(x in t for x in ["telugu", " tel "]):
        langs.append("Telugu")
    if any(x in t for x in ["malayalam", " mal "]):
        langs.append("Malayalam")
    if any(x in t for x in ["kannada", " kan "]):
        langs.append("Kannada")

    if not langs:
        return None
    return " / ".join(langs)


def detect_year_strong(text: str):
    m = re.search(r'\b(19|20)\d{2}\b', text)
    return int(m.group()) if m else None


def detect_file_size(text: str):
    m = re.search(r'(\d+(?:\.\d+)?)\s*(GB|MB)', text.upper())
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}"
    

def get_supabase_table_from_title(title: str) -> str:
    if not title:
        return "movies_09"

    c = title[0].lower()

    if c.isdigit():
        return "movies_09"
    if 'a' <= c <= 'e':
        return "movies_ae"
    if 'f' <= c <= 'j':
        return "movies_fj"
    if 'k' <= c <= 'o':
        return "movies_ko"
    if 'p' <= c <= 't':
        return "movies_pt"
    if 'u' <= c <= 'z':
        return "movies_uz"

    return "movies_09"

# =========================================================
# 🧩 SUPABASE CLIENT (JSON ONLY) — FIXED & STABLE
# =========================================================
import requests
import re
import time

def supabase_insert_movie_clean(movie: dict):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return

    # =====================================================
    # 🔹 RAW VALUES (INDEX CHANNEL — TRUSTED SOURCE)
    # =====================================================
    raw_title = movie.get("title", "")
    caption   = movie.get("caption", "")

    # 🔥 IMPORTANT: METADATA DIRECT FROM INDEX CHANNEL
    season   = movie.get("season")
    episode  = movie.get("episode")
    language = movie.get("language")
    quality  = movie.get("quality")
    year     = movie.get("year")

    # =====================================================
    # 🔹 TITLE CLEANING (ONLY FOR SUPABASE)
    # =====================================================
    t = raw_title.lower()

    # 1️⃣ remove bracket content
    t = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', ' ', t)

    # 2️⃣ normalize separators
    t = re.sub(r'[_\.\+\-×]+', ' ', t)

    # 3️⃣ remove season / episode patterns (TITLE ONLY)
    t = re.sub(r'\b(s\d{1,2}\s*e\d{1,3}|s\d{1,2}|e\d{1,3})\b', ' ', t)
    t = re.sub(r'\bseason\s*\d+\b|\bepisode\s*\d+\b', ' ', t)

    # 4️⃣ remove years from title
    t = re.sub(r'\b(19|20)\d{2}\b', ' ', t)

    # 5️⃣ HARD junk words (TITLE ONLY)
    t = re.sub(
        r'\b('
        r'sd|hd|bd|uhd|'
        r'240p|360p|480p|720p|1080p|2160p|4k|8k|10bit|'
        r'bluray|brrip|bdrip|hdrip|hdts|dvdrip|'
        r'webrip|webdl|web|'
        r'x264|x265|h264|h265|hevc|av1|'
        r'aac|dd|ddp|dd2|dts|atmos|'
        r'2ch|2c|5\.1|7\.1|'
        r'eng|english|hin|hindi|tam|tamil|tel|telugu|'
        r'jap|jpn|japanese|kan|kannada|mal|malayalam|'
        r'dual|multi|audio|dub|dubbed|'
        r'esub|esubs|subs|sub|msub|'
        r'mp4|mkv|avi|mov|'
        r'full|uncut|proper|repack|extended|'
        r'hq|hqline|camrip|'
        r'trailer|sample|'
        r'part\s*\d+|cd\s*\d+|'
        r'watch|download|'
        r'official|original|'
        r'psa|rarbg|yts|'
        r'nf|netflix|prime|zee5|jio|'
        r'jc|mkvcinemas|cineprime|rabbitmovies|primeplay'
        r')\b',
        ' ',
        t,
        flags=re.I
    )

    # 6️⃣ final cleanup
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    if not t:
        return

    # 🔑 SHORT & CLEAN TITLE
    clean_title = " ".join(t.split()[:3])
    base_slug   = make_slug(clean_title)

    # =====================================================
    # 🔹 STRONG UNIQUE SLUG (NO DATA LOSS)
    # =====================================================
    slug_parts = [base_slug]

    if season is not None:
        slug_parts.append(f"s{season}")

    if episode is not None:
        slug_parts.append(f"e{episode}")

    if language:
        slug_parts.append(language.lower())

    if quality:
        slug_parts.append(quality.lower())

    if year:
        slug_parts.append(str(year))

    clean_slug = "-".join(slug_parts)

    # =====================================================
    # 🔹 TABLE SHARD
    # =====================================================
    table = get_supabase_table_from_title(clean_title)

    # =====================================================
    # 📦 PAYLOAD (🔥 METADATA NEVER TOUCHED)
    # =====================================================
    payload = {
        "slug": clean_slug,
        "title": clean_title,

        "season": season,
        "episode": episode,
        "language": language,
        "quality": quality,
        "year": year,

        "tmdb_id": extract_tmdb_id(caption),
        "channel_id": movie.get("channel_id"),
        "message_id": movie.get("message_id"),
        "category": movie.get("category"),
        "created_at": int(time.time()),
    }

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=headers,
            json=payload,
            timeout=10
        )

        if r.status_code in (200, 201):
            logger.info(
                "🟢 SUPABASE INSERTED → %s | S:%s E:%s | %s %s",
                clean_slug, season, episode, language, quality
            )
        else:
            logger.error("🔴 SUPABASE ERROR %s → %s", r.status_code, r.text)

    except Exception:
        logger.exception("SUPABASE insert failed")

def supabase_search_movies(query: str, limit: int = 8):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return []

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }

    q = query.strip()
    results = []

    for shard in ["ae", "fj", "ko", "pt", "uz", "09"]:
        try:
            r = requests.get(
                f"{SUPABASE_URL}/rest/v1/movies_{shard}",
                headers=headers,
                params={
                    "select": "slug,title,channel_id,message_id,category,language,quality,season,episode",
                    "title": f"ilike.*{q}*",
                    "limit": limit
                },
                timeout=8
            )

            if r.status_code == 200:
                rows = r.json()
                if rows:
                    results.extend(rows)
                    break   # ✅ first hit is enough

        except Exception:
            logger.exception("Supabase search failed on shard %s", shard)

    return results[:limit]

# ================= INIT JSON FILES =================
ensure_json_files()

# ================= TMDB HELPERS =================

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

def tmdb_get(path, params=None):
    if not TMDB_KEY:
        return {}
    try:
        params = params or {}
        params["api_key"] = TMDB_KEY
        r = requests.get(f"{TMDB_BASE}/{path}", params=params, timeout=8)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def tmdb_safe_title(title: str) -> str:
    if not title:
        return ""

    t = title.lower()

    # 1️⃣ remove anything inside brackets
    t = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', ' ', t)

    # 2️⃣ normalize separators
    t = re.sub(r'[_\.\+\-×]+', ' ', t)

    # 3️⃣ kill season / episode patterns (S01E12, E12, Season 1 etc.)
    t = re.sub(r'\b(s\d{1,2}\s*e\d{1,3}|s\d{1,2}|e\d{1,3})\b', ' ', t)
    t = re.sub(r'\bseason\s*\d+\b|\bepisode\s*\d+\b', ' ', t)

    # 4️⃣ kill dates
    t = re.sub(r'\b\d{1,2}(st|nd|rd|th)?\s+[a-z]+\s+\d{4}\b', ' ', t)

    # 5️⃣ remove ALL years
    t = re.sub(r'\b(19|20)\d{2}\b', ' ', t)

    # 6️⃣ KNOWN junk words (expanded)
    junk_words = (
        r'sd|hd|bd|uhd|'
        r'eng|english|jap|jpn|japanese|'
        r'hin|hindi|tam|tamil|tel|telugu|'
        r'kan|kannada|mal|malayalam|'
        r'dual|multi|audio|dub|dubbed|'
        r'esub|esubs|sub|subs|msub|'
        r'480p|720p|1080p|2160p|4k|8k|10bit|'
        r'bluray|brrip|bdrip|hdrip|hdts|dvdrip|'
        r'webrip|webdl|web|'
        r'x264|x265|h264|h265|hevc|av1|'
        r'aac|dd|ddp|dts|atmos|2ch|5\.1|7\.1|'
        r'mp4|mkv|avi|mov|'
        r'org|nf|netflix|prime|zee5|'
        r'mkvcinemas|cineprime|rabbitmovies|primeplay'
    )

    t = re.sub(rf'\b({junk_words})\b', ' ', t)

    # 7️⃣ remove leftover tech tokens (2–4 char junk words)
    t = re.sub(r'\b[a-z0-9]{1,3}\b', ' ', t)

    # 8️⃣ final cleanup
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    # 9️⃣ fallback safety
    if not t:
        return title.split()[0].lower()

    # 🔟 TMDB prefers very short queries
    return " ".join(t.split()[:3])


def tmdb_best_match(title):
    data = tmdb_get("search/movie", {
        "query": title,
        "include_adult": False,
        "page": 1
    })
    if not data or not data.get("results"):
        return None

    # best match = first result
    return data["results"][0]

def poster_from_title(title):
    m = tmdb_best_match(title)
    if not m:
        return ""
    if not m.get("poster_path"):
        return ""
    return "https://image.tmdb.org/t/p/w500" + m["poster_path"]

def normalize_tmdb(m):
    return {
        # ✅ TITLE: always trust stored title first
        "title": m.get("title") or m.get("name") or "",

        # ✅ SLUG:
        # - if already from supabase → keep it
        # - else (tmdb api) → make tmdb-xxx
        "slug": (
            m.get("slug")
            or (f"tmdb-{m['id']}" if m.get("id") else "")
        ),

        # ✅ POSTER
        "poster": (
            f"https://image.tmdb.org/t/p/w500{m['poster_path']}"
            if m.get("poster_path")
            else m.get("poster")
            or "https://via.placeholder.com/500x750?text=No+Poster"
        ),

        "backdrop": (
            f"https://image.tmdb.org/t/p/original{m['backdrop_path']}"
            if m.get("backdrop_path")
            else m.get("backdrop")
        ),

        "overview": m.get("overview", ""),

        # ✅ YEAR (both cases)
        "year": (
            m.get("year")
            or (m.get("release_date") or m.get("first_air_date") or "")[:4]
        )
    }
     
def normalize_db(row):
    title = row.get("title", "")
    return {
        "slug": row.get("slug"),
        "title": title,
        "overview": row.get("caption", ""),
        "poster": best_poster(title),  # ← यह add करो
        "year": row.get("year", "") or ""
    }

def extract_tmdb_id(text: str):
    """
    Optional TMDB id extractor.
    Currently disabled — returns None safely.
    """
    return None
    
def tmdb_search_movie(title):
    if not TMDB_KEY or not title:
        return []

    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": TMDB_KEY,
                "query": title,
                "page": 1
            },
            timeout=6
        )
        if r.ok:
            return r.json().get("results", [])
    except Exception:
        logger.exception("TMDB search failed")

    return []

def tmdb_movie_full(tid):
    """
    Fetch full TMDB movie data:
    - details
    - images
    - trailer (YouTube)
    """
    base = "https://api.themoviedb.org/3"

    d = requests.get(
        f"{base}/movie/{tid}",
        params={"api_key": TMDB_KEY},
        timeout=8
    ).json()

    imgs = requests.get(
        f"{base}/movie/{tid}/images",
        params={"api_key": TMDB_KEY},
        timeout=8
    ).json()

    vids = requests.get(
        f"{base}/movie/{tid}/videos",
        params={"api_key": TMDB_KEY},
        timeout=8
    ).json()

    # 🎬 trailer (YouTube only)
    trailer = None
    for v in vids.get("results", []):
        if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser"):
            trailer = v.get("key")
            break

    gallery = [
        "https://image.tmdb.org/t/p/original" + i["file_path"]
        for i in imgs.get("backdrops", [])[:10]
    ]

    return {
        "title": d.get("title"),
        "year": (d.get("release_date") or "")[:4],
        "overview": d.get("overview"),
        "poster": (
            "https://image.tmdb.org/t/p/w500" + d["poster_path"]
            if d.get("poster_path") else ""
        ),
        "rating": d.get("vote_average"),
        "runtime": d.get("runtime"),
        "language": (d.get("spoken_languages") or [{}])[0].get("english_name"),
        "images": gallery,
        "trailer": trailer
    }

# ==============================================
if not BOT_TOKEN:
    logger.error("BOT_TOKEN required")
    raise SystemExit("BOT_TOKEN missing")
# 🟢 Skipping single DATABASE_URL check (multi-Neon mode)
# if not DATABASE_URL:
#     logger.error("DATABASE_URL required")
#     raise SystemExit("DATABASE_URL missing")

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
BOT_USERNAME = os.getenv("BOT_USERNAME", BOT_TOKEN.split(":", 1)[0])



# --- simple in-memory cache ---
_cache = {}
def cache_get(k):
    rec = _cache.get(k)
    if not rec: return None
    v, exp = rec
    if time.time() > exp:
        _cache.pop(k, None)
        return None
    return v

def cache_set(k, v, ttl=CACHE_TTL):
    _cache[k] = (v, time.time() + ttl)

# --- poster helpers ---
def poster_from_tmdb_path(p):
    return f"https://image.tmdb.org/t/p/w500{p}" if p else ""

def tmdb_search_poster(title):
    if not TMDB_KEY or not title:
        return "https://via.placeholder.com/500x750?text=No+Poster"

    key = f"tmdb:{title.lower()}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_KEY, "query": title, "page": 1},
            timeout=8
        )
        if r.ok:
            j = r.json()
            if j.get("results"):
                p = j["results"][0].get("poster_path")
                url = poster_from_tmdb_path(p)
                if url:
                    cache_set(key, url)
                    return url
    except Exception:
        logger.debug("tmdb search fail", exc_info=True)

    # ✅ fallback (FUNCTION KE ANDAR)
    fallback = "https://via.placeholder.com/500x750?text=No+Poster"
    cache_set(key, fallback)
    return fallback

def omdb_poster(title):
    if not OMDB_API_KEY or not title: return ""
    key = f"omdb:{title.lower()}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        r = requests.get("http://www.omdbapi.com/", params={"t": title, "apikey": OMDB_API_KEY}, timeout=6)
        if r.ok:
            j = r.json()
            if j.get("Response") == "True" and j.get("Poster") and j.get("Poster") != "N/A":
                cache_set(key, j.get("Poster"))
                return j.get("Poster")
    except Exception:
        logger.debug("omdb fail", exc_info=True)
    cache_set(key, "")
    return ""

def best_poster(title: str) -> str:
    if not title:
        return "/static/placeholder.jpg"

    clean = tmdb_safe_title(title)
    if not clean:
        return "/static/placeholder.jpg"

    m = tmdb_best_match(clean)
    if not m or not m.get("poster_path"):
        return "/static/placeholder.jpg"

    return "https://image.tmdb.org/t/p/w500" + m["poster_path"]
    
def build_category_home(category="movies"):
    now = time.time()

    if category in CATEGORY_CACHE:
        if now - CATEGORY_CACHE[category]["ts"] < CATEGORY_TTL:
            return CATEGORY_CACHE[category]

    # TMDB query per category
    genre_map = {
        "movies": None,
        "anime": 16,
        "cartoon": 16,
        "drama": 18,
        "horror": 27,
        "tv": "tv"
    }

    if genre_map.get(category) == "tv":
        popular = tmdb_get("tv/popular").get("results", [])
        trending = tmdb_get("trending/tv/week").get("results", [])
    else:
        popular = tmdb_get("movie/popular").get("results", [])
        trending = tmdb_get("trending/movie/week").get("results", [])

        gid = genre_map.get(category)
        if gid:
            popular = tmdb_get("discover/movie", {
                "with_genres": gid,
                "sort_by": "popularity.desc"
            }).get("results", [])

    hero = random.choice(popular) if popular else None

    data = {
        "hero": normalize_tmdb(hero) if hero else None,
        "latest": [normalize_tmdb(m) for m in trending[:20]],
        "popular": [normalize_tmdb(m) for m in popular[:20]],
        "random": [normalize_tmdb(m) for m in random.sample(popular, min(20,len(popular)))],
        "ts": now
    }

    CATEGORY_CACHE[category] = data
    return data

# --- Telegram helpers (bot API) ---
def tg_post(method, payload=None, files=None, timeout=20):
    url = f"{API_BASE}/{method}"
    try:
        if files:
            r = requests.post(url, data=payload or {}, files=files, timeout=timeout)
        else:
            r = requests.post(url, json=payload or {}, timeout=timeout)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "http_status": r.status_code, "text": r.text}
    except Exception:
        logger.exception("tg_post exception")
        return {"ok": False, "error": "request error"}

def send_message(chat_id, text, reply_markup=None, parse_mode="HTML"):
    payload = {"chat_id": int(chat_id), "text": text, "parse_mode": parse_mode}
    if reply_markup:
        payload["reply_markup"] = reply_markup if isinstance(reply_markup, str) else json.dumps(reply_markup)
    return tg_post("sendMessage", payload)

def forward_message(chat_id, from_chat_id, message_id):
    return tg_post("forwardMessage", {"chat_id": int(chat_id), "from_chat_id": int(from_chat_id), "message_id": int(message_id)})

def delete_message(chat_id, message_id):
    return tg_post("deleteMessage", {"chat_id": int(chat_id), "message_id": int(message_id)})

# --- delete queue ---
_mem_delete_queue = []
_delete_lock = threading.Lock()

def schedule_delete(chat_id, message_id, delay=DELETE_AFTER_SECONDS):
    if not delay:
        return

    delete_at = int(time.time()) + int(delay)

    for eng in SQLITE_ENGINES_LIST:
        try:
            with eng.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO delete_queue (chat_id, message_id, delete_at)
                        VALUES (:c, :m, :d)
                    """),
                    {"c": chat_id, "m": message_id, "d": delete_at}
                )
        except Exception:
            logger.exception("schedule_delete sqlite failed")


def make_result(title, slug, year="", tg=None):
    tmdb_title = tmdb_safe_title(title)   # ✅ ONLY FOR TMDB

    return {
        "ok": True,
        "result": {
            "title": title,               # 👈 raw / display
            "slug": slug,
            "year": year,
            "poster": tmdb_search_poster(tmdb_title),  # 🔥 FIX
            "overview": "",
            "tg": tg
        }
    }
# --- DB helpers: insert/search/get by slug ---
def make_slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\.(mkv|mp4|avi|mov)$', '', text)
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text[:180] or str(int(time.time()))

def clean_caption(caption: str) -> str:
    if not caption:
        return ""

    # take first line only
    text = caption.splitlines()[0]

    # remove mentions / promo
    text = re.sub(r"@[\w_]+", " ", text)
    text = re.sub(r"(join|follow|subscribe).*", " ", text, flags=re.I)

    # remove brackets content
    text = re.sub(r'[\[\(\{].*?[\]\)\}]', ' ', text)

    # remove common garbage words early
    text = re.sub(
        r'\b('
        r'480p|720p|1080p|2160p|4k|8k|'
        r'hdr|hdr10|dv|'
        r'bluray|brrip|bdrip|webrip|webdl|web-dl|'
        r'cam|ts|tc|'
        r'x264|x265|h\.?264|h\.?265|hevc|'
        r'aac|dd|ddp|dts|atmos|'
        r'eng|english|jpn|japanese|hin|hindi|dual|multi|'
        r'nf|netflix|amzn|prime|'
        r'mkv|mp4|avi'
        r')\b',
        ' ',
        text,
        flags=re.I
    )

    # remove symbols
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text[:200]



def detect_quality(text: str):
    text = text.lower()
    for q in ("2160p", "1080p", "720p", "480p"):
        if q in text:
            return q
    return None


def detect_language(text: str):
    t = text.lower()
    if "hindi" in t:
        return "Hindi"
    if "english" in t:
        return "English"
    if "korean" in t or "k-drama" in t or "viki" in t:
        return "Korean"
    if "japanese" in t or "anime" in t:
        return "Japanese"
    return None


def detect_type(text: str):
    t = text.lower()
    if any(x in t for x in ("s01", "s02", "season", "episode", "ep", "e01", "e02")):
        return "series"
    return "movie"

def extract_season_episode(text: str):
    if not text:
        return None, None

    t = text.lower()

    season = None
    episode = None

    # ===============================
    # 🎯 SEASON PATTERNS
    # ===============================
    season_patterns = [
        r'season[\s\.\-_]*(\d{1,2})',
        r'\bs(\d{1,2})\b',
    ]

    for p in season_patterns:
        m = re.search(p, t)
        if m:
            season = int(m.group(1))
            break

    # ===============================
    # 🎯 EPISODE PATTERNS
    # ===============================
    episode_patterns = [
        r'episode[\s\.\-_]*(\d{1,3})',
        r'\be(\d{1,3})\b',
        r'\be(\d{1,3})\s*[-to]+\s*(\d{1,3})',   # E06-E10
        r'\bep[\s\.\-_]*(\d{1,3})',
    ]

    for p in episode_patterns:
        m = re.search(p, t)
        if m:
            # range case (E06-E10)
            if len(m.groups()) == 2 and m.group(2):
                episode = int(m.group(1))   # START episode
            else:
                episode = int(m.group(1))
            break

    # ===============================
    # 🎯 COMBINED FORMAT (S02.E06-E10)
    # ===============================
    m = re.search(r's(\d{1,2})[\s\.\-_]*e(\d{1,3})(?:\s*[-to]+\s*e?(\d{1,3}))?', t)
    if m:
        season = int(m.group(1))
        episode = int(m.group(2))  # first episode of range

    return season, episode


def detect_category(text: str, content_type: str):
    t = text.lower()

    if "anime" in t:
        return "Anime"
    if "cartoon" in t:
        return "Cartoon"
    if "horror" in t:
        return "Horror"
    if "korean" in t or "k-drama" in t or "viki" in t:
        return "Drama"
    if "bollywood" in t or "hindi" in t:
        return "Bollywood"
    if "hollywood" in t or "english" in t:
        return "Hollywood"

    if content_type == "series":
        return "Web Series"

    return "Movies"   # 🔒 SAFE FALLBACK


def format_movie_button_text(r: dict) -> str:
    parts = []

    # 🎬 Title
    title = r.get("title")
    if title:
        parts.append(f"🎬 {title}")

    # 📺 Season / Episode
    season = r.get("season")
    episode = r.get("episode")
    if season and episode:
        parts.append(f"📺 S{season:02d}E{episode:02d}")
    elif season:
        parts.append(f"📺 Season {season}")

    # 🌐 Language
    if r.get("language"):
        parts.append(f"🌐 {r['language']}")

    # 🎞 Quality
    if r.get("quality"):
        parts.append(f"🎞 {r['quality']}")

    return " | ".join(parts)


def supabase_get_movie_by_slug(slug: str):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }

    for shard in ["ae", "fj", "ko", "pt", "uz", "09"]:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/movies_{shard}",
            headers=headers,
            params={
                "select": "*",
                "slug": f"eq.{slug}",
                "limit": 1
            },
            timeout=8
        )
        if r.status_code == 200:
            rows = r.json()
            if rows:
                return rows[0]

    return None

def title_from_slug(slug: str) -> str:
    if not slug:
        return ""
    return slug.replace("-", " ").strip()
    

SUPABASE_SHARDS = ["ae", "fj", "ko", "pt", "uz", "09"]

def supabase_collect_options(slug: str):
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }

    options = {
        "language": set(),
        "quality": set(),
        "year": set(),
        "season": set(),
        "episode": set(),
    }

    for shard in SUPABASE_SHARDS:
        try:
            r = requests.get(
                f"{SUPABASE_URL}/rest/v1/movies_{shard}",
                headers=headers,
                params={
                    "select": "language,quality,year,season,episode",
                    "slug": f"eq.{slug}",
                    "limit": 500
                },
                timeout=8
            )

            if not r.ok:
                continue

            for row in r.json():
                if row.get("language"):
                    options["language"].add(row["language"])

                if row.get("quality"):
                    options["quality"].add(row["quality"])

                if row.get("year"):
                    options["year"].add(str(row["year"]))

                if row.get("season") is not None:
                    options["season"].add(int(row["season"]))

                if row.get("episode") is not None:
                    options["episode"].add(int(row["episode"]))

        except Exception:
            logger.exception("option fetch failed")

    return {k: sorted(v) for k, v in options.items() if v}

def db_get_movies_by_slug(slug, limit=50):
    if not USE_DB:
        return []

    try:
        return db_select(
            "SELECT * FROM movies WHERE slug=? LIMIT ?",
            (slug, limit)
        )
    except Exception:
        logger.exception("db_get_movies_by_slug failed")
        return []

def db_get_movie_by_id(movie_id):
    try:
        rows = db_select(
            "SELECT * FROM movies WHERE id=? LIMIT 1",
            (int(movie_id),)
        )
        return rows[0] if rows else None
    except Exception:
        logger.exception("db_get_movie_by_id failed")
        return None

def db_get_movie_by_message_id(message_id: int):
    for shard in SHARDS:
        path = os.path.join(MOVIE_DIR, f"{shard}.json")

        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for m in data:
            if int(m.get("message_id", 0)) == int(message_id):
                return m

    return None

        
def extract_bot_options(rows):
    opts = {
        "season": set(),
        "episode": set(),
        "quality": set(),
        "language": set(),
        "year": set(),
    }

    for r in rows:
        if r.get("season") is not None:
            opts["season"].add(r["season"])

        if r.get("episode") is not None:
            opts["episode"].add(r["episode"])

        if r.get("quality"):
            opts["quality"].add(r["quality"])

        if r.get("language"):
            opts["language"].add(r["language"])

        if r.get("year"):
            opts["year"].add(r["year"])

    return {k: sorted(v) for k, v in opts.items() if v}
    
def build_option_keyboard(slug, options, step, page=1):
    values = options.get(step, [])
    if not values:
        return None

    total = len(values)
    page_size = BOT_OPTION_PAGE_SIZE
    total_pages = (total + page_size - 1) // page_size

    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size

    kb = []

    # 🎯 OPTION BUTTONS
    for v in values[start:end]:
        kb.append([{
            "text": f"{step.capitalize()} {v}",
            "callback_data": f"opt:{step}:{v}:{slug}"
        }])

    # 🔁 PAGINATION CONTROLS
    nav = []

    if page > 1:
        nav.append({
            "text": "⬅ Prev",
            "callback_data": f"page:{step}:{page-1}:{slug}"
        })

    nav.append({
        "text": f"📄 {page}/{total_pages}",
        "callback_data": "noop"
    })

    if page < total_pages:
        nav.append({
            "text": "Next ➡",
            "callback_data": f"page:{step}:{page+1}:{slug}"
        })

    if nav:
        kb.append(nav)

    return {"inline_keyboard": kb}


def normalize_title(s):
    if not s:
         return ""
    s = s.lower()
    s = re.sub(r"\(.*?\)", "", s)      # (2008)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def db_find_movie_by_title_guess(clean_title, limit=20):
    rows = db_get_recent_movies(limit=limit)  
    # ye function tum already bana sakte ho:
    # SELECT * FROM movies ORDER BY id DESC LIMIT ?

    target = normalize_title(clean_title)

    best = None
    best_score = 0

    for r in rows:
        caption = r.get("caption") or r.get("title") or ""
        cand = normalize_title(caption)

        score = sum(1 for w in target.split() if w in cand)
        if score > best_score:
            best_score = score
            best = r

    return best if best_score >= 2 else None


# --- optional Pyrogram client for direct fetch/iter (only if configured) ---
pyro = None
_pyro_started = False
def make_pyro():
    global pyro, _pyro_started
    if _pyro_started:
        return pyro
    if not PYRO:
        logger.info("Pyrogram not installed — skipping pyro features")
        return None
    try:
        if PYRO_SESSION:
            pyro = PyroClient("metflic_user", session_string=PYRO_SESSION, api_id=int(API_ID) if API_ID else None, api_hash=API_HASH)
        elif API_ID and API_HASH:
            pyro = PyroClient("metflic_bot", api_id=int(API_ID), api_hash=API_HASH, bot_token=BOT_TOKEN)
        else:
            logger.info("No Pyrogram credentials provided — skip pyro features")
            pyro = None
            _pyro_started = False
            return None
        pyro.start()
        _pyro_started = True
        logger.info("Pyrogram started")
        return pyro
    except Exception:
        logger.exception("pyrogram start failed")
        pyro = None
        _pyro_started = False
        return None

if DB_CHANNEL:
    try:
        make_pyro()
    except Exception:
        pass

def iter_channel_messages(limit=200):
    client = pyro or make_pyro()
    if not client:
        return []
    try:
        msgs = []
        for m in client.iter_history(DB_CHANNEL, limit=limit):
            msgs.append(m)
        return msgs
    except Exception:
        logger.exception("iter_channel_messages failed")
        return []

# --- Flask app ---
app = Flask(__name__, static_folder="static", template_folder="templates")
from flask import render_template, request

@app.route("/")
def home():
    return render_template("movie.html")
    
# @app.route("/category.html")
# def category_page():
#     return render_template("category.html")

# ---------- TV PAGE ----------

@app.route("/tv")
def tv_page():
    return render_template("tv.html")

# ---------- DETAIL PAGE ----------
@app.route("/detail.html")
def detail_page():
    return render_template("detail.html")


# ---------- ANIME ----------
@app.route("/anime.html")
def anime_page():
    return render_template("anime.html")


# ---------- CARTOON ----------
@app.route("/cartoon.html")
def cartoon_page():
    return render_template("cartoon.html")


# ---------- WEB SERIES ----------
@app.route("/webseries.html")
def webseries_page():
    return render_template("webseries.html")


# ---------- MOVIES (OPTIONAL SEPARATE PAGE) ----------
@app.route("/movies.html")
def movies_page():
    return render_template("movies.html")

@app.route("/list.html")
def list_page():
    return render_template("list.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "ok": True,
        "service": "Metflic API",
        "website": WEBSITE_URL or "not-set"
    })

@app.route("/api/tmdb_images/<int:tid>")
def api_tmdb_images(tid):
    if not TMDB_KEY:
        return jsonify({})
    r = requests.get(
        f"https://api.themoviedb.org/3/movie/{tid}/images",
        params={"api_key": TMDB_KEY},
        timeout=8
    )
    return jsonify(r.json() if r.ok else {})

@app.route("/movie/<slug>")
def movie_page(slug):
    # 🟢 Fetch movie from Supabase
    row = supabase_get_movie_by_slug(slug)

    if not row:
        return render_template("404.html"), 404

    # 🔹 Supabase clean title (TMDB ka source)
    clean_title = row.get("title") or ""

    # 🔹 TMDB poster fetch
    poster = None
    try:
        if clean_title and TMDB_KEY:
            poster = best_poster(clean_title)
    except Exception:
        poster = None

    # 📦 metadata for website
    return render_template(
        "movie.html",
        slug=slug,
        title=clean_title,
        poster=poster,
        season=row.get("season"),
        episode=row.get("episode"),
        language=row.get("language"),
        quality=row.get("quality"),
        category=row.get("category")
        )
    
# Webhook endpoint
@app.route("/webhook", methods=["POST"])
def webhook():
    if WEBHOOK_SECRET_TOKEN:
        header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if header != WEBHOOK_SECRET_TOKEN:
            logger.warning("Invalid webhook secret token")
            return jsonify({"ok": False}), 403
    data = request.get_json(force=True)
    threading.Thread(target=handle_update, args=(data,), daemon=True).start()
    return jsonify({"ok": True})

# API endpoints used by website
@app.route("/api/latest")
def api_latest():
    try:
        out = []

        for eng in SQLITE_ENGINES_LIST:
            with eng.connect() as conn:
                rows = conn.execute(
                    text("""
                        SELECT slug, title, year, channel_id, message_id
                        FROM movies
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                ).mappings().all()

                for r in rows:
                    out.append({
                        "slug": r["slug"],
                        "title": r["title"],
                        "year": r["year"],
                        "poster": best_poster(r["title"]),
                        "channel_id": r["channel_id"],
                        "message_id": r["message_id"]
                    })

        return jsonify(out[:20])

    except Exception:
        logger.exception("api_latest failed")
        return jsonify({"error": "internal"}), 500

@app.route("/api/a2z")
def api_a2z():
    try:
        page = int(request.args.get("page", "1"))
        per  = int(request.args.get("per", "48"))

        items = []

        if USE_DB:
            for eng in engines:
                with eng.connect() as conn:
                    rows = conn.execute(
                        text("""
                            SELECT slug, title, year
                            FROM movies
                            ORDER BY LOWER(title)
                            LIMIT :per OFFSET :off
                        """),
                        {"per": per, "off": (page - 1) * per}
                    ).mappings().all()

                    for r in rows:
                        items.append({
                            "slug": r["slug"],
                            "title": r["title"],
                            "year": r.get("year", ""),
                            "poster": best_poster(r["title"])
                        })

        return jsonify({
            "ok": True,
            "page": page,
            "per": per,
            "items": items[:per]
        })

    except Exception:
        logger.exception("api_a2z failed")
        return jsonify({"ok": False}), 500

@app.route("/api/movie/<slug>")
def api_movie(slug):

    # ================= DB FIRST =================
    try:
        if USE_DB:
            row = db_get_movie_by_slug(slug)
            if row:
                return jsonify({
                    "ok": True,
                    "result": {
                        "slug": slug,
                        "title": row["title"],
                        "year": row.get("year", ""),
                        "poster": best_poster(row["title"]),
                        "overview": "",
                        "images": [],
                        "trailer": None,
                        "rating": None,
                        "runtime": None,
                        "language": None,
                        "download": f"https://t.me/{BOT_USERNAME}?start={quote_plus(slug)}",
                        "from_db": True,
                        "channel_id": row["channel_id"],
                        "message_id": row["message_id"]
                    }
                })
    except Exception:
        logger.exception("DB fetch failed")


@app.route("/api/movie_options")
def api_movie_options():
    slug = (request.args.get("slug") or "").strip()
    if not slug:
        return jsonify({"ok": False, "error": "slug missing"})

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }

    options = {
        "language": set(),
        "quality": set(),
        "year": set(),
        "season": set(),
        "episode": set(),
    }

    # 🔥 IMPORTANT: options ONLY from same slug
    for shard in SUPABASE_SHARDS:
        params = {
            "select": "language,quality,year,season,episode",
            "slug": f"eq.{slug}",
            "limit": 200,
        }

        try:
            r = requests.get(
                f"{SUPABASE_URL}/rest/v1/movies_{shard}",
                headers=headers,
                params=params,
                timeout=8
            )
            if not r.ok:
                continue

            for row in r.json():
                if row.get("language"):
                    options["language"].add(row["language"])

                if row.get("quality"):
                    options["quality"].add(row["quality"])

                if row.get("year"):
                    options["year"].add(str(row["year"]))

                if row.get("season") is not None:
                    options["season"].add(int(row["season"]))

                if row.get("episode") is not None:
                    options["episode"].add(int(row["episode"]))

        except Exception:
            logger.exception("movie_options fetch failed")

    # 🔥 clean + sort output
    clean_options = {}
    for k, v in options.items():
        if v:
            clean_options[k] = sorted(v)

    return jsonify({
        "ok": True,
        "slug": slug,
        "options": clean_options
    })

@app.route("/api/movie_versions")
def api_movie_versions():
    slug = (request.args.get("slug") or "").strip()
    if not slug:
        return jsonify({"ok": False, "error": "slug missing"})

    rows = []
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }

    for shard in SUPABASE_SHARDS:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/movies_{shard}",
            headers=headers,
            params={
                "select": "id,message_id,title,season,episode,language,quality,year",
                "slug": f"eq.{slug}",
                "order": "season.asc,episode.asc",
                "limit": 500
            },
            timeout=8
        )
        if r.ok:
            rows.extend(r.json())

    if not rows:
        return jsonify({"ok": False, "error": "no versions found"})

    return jsonify({
        "ok": True,
        "versions": rows
    })


@app.route("/api/movie_resolve")
def api_movie_resolve():
    slug = (request.args.get("slug") or "").strip()
    if not slug:
        return jsonify({"ok": False, "error": "slug missing"})

    filters = {
        "language": request.args.get("language"),
        "quality": request.args.get("quality"),
        "year": request.args.get("year"),
        "season": request.args.get("season"),
        "episode": request.args.get("episode"),
    }

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }

    for shard in SUPABASE_SHARDS:
        params = {
            "select": "channel_id,message_id",
            "slug": f"eq.{slug}",
            "limit": 1,
        }

        for k, v in filters.items():
            if v:
                params[k] = f"eq.{v}"

        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/movies_{shard}",
            headers=headers,
            params=params,
            timeout=8
        )

        if r.ok and r.json():
            return jsonify({"ok": True, "result": r.json()[0]})

    return jsonify({"ok": False, "error": "No matching version found"})


    # ================= TELEGRAM MSG =================
    if slug.startswith("msg-"):
        try:
            mid = int(slug.split("-", 1)[1])
            client = pyro or make_pyro()
            if client:
                m = client.get_messages(DB_CHANNEL, mid)
                if m:
                    title_raw = (getattr(m, "caption", None) or getattr(m, "text", ""))[:500]
                    title = clean_caption(title_raw)
                    return jsonify({
                        "ok": True,
                        "result": {
                            "slug": slug,
                            "title": title,
                            "year": "",
                            "poster": best_poster(title),
                            "overview": "",
                            "images": [],
                            "trailer": None,
                            "rating": None,
                            "runtime": None,
                            "language": None,
                            "download": f"https://t.me/{BOT_USERNAME}?start={quote_plus(slug)}",
                            "channel_id": m.chat.id,
                            "message_id": m.message_id
                        }
                    })
        except Exception:
            logger.exception("Telegram fetch failed")

    # ================= TMDB FULL DETAIL + GALLERY + TRAILER =================
    if slug.startswith("tmdb-") and TMDB_KEY:
        try:
            tid = int(slug.split("-", 1)[1])

            # main detail
            d = requests.get(
                f"https://api.themoviedb.org/3/movie/{tid}",
                params={"api_key": TMDB_KEY},
                timeout=8
            ).json()

            # gallery
            imgs = requests.get(
                f"https://api.themoviedb.org/3/movie/{tid}/images",
                params={"api_key": TMDB_KEY},
                timeout=8
            ).json()

            gallery = [
                "https://image.tmdb.org/t/p/original" + i["file_path"]
                for i in imgs.get("backdrops", [])[:10]
            ]

            # trailer (YouTube key only – no channel info)
            vids = requests.get(
                f"https://api.themoviedb.org/3/movie/{tid}/videos",
                params={"api_key": TMDB_KEY},
                timeout=8
            ).json()

            trailer = None
            for v in vids.get("results", []):
                if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser"):
                    trailer = v.get("key")
                    break

            return jsonify({
                "ok": True,
                "result": {
                    "slug": slug,
                    "title": d.get("title"),
                    "year": (d.get("release_date") or "")[:4],
                    "poster": (
                        "https://image.tmdb.org/t/p/w500" + d["poster_path"]
                        if d.get("poster_path") else "/static/placeholder.jpg"
                    ),
                    "overview": d.get("overview"),
                    "images": gallery,          # 🔥 REAL MOVIE SCENES
                    "trailer": trailer,         # ▶ YouTube key only
                    "rating": d.get("vote_average"),
                    "runtime": d.get("runtime"),
                    "language": (
                        (d.get("spoken_languages") or [{}])[0].get("english_name")
                    ),
                    "download": f"https://t.me/{BOT_USERNAME}?start={quote_plus(slug)}"
                }
            })
        except Exception:
            logger.exception("TMDB detail+gallery+trailer failed")

    # ================= FALLBACK SEARCH =================
    try:
        guess = slug.replace("-", " ")

        if TMDB_KEY:
            r = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": TMDB_KEY, "query": guess},
                timeout=8
            )
            if r.ok:
                j = r.json()
                if j.get("results"):
                    m = j["results"][0]
                    return jsonify({
                        "ok": True,
                        "result": {
                            "slug": slug,
                            "title": m.get("title"),
                            "year": (m.get("release_date") or "")[:4],
                            "poster": (
                                "https://image.tmdb.org/t/p/w500" + m["poster_path"]
                                if m.get("poster_path") else ""
                            ),
                            "overview": m.get("overview"),
                            "images": [],
                            "trailer": None,
                            "rating": None,
                            "runtime": None,
                            "language": None,
                            "download": f"https://t.me/{BOT_USERNAME}?start={quote_plus(slug)}"
                        }
                    })

        if OMDB_API_KEY:
            r = requests.get(
                "http://www.omdbapi.com/",
                params={"t": guess, "apikey": OMDB_API_KEY},
                timeout=6
            )
            if r.ok:
                j = r.json()
                if j.get("Response") == "True":
                    return jsonify({
                        "ok": True,
                        "result": {
                            "slug": slug,
                            "title": j.get("Title"),
                            "year": j.get("Year"),
                            "poster": j.get("Poster"),
                            "overview": j.get("Plot"),
                            "images": [],
                            "trailer": None,
                            "rating": j.get("imdbRating"),
                            "runtime": j.get("Runtim e"),
                            "language": j.get("Language"),
                            "download": f"https://t.me/{BOT_USERNAME}?start={quote_plus(slug)}"
                        }
                    })
    except Exception:
        logger.exception("Fallback search failed")

    return jsonify({"ok": False, "error": "not found"}), 404

@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"ok": True, "result": []})

    results = []

    try:
        # ======================
        # 1️⃣ DATABASE SEARCH
        # ======================
        if USE_DB:
            rows = db_search_smart(q, limit=40)

            for r in rows:
                slug = r.get("slug")
                if not slug:
                    continue

                raw_title = r.get("title") or ""
                clean = clean_title(raw_title)

                overview = r.get("overview") or ""
                year = r.get("year") or ""

                poster = best_poster(clean)

                results.append({
                    "slug": slug,                     # UI / TMDB
                    "title": clean,                   # CLEAN
                    "overview": overview,
                    "year": year,
                    "poster": poster,
                    "tg": f"msg-{r['message_id']}" if r.get("message_id") else None
                })

            return jsonify({"ok": True, "result": results})

    except Exception:
        logger.exception("api_search failed")
        return jsonify({"ok": False, "error": "search failed"}), 500
        
# ------------------ TMDB API ROUTES ------------------
TMDB_BASE = "https://api.themoviedb.org/3"
def tmdb_get(path, params=None):
    if not TMDB_KEY:
        return {"error": "TMDB_KEY not set"}
    try:
        params = params or {}
        params["api_key"] = TMDB_KEY
        r = requests.get(f"{TMDB_BASE}/{path}", params=params, timeout=8)
        return r.json()
    except Exception:
        return {"error": "TMDB request failed"}

@app.route("/api/tmdb/search")
def api_tmdb_search():
    q = request.args.get("query","").strip()
    if not q:
        return jsonify({"ok": False})

    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_KEY, "query": q, "page": 1},
            timeout=6
        )
        if r.ok:
            j = r.json()
            if j.get("results"):
                p = j["results"][0].get("poster_path")
                if p:
                    return jsonify({
                        "ok": True,
                        "result": {
                            "poster": "https://image.tmdb.org/t/p/w500" + p
                        }
                    })
    except:
        pass

    return jsonify({"ok": False})

@app.route("/api/trending")
def api_trending():
    data = tmdb_get("trending/movie/week")
    movies = [normalize_tmdb(m) for m in data.get("results", [])]
    return jsonify({"ok": True, "result": movies})

@app.route("/api/popular")
def api_popular():
    data = tmdb_get("movie/popular", {"page": 1})
    out = []

    for m in data.get("results", []):
        out.append({
            "slug": f"tmdb-{m['id']}",
            "title": m.get("title"),
            "year": (m.get("release_date") or "")[:4],
            "poster": (
                "https://image.tmdb.org/t/p/w500" + m["poster_path"]
                if m.get("poster_path") else "/templates/placeholder.jpg"
            ),
            "overview": m.get("overview", "")
        })

    return jsonify({"ok": True, "result": out})
    
@app.route("/api/featured")
def api_featured():
    data = tmdb_get("movie/top_rated", {"page": 1})
    movies = [normalize_tmdb(m) for m in data.get("results", [])]
    return jsonify({"ok": True, "result": movies})

@app.route("/api/nowplaying")
def api_nowplaying():
    data = tmdb_get("movie/now_playing", {"page": 1})
    movies = [normalize_tmdb(m) for m in data.get("results", [])]
    return jsonify({"ok": True, "result": movies})

@app.route("/api/category/<ctype>")
def api_category(ctype):

    sections = {
        "trending": [],
        "popular": [],
        "latest": []
    }

    genre_map = {
        "movies": None,
        "tv": None,
        "anime": 16,
        "cartoon": 16,
        "drama": 18,
        "horror": 27,
        "web": None
    }

    media_type = "movie"
    if ctype == "tv":
        media_type = "tv"

    gid = genre_map.get(ctype)

    # ---------- TRENDING (CATEGORY AWARE) ----------
    if media_type == "tv":
        t = tmdb_get("trending/tv/week")
    else:
        t = tmdb_get("trending/movie/week")

    if gid:
        t["results"] = [
            m for m in t.get("results", [])
            if gid in m.get("genre_ids", [])
        ]

    sections["trending"] = [
        normalize_tmdb(m) for m in t.get("results", [])[:20]
    ]

    # ---------- POPULAR (CATEGORY AWARE) ----------
    if media_type == "tv":
        p = tmdb_get("tv/popular")
    else:
        p = tmdb_get("movie/popular")

    if gid:
        p["results"] = [
            m for m in p.get("results", [])
            if gid in m.get("genre_ids", [])
        ]

    sections["popular"] = [
        normalize_tmdb(m) for m in p.get("results", [])[:20]
    ]

    # ---------- LATEST ----------
    params = {"sort_by": "popularity.desc"}
    if gid:
        params["with_genres"] = gid

    if media_type == "tv":
        l = tmdb_get("discover/tv", params)
    else:
        l = tmdb_get("discover/movie", params)

    sections["latest"] = [
        normalize_tmdb(m) for m in l.get("results", [])[:20]
    ]

    return jsonify({"ok": True, "sections": sections})

# ================== HOMEPAGE ENGINE (FINAL) ==================
import time, random

HOME_CACHE = {}
HOME_TTL = 60 * 60 * 5   # 🔥 5 hours

def build_homepage(category="movies"):
    now = time.time()
    key = f"home_{category}"

    # ✅ Cache hit
    if key in HOME_CACHE and now - HOME_CACHE[key]["ts"] < HOME_TTL:
        return HOME_CACHE[key]["data"]

    # ================= TMDB STRATEGY =================
    if category == "anime":
        popular = tmdb_get("discover/movie", {
            "with_genres": 16,
            "sort_by": "popularity.desc"
        }).get("results", [])

        trending = tmdb_get("trending/movie/week").get("results", [])

    elif category == "drama":
        popular = tmdb_get("discover/movie", {
            "with_genres": 18,
            "sort_by": "popularity.desc"
        }).get("results", [])

        trending = tmdb_get("trending/movie/week").get("results", [])

    elif category == "horror":
        popular = tmdb_get("discover/movie", {
            "with_genres": 27,
            "sort_by": "popularity.desc"
        }).get("results", [])

        trending = tmdb_get("trending/movie/week").get("results", [])

    elif category == "tv":
        popular = tmdb_get("tv/popular").get("results", [])
        trending = tmdb_get("trending/tv/week").get("results", [])

    else:  # 🎬 MOVIES (DEFAULT)
        popular = tmdb_get("movie/popular", {"page": 1}).get("results", [])
        trending = tmdb_get("trending/movie/week").get("results", [])

    # ================= SAFE FALLBACK =================
    if not popular:
        data = {
            "hero": None,
            "latest": [],
            "popular": [],
            "random": []
        }
        HOME_CACHE[key] = {"ts": now, "data": data}
        return data

    hero_raw = random.choice(popular)

    data = {
        "hero": normalize_tmdb(hero_raw),
        "latest": [normalize_tmdb(m) for m in trending[:20]],
        "popular": [normalize_tmdb(m) for m in popular[:20]],
        "random": [
            normalize_tmdb(m)
            for m in random.sample(popular, min(20, len(popular)))
        ]
    }

    HOME_CACHE[key] = {"ts": now, "data": data}
    return data



@app.route("/api/tv/search")
def api_tv_search():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"ok": True, "result": []})

    data = tmdb_get("search/tv", {"query": q, "page": 1}).get("results", [])

    return jsonify({
        "ok": True,
        "result": [normalize_tmdb(m) for m in data[:30]]
    })

@app.route("/tv/<slug>")
def tv_detail_page(slug):
    return render_template("tv_detail.html", slug=slug)
    
@app.route("/api/home")
def api_home():
    q = (request.args.get("q") or "").strip()
    slug = (request.args.get("slug") or "").strip()
    category = request.args.get("cat", "movies")

    # 🔹 Prepare homepage sections ONCE
    home = build_homepage(category)

    # 🟢 1️⃣ SLUG HERO (Telegram / Direct movie)
    if slug:
        row = db_get_movie_by_slug(slug)
        if row:
            clean = clean_title(row.get("title", ""))
            poster = best_poster(clean)

            return jsonify({
                "ok": True,
                "hero": {
                    "slug": row.get("slug"),
                    "title": clean,
                    "overview": "",
                    "year": row.get("year", ""),
                    "poster": poster
                },
                "latest": home.get("latest", []),
                "popular": home.get("popular", []),
                "random": home.get("random", [])
            })

    # 🔍 2️⃣ SEARCH HERO (?q=)
    if q:
        rows = db_search_smart(q, limit=1)
        if rows:
            r = rows[0]
            clean = clean_title(r.get("title", ""))
            poster = best_poster(clean)

            return jsonify({
                "ok": True,
                "hero": {
                    "slug": r.get("slug"),
                    "title": clean,
                    "overview": "",
                    "year": r.get("year", ""),
                    "poster": poster,
                    "tg": f"msg-{r['message_id']}"   # 🔥 ADD
                },
                "latest": home.get("latest", []),
                "popular": home.get("popular", []),
                "random": home.get("random", [])
            })

    # 🏠 3️⃣ NORMAL HOME
    return jsonify({
        "ok": True,
        "hero": home.get("hero"),
        "latest": home.get("latest", []),
        "popular": home.get("popular", []),
        "random": home.get("random", [])
    })
    
# ---------- MOVIE PAGE (SEARCH + SLUG HERO SUPPORT) ----------

def clean_title(text: str) -> str:
    if not text:
        return ""

    t = text.lower()

    # remove brackets content
    t = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', ' ', t)

    # normalize separators
    t = re.sub(r'[_\.\+\-×]+', ' ', t)

    # remove season / episode
    t = re.sub(r'\bs\d{1,2}\s*e\d{1,3}\b', ' ', t)
    t = re.sub(r'\bs\d{1,2}\b', ' ', t)
    t = re.sub(r'\be\d{1,3}\b', ' ', t)

    # remove dates like 17th December 2022
    t = re.sub(
        r'\b\d{1,2}(st|nd|rd|th)?\s+[a-z]+\s+\d{4}\b',
        ' ',
        t
    )

    # extract year (keep only one)
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', t)
    t = re.sub(r'\b(19\d{2}|20\d{2})\b', ' ', t)

    # remove known junk words ONLY
    junk_words = {
        "480p","720p","1080p","2160p","4k","8k",
        "hdr","hdrip","hdts","dvdrip",
        "bluray","brrip","bdrip","webrip","webdl","web",
        "x264","x265","h264","h265","hevc","av1",
        "aac","dd","ddp","dts","atmos",
        "mp4","mkv","avi","mov",
        "hindi","english","eng","jpn","japanese","telugu","tamil",
        "malayalam","bengali","korean","gujarati","punjabi","marathi",
        "dual","multi","audio","dub","dubbed",
        "esub","esubs","subs","msub",
        "full","complete","uncut","hq","hd",
        "hot","series","webseries","show",
        "nf","netflix","amzn","prime","zee5",
        "cineprime","rabbitmovies","kooku","primeplay",
        "org","mkvcinemas","m2links"
    }

    words = []
    for w in t.split():
        if w not in junk_words and not w.isdigit():
            words.append(w)

    t = " ".join(words)

    # restore ONE year at end
    if years:
        t = f"{t} {years[0]}"

    # final cleanup
    t = re.sub(r'\s+', ' ', t).strip()

    return t.title()

@app.route("/api/movie_hero")
def api_movie_hero():
    slug = (request.args.get("slug") or "").strip()
    if not slug:
        return jsonify({"ok": False, "error": "slug missing"})

    # =================================================
    # 🔥 SOURCE OF TRUTH = SUPABASE TITLE
    # =================================================

    # -----------------------------
    # CASE 1️⃣ : tmdb-12345 slug
    # -----------------------------
    if slug.startswith("tmdb-"):
        try:
            tid = int(slug.split("-", 1)[1])
            m = tmdb_movie_full(tid)
            if m:
                return jsonify({
                    "ok": True,
                    "result": {
                        "title": m.get("title", ""),
                        "slug": slug,
                        "year": m.get("year", ""),
                        "poster": m.get("poster", ""),
                        "overview": m.get("overview", ""),
                        "tg": None
                    }
                })
        except Exception:
            logger.exception("tmdb id based hero fetch failed")

    # -----------------------------
    # CASE 2️⃣ : slug → SUPABASE → TITLE
    # -----------------------------
    rows = db_get_movies_by_slug(slug)
    if not rows:
        return jsonify({"ok": False}), 404

    r = rows[0]

    # ✅ ONLY SUPABASE TITLE GOES TO TMDB
    title_for_tmdb = r.get("title", "").strip()
    year = r.get("year")

    poster = best_poster(title_for_tmdb, year)

    return jsonify({
        "ok": True,
        "result": {
            "title": title_for_tmdb,
            "slug": slug,
            "year": year or "",
            "poster": poster,
            "overview": "",
            "tg": r.get("message_id")
        }
    })
    
@app.route("/api/tv/<slug>")
def api_tv_detail(slug):
    if not slug.startswith("tmdb-"):
        return jsonify({"ok": False}), 404

    try:
        tid = int(slug.split("-",1)[1])

        show = tmdb_get(f"tv/{tid}")
        seasons_raw = tmdb_get(f"tv/{tid}").get("seasons", [])

        seasons = []
        for s in seasons_raw:
            sn = s.get("season_number")
            if sn is None or sn == 0:
                continue

            eps = tmdb_get(f"tv/{tid}/season/{sn}").get("episodes", [])
            seasons.append({
                "season_number": sn,
                "episodes": [
                    {
                        "episode_number": e.get("episode_number"),
                        "title": e.get("name")
                     } for e in eps
                ]
            })

        return jsonify({
            "ok": True,
            "show": normalize_tmdb(show),
            "seasons": seasons
        })

    except Exception:
        logger.exception("TV DETAIL ERROR")
        return jsonify({"ok": False}), 500
    
# =========================================================
# 🎭 DRAMA MODULE — FINAL & CLEAN
# =========================================================

@app.route("/drama")
def drama_page():
    return render_template("drama.html")

# ---------- DRAMA HOME ----------
@app.route("/api/drama/home")
def api_drama_home():
    try:
        rows = get_dramas(limit=20)
        items = [normalize_tmdb(m) for m in rows]

        return jsonify({
            "ok": True,
            "hero": items[0] if items else None,
            "list": items[1:20]
        })
    except Exception:
        logger.exception("DRAMA HOME ERROR")
        return jsonify({"ok": False}), 500


# ---------- DRAMA SEARCH ----------
@app.route("/api/drama/search")
def api_drama_search():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"ok": True, "result": []})

    try:
        rows = db_search_drama(q, limit=30)
        return jsonify({
            "ok": True,
            "result": [normalize_tmdb(m) for m in rows]
        })
    except Exception:
        logger.exception("DRAMA SEARCH ERROR")
        return jsonify({"ok": False}), 500


# ---------- DRAMA DETAIL ----------
@app.route("/drama/<slug>")
def drama_detail_page(slug):
    return render_template("drama_detail.html", slug=slug)


@app.route("/api/drama/<slug>")
def api_drama_detail(slug):
    try:
        show = get_drama(slug)
        if not show:
            return jsonify({"ok": False}), 404

        return jsonify({
            "ok": True,
            "movie": normalize_tmdb(show)
        })
    except Exception:
        logger.exception("DRAMA DETAIL ERROR")
        return jsonify({"ok": False}), 500

# =========================================================
# 🔔 TELEGRAM UPDATE HANDLER — FINAL SAFE FIX
# =========================================================
def handle_update(update):
    try:
        # =================================================
        # 🔁 CALLBACK QUERY
        # =================================================
        if "callback_query" in update:
            cq = update["callback_query"]
            uid = cq["from"]["id"]
            data = cq.get("data", "")

            # 🔓 JOIN CONFIRM
            if data.startswith("joined:"):
                slug = data.split(":", 1)[1]
                threading.Thread(
                    target=handle_start_with_slug,
                    args=(uid, uid, slug),
                    daemon=True
                ).start()
                return

            # =================================================
            # 📄 PAGINATION HANDLER
            # =================================================
            if data.startswith("page:"):
                _, step, page, slug = data.split(":", 3)
                page = int(page)

                rows = db_get_movies_by_slug(slug)
                options = extract_bot_options(rows)

                kb = build_option_keyboard(slug, options, step, page)
                if kb:
                    send_message(uid, f"Select {step}", reply_markup=kb)
                return

            # =================================================
            # 🎯 OPTION BUTTON HANDLER
            # =================================================
            if data.startswith("opt:"):
                _, key, value, slug = data.split(":", 3)

                rows = db_get_movies_by_slug(slug)

                # FILTER SELECTED OPTION
                rows = [r for r in rows if str(r.get(key)) == value]

                options = extract_bot_options(rows)

                # NEXT OPTION STEP (ORDER IMPORTANT)
                for step in ("season", "episode", "quality", "language", "year"):
                    if step in options and step != key:
                        kb = build_option_keyboard(slug, options, step, page=1)
                        send_message(uid, f"Select {step}", reply_markup=kb)
                        return

                # 🎬 FINAL FILE SEND
                if rows:
                    r = rows[0]
                    forward_message(uid, r["channel_id"], r["message_id"])
                else:
                    send_message(uid, "❌ Selected option available nahi hai.")
                return
        
        # =================================================
        # 📩 MESSAGE / CHANNEL_POST
        # =================================================
        m = (
            update.get("channel_post")
            or update.get("message")
            or update.get("edited_channel_post")
        )

        if not m:
            return

        chat = m.get("chat", {})
        cid = chat.get("id")
        text = m.get("text", "") or ""
        uid = (m.get("from") or {}).get("id") or cid

        chat_type = chat.get("type")

        # =================================================
        # ▶️ /start <message_id | slug> (PRIVATE ONLY)
        # =================================================
        if chat_type == "private" and isinstance(text, str) and text.startswith("/start"):
            try:
                parts = text.split(" ", 1)
                param = parts[1].strip() if len(parts) > 1 else None

                if not param:
                    send_message(
                        cid,
                        "⚠️ Please click the Download button again from the website."
                    )
                    return

                # ==========================================
                # 🔥 CASE 1: START BY MESSAGE ID (FINAL)
                # ==========================================
                if param.isdigit():
                    row = db_get_movie_by_message_id(int(param))
                    if row:
                        forward_message(uid, row["channel_id"], row["message_id"])
                    else:
                        send_message(uid, "❌ Movie not found.")
                    return

                # ==========================================
                # 🟡 CASE 2: SLUG FLOW (OPTION SELECT)
                # ==========================================
                USER_LAST_SLUG[uid] = param

                threading.Thread(
                    target=handle_start_with_slug,
                    args=(uid, cid, param),
                    daemon=True
                ).start()
                return

            except Exception:
                logger.exception("/start handler failed")
                send_message(uid, "❌ Something went wrong.")
                return
                
        # =================================================
        # 🔍 PRIVATE QUICK SEARCH
        # =================================================
        if chat_type == "private" and text and not text.startswith("/"):
            q = text.strip()[:200]
            rows = supabase_search_movies(q, limit=8)

            if not rows:
                send_message(cid, "❌ Movie not found.")
                if WEBSITE_URL:
                    send_message(cid, WEBSITE_URL)
            else:
                kb = {
                    "inline_keyboard": [
                        [{
                            "text": format_movie_button_text(r),
                            "url": f"{WEBSITE_URL}/movie/{r['slug']}"
                        }]
                        for r in rows
                    ]
                }
                send_message(cid, f"Results for: <b>{q}</b>", reply_markup=kb)
            return

        # =================================================
        # 👥 GROUP QUICK SEARCH
        # =================================================
        if chat_type in ("group", "supergroup") and text and not text.startswith("/"):
            q = text.strip()[:200]
            if len(q) < 120 and re.search(r"[A-Za-z0-9]", q):
                rows = db_search_smart(q, limit=4) if USE_DB else []
                if rows:
                    kb = {
                        "inline_keyboard": [
                            [{
                                "text": format_movie_button_text(r),
                                "url": f"{WEBSITE_URL}/movie/{r['slug']}"
                            }]
                            for r in rows
                        ]
                    }
                    send_message(cid, f"Found results for: <b>{q}</b>", reply_markup=kb)
            return

        # =================================================
        # 📥 DB CHANNEL → INSERT MOVIE (CHANNEL ONLY)
        # =================================================
        if chat_type != "channel" or not DB_CHANNEL:
            return

        try:
            if int(cid) != int(DB_CHANNEL):
                return

            logger.info(f"✅ DB_CHANNEL MATCHED → {cid}")

            media = None
            for k in ("document", "video", "animation", "audio", "voice", "photo"):
                if m.get(k):
                    media = m.get(k)
                    break

            if not media:
                return

            caption = (m.get("caption") or "")[:1000]

            title_guess = (
                clean_caption(caption)
                or media.get("file_name")
                or media.get("file_id")
            )

            # ✅ STRONG & FINAL DETECTION (INDEX CHANNEL LEVEL)
            season, episode = extract_season_episode(caption)

            quality = (
                detect_quality(caption)
                or detect_quality_strong(caption)
                or "Unknown"
            )

            language = (
                detect_language(caption)
                or detect_language_strong(caption)
                or "Unknown"
            )

            year = detect_year_strong(caption)
            file_size = detect_file_size(caption)

            content_type = detect_type(caption) or "movie"
            category = detect_category(caption, content_type) or "unknown"

            tmdb_id = extract_tmdb_id(caption)
            slug = f"tmdb-{tmdb_id}" if tmdb_id else make_slug(title_guess)

            movie = {
                "slug": slug,
                "title": title_guess,
                "season": season,
                "episode": episode,
                "channel_id": cid,
                "message_id": m.get("message_id"),
                "file_id": media.get("file_id"),
                "file_unique_id": media.get("file_unique_id"),
                "caption": caption,
                "category": category,
                "language": language,
                "year": year,
                "file_size": file_size,
                "quality": quality,
                "created_at": int(time.time())
            }

            json_insert_movie(movie)

        except Exception:
            logger.exception("DB CHANNEL INSERT FAILED")
            return


        # ===============================
        # 🟢 SUPABASE CLEAN INSERT
        # ===============================
        if SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                supabase_insert_movie_clean(movie)
                logger.info("🟢 SUPABASE INSERTED → %s", title_guess)
            except Exception:
                logger.exception("Supabase insert failed")

        # ===============================
        # 📤 SEND JSON TO INDEX_CHANNEL
        # ===============================
        if INDEX_CHANNEL:
            try:
                payload = json.dumps(movie, ensure_ascii=False, indent=2)

                # Telegram safety limit
                if len(payload) > 3500:
                    payload = payload[:3500] + "\n…TRUNCATED"

                send_message(
                    INDEX_CHANNEL,
                    f"<pre>{payload}</pre>"
                )
                logger.info("📦 JSON sent to INDEX_CHANNEL")

            except Exception:
                logger.exception("INDEX_CHANNEL send failed")

        logger.info(f"🎬 MOVIE INSERTED → {title_guess} | {slug}")

    except Exception:
        logger.exception("handle_update error")

# =========================================================
# 🎬 DOWNLOAD HANDLER WITH SILENT JOIN CHECK
# =========================================================
def handle_start_with_slug(uid, chat_id, slug):

    # 🔒 JOIN CHECK (same behaviour)
    if REQUIRED_CHANNELS and not check_user_join(uid):
        kb = {
            "inline_keyboard": [
                [{"text": "📢 Join Channel", "url": f"https://t.me/{REQUIRED_CHANNELS[0].lstrip('@')}"}],
                [{"text": "✅ I've Joined", "callback_data": f"joined:{slug}"}]
            ]
        }
        send_message(
            uid,
            "🔒 Movie download ke liye channel join karna zaroori hai.",
            reply_markup=kb
        )
        return

    # 🔥 FETCH ALL MATCHING MOVIES (NOT SINGLE)
    rows = db_get_movies_by_slug(slug)

    if not rows:
        send_message(
            uid,
            f"❌ Movie available nahi hai.\n{WEBSITE_URL}/movie/{quote_plus(slug)}"
        )
        return

    # 🔥 EXTRACT OPTIONS (season / episode / quality / language)
    options = extract_bot_options(rows)

    # 🎬 STEP PRIORITY ORDER
    for step, label in [
        ("season", "📺 Select Season"),
        ("episode", "🎞 Select Episode"),
        ("quality", "🎥 Select Quality"),
        ("language", "🌐 Select Language"),
    ]:
        if step in options:
            kb = build_option_keyboard(slug, options, step)
            send_message(uid, label, reply_markup=kb)
            return

    # 🟢 FINAL FALLBACK (only ONE version exists)
    r = rows[0]
    res = forward_message(uid, r["channel_id"], r["message_id"])
    if res.get("ok"):
        send_message(
            uid,
            f"🎬 {r.get('title','Movie')}\n⏳ Auto delete in {DELETE_AFTER_SECONDS}s"
    )

# =========================================================
# 🔍 JOIN CHECK HELPER
# =========================================================
def check_user_join(uid):
    try:
        for ch in REQUIRED_CHANNELS:
            res = tg_post("getChatMember", {"chat_id": ch, "user_id": uid})
            if not res.get("ok"):
                return False
            status = res.get("result", {}).get("status")
            if status in ("left", "kicked"):
                return False
        return True
    except Exception:
        logger.exception("check_user_join failed")
        return False


# --- webhook setter (runs at startup if WEBSITE_URL provided) ---
def set_webhook():
    if not WEBSITE_URL:
        logger.info("WEBSITE_URL not set — skipping webhook set")
        return
    payload = {"url": f"{WEBSITE_URL}/webhook"}
    if WEBHOOK_SECRET_TOKEN:
        payload["secret_token"] = WEBHOOK_SECRET_TOKEN
    try:
        r = requests.post(f"{API_BASE}/setWebhook", json=payload, timeout=10)
        logger.info("set_webhook: %s", r.text)
    except Exception:
        logger.exception("set_webhook failed")

# --- cache warmer (TMDB calls) ---
def cache_warmer():
    while True:
        try:
            if TMDB_KEY:
                requests.get("https://api.themoviedb.org/3/trending/movie/day", params={"api_key": TMDB_KEY}, timeout=6)
        except Exception:
            pass
        time.sleep(1800)


# --- Telegram Webhook Receiver ---
@app.post("/webhook")
def telegram_webhook():
    try:
        update = request.get_json(force=True, silent=True)
        if not update:
            return "no-update", 200

        # Process update
        threading.Thread(target=handle_update, args=(update,), daemon=True).start()

        return "ok", 200
    except Exception as e:
        logger.exception("webhook receive failed")
        return "error", 200

# --- startup ---
if __name__ == "__main__":

    ensure_json_files()   # 🔥 MUST BE FIRST

    logger.info("Starting Metflic backend")
    logger.info("BOT_USERNAME: %s", BOT_USERNAME)
    logger.info("WEBSITE_URL: %s", WEBSITE_URL or "not-set")
    logger.info("DATABASE_URL present: %s", bool(DATABASE_URL))
    logger.info("DB_CHANNEL: %s", DB_CHANNEL)
    logger.info("PYRO available: %s", PYRO)
    logger.info("TMDB_KEY present: %s", bool(TMDB_KEY))

    try:
        set_webhook()
    except Exception:
        logger.exception("set_webhook error")

    # cache warmer
    threading.Thread(target=cache_warmer, daemon=True).start()

    # ❌ Neon/Postgres backup disabled for SQLite/JSON mode
    # if USE_DB and ADMIN_ID:
    #     threading.Thread(
    #         target=backup_worker,
    #         args=(BACKUP_HOURS,),
    #         daemon=True
    #     ).start()

    app.run(host="0.0.0.0", port=PORT)
