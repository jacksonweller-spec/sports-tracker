#!/usr/bin/env python3
"""
Sports Storyline Tracker Agent
================================
Fetches viral sports stories from ESPN RSS, NBA Stats API, and Google News
(via Serper), ranks and enriches them with Claude Haiku, then writes results
to Supabase — or prints them to the terminal in --test mode.

Usage
-----
  python agent.py           # full run → writes to Supabase
  python agent.py --test    # dry run  → prints to terminal, no DB writes
  python agent.py --test --sport nba   # filter to one sport
"""

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from typing import Optional

import feedparser
from groq import Groq
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

ESPN_FEEDS: dict[str, str] = {
    "NBA":    "https://www.espn.com/espn/rss/nba/news",
    "NFL":    "https://www.espn.com/espn/rss/nfl/news",
    "MLB":    "https://www.espn.com/espn/rss/mlb/news",
    "NHL":    "https://www.espn.com/espn/rss/nhl/news",
    "SOCCER": "https://www.espn.com/espn/rss/soccer/news",
    "TENNIS": "https://www.espn.com/espn/rss/tennis/news",
    "GOLF":   "https://www.espn.com/espn/rss/golf/news",
    "MMA":    "https://www.espn.com/espn/rss/mma/news",
}

SERPER_QUERIES: list[str] = [
    "viral sports moment today",
    "NBA breaking news",
    "NFL trade rumor",
    "sports record broken",
    "athlete injury update",
]

NBA_REQUEST_HEADERS: dict[str, str] = {
    "Accept":             "application/json",
    "Accept-Language":    "en-US,en;q=0.9",
    "Connection":         "keep-alive",
    "Referer":            "https://www.nba.com/",
    "User-Agent":         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
}

ESPN_PER_FEED = 5    # stories pulled per ESPN sport feed
SERPER_PER_Q  = 3    # news results per Serper query
TOP_N         = 10   # final ranked stories written to Supabase / printed


# ── Data fetchers ──────────────────────────────────────────────────────────────

def fetch_espn(sport_filter: Optional[str] = None) -> list[dict]:
    """Parse ESPN RSS feeds and return raw story dicts."""
    stories: list[dict] = []
    feeds = (
        {sport_filter.upper(): ESPN_FEEDS[sport_filter.upper()]}
        if sport_filter and sport_filter.upper() in ESPN_FEEDS
        else ESPN_FEEDS
    )
    for sport, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:ESPN_PER_FEED]:
                stories.append({
                    "source":     "ESPN",
                    "sport":      sport,
                    "title":      entry.get("title", "").strip(),
                    "summary":    _strip_html(entry.get("summary", "")),
                    "url":        entry.get("link", ""),
                    "published":  entry.get("published", ""),
                })
        except Exception as exc:
            _warn(f"ESPN {sport}: {exc}")
    return stories


def fetch_nba_stats() -> list[dict]:
    """
    Pull today's NBA scoreboard from stats.nba.com.
    Returns one story per game with team names and status.
    """
    stories: list[dict] = []
    try:
        today = datetime.now().strftime("%m/%d/%Y")
        resp = requests.get(
            "https://stats.nba.com/stats/scoreboardv2",
            headers=NBA_REQUEST_HEADERS,
            params={"DayOffset": "0", "LeagueID": "00", "gameDate": today},
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()

        # LineScore has TEAM_ABBREVIATION; GameHeader has status text
        game_headers = _nba_result_set(data, "GameHeader")
        line_scores  = _nba_result_set(data, "LineScore")

        # Map game_id → team abbreviations from LineScore
        game_teams: dict[str, list[str]] = {}
        if line_scores:
            h = line_scores["headers"]
            for row in line_scores["rows"]:
                r = dict(zip(h, row))
                gid = r.get("GAME_ID", "")
                game_teams.setdefault(gid, []).append(r.get("TEAM_ABBREVIATION", "?"))

        if game_headers:
            h = game_headers["headers"]
            for row in game_headers["rows"]:
                r      = dict(zip(h, row))
                gid    = r.get("GAME_ID", "")
                status = r.get("GAME_STATUS_TEXT", "Scheduled").strip()
                teams  = game_teams.get(gid, ["TBD", "TBD"])
                away, home = (teams + ["TBD"])[:2]
                title = f"NBA: {away} vs {home} — {status}"
                stories.append({
                    "source":    "NBA Stats API",
                    "sport":     "NBA",
                    "title":     title,
                    "summary":   f"Today's game ({today}). Status: {status}",
                    "url":       f"https://www.nba.com/game/{gid}",
                    "published": r.get("GAME_DATE_EST", today),
                })
    except Exception as exc:
        _warn(f"NBA Stats API: {exc}")
    return stories


def fetch_google_news(sport_filter: Optional[str] = None) -> list[dict]:
    """Query Serper's Google News API for trending sports stories."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        _warn("SERPER_API_KEY not set — skipping Google News")
        return []

    stories: list[dict] = []
    queries = (
        [f"{sport_filter} news highlights"]
        if sport_filter
        else SERPER_QUERIES
    )

    for query in queries:
        try:
            resp = requests.post(
                "https://google.serper.dev/news",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": SERPER_PER_Q},
                timeout=10,
            )
            resp.raise_for_status()
            for item in resp.json().get("news", []):
                stories.append({
                    "source":    "Google News (Serper)",
                    "sport":     _infer_sport(item.get("title", "") + " " + item.get("snippet", "")),
                    "title":     item.get("title", "").strip(),
                    "summary":   item.get("snippet", "").strip(),
                    "url":       item.get("link", ""),
                    "published": item.get("date", ""),
                })
        except Exception as exc:
            _warn(f"Serper query '{query}': {exc}")

    return stories


# ── Deduplication ──────────────────────────────────────────────────────────────

def deduplicate(stories: list[dict]) -> list[dict]:
    """
    Remove near-duplicate headlines by normalising titles and keeping the
    first occurrence of any title whose normalised form was already seen.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for s in stories:
        key = re.sub(r"[^a-z0-9 ]", "", s["title"].lower())
        key = " ".join(key.split())  # collapse whitespace
        if key and key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ── Google News coverage signal ────────────────────────────────────────────────

def fetch_coverage_count(title: str) -> int:
    """Return the number of Google News RSS results for a story title."""
    try:
        import urllib.parse
        query = urllib.parse.quote(title)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        return len(feed.entries)
    except Exception:
        return 0


# ── AI enrichment ──────────────────────────────────────────────────────────────

ENRICHMENT_PROMPT = """\
You are a sports media analyst at a major broadcast network.
Evaluate these sports news stories for virality potential and craft broadcast content.

Each story includes a `coverage_count` — the number of distinct news articles currently
covering that story on Google News. Use this as a real signal of public interest:
higher coverage = more viral momentum. Weight it heavily alongside the story's content.

Return a JSON array (same length, same order) where every element has EXACTLY these fields:
  title              — original title, unchanged
  sport              — sport name in ALL CAPS (NBA, NFL, MLB, NHL, SOCCER, etc.)
  virality_score     — integer 1-100 (100 = all-time viral potential)
  heat_score         — integer 1-100 (current buzz/momentum, may differ from virality)
  historical_context — 1-2 sentences comparing to historical precedent or stat
  vo_hook            — punchy VO opening line a TV anchor would read cold (≤25 words)

Respond with ONLY a valid JSON array. No markdown fences, no extra text.

Stories:
{stories_json}
"""

def enrich_with_gemini(stories: list[dict]) -> list[dict]:
    """
    Send raw stories to Gemini Flash. Returns enriched list sorted by
    virality_score descending, trimmed to TOP_N.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        sys.exit("[ERROR] GROQ_API_KEY not set in .env")

    client = Groq(api_key=api_key)

    candidates = stories[:TOP_N * 2]
    print("      Fetching Google News coverage counts...", flush=True)
    slim = [
        {
            "id":             i,
            "title":          s["title"],
            "sport":          s["sport"],
            "summary":        s["summary"],
            "coverage_count": fetch_coverage_count(s["title"]),
        }
        for i, s in enumerate(candidates)
    ]

    prompt = ENRICHMENT_PROMPT.format(stories_json=json.dumps(slim, indent=2))

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        raw = completion.choices[0].message.content.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",       "", raw)
        enriched: list[dict] = json.loads(raw)

    except json.JSONDecodeError as exc:
        _warn(f"Gemini returned non-JSON — falling back to placeholders. {exc}")
        enriched = [
            {
                "title":              s["title"],
                "sport":              s["sport"],
                "virality_score":     50,
                "heat_score":         50,
                "historical_context": "(enrichment unavailable)",
                "vo_hook":            s["title"],
            }
            for s in stories
        ]

    title_to_original: dict[str, dict] = {s["title"]: s for s in stories}
    merged: list[dict] = []
    for e in enriched:
        base = title_to_original.get(e.get("title", ""), {})
        merged.append({**base, **e})

    merged.sort(key=lambda x: x.get("virality_score", 0), reverse=True)
    return merged[:TOP_N]


def mock_enrich(stories: list[dict]) -> list[dict]:
    """Placeholder enrichment used in --no-ai mode."""
    return [
        {
            **s,
            "virality_score":     0,
            "heat_score":         0,
            "historical_context": "(AI enrichment skipped — run without --no-ai to enable)",
            "vo_hook":            s["title"],
        }
        for s in stories[:TOP_N]
    ]


# ── Output: Supabase ───────────────────────────────────────────────────────────

def write_to_supabase(stories: list[dict]) -> None:
    """
    Upsert enriched stories into the `storylines` table.

    Required env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

    Table schema (create this in Supabase SQL editor):
    ─────────────────────────────────────────────────
    create table storylines (
      id                 uuid primary key default gen_random_uuid(),
      fetched_at         timestamptz not null,
      sport              text,
      title              text unique,
      summary            text,
      source             text,
      url                text,
      published_at       text,
      virality_score     int,
      heat_score         int,
      historical_context text,
      vo_hook            text,
      raw                jsonb
    );
    create index on storylines (sport);
    create index on storylines (virality_score desc);
    ─────────────────────────────────────────────────
    """
    try:
        from supabase import create_client
    except ImportError:
        sys.exit("[ERROR] supabase-py not installed. Run: pip install supabase")

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        sys.exit("[ERROR] SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in .env")

    client = create_client(url, key)
    now    = datetime.now(timezone.utc).isoformat()

    rows = [
        {
            "fetched_at":         now,
            "sport":              s.get("sport", ""),
            "title":              s.get("title", ""),
            "summary":            s.get("summary", ""),
            "source":             s.get("source", ""),
            "url":                s.get("url", ""),
            "published_at":       s.get("published", ""),
            "virality_score":     s.get("virality_score", 0),
            "heat_score":         s.get("heat_score", 0),
            "historical_context": s.get("historical_context", ""),
            "vo_hook":            s.get("vo_hook", ""),
            "raw":                json.dumps(s),
        }
        for s in stories
    ]

    result = (
        client.table("storylines")
        .upsert(rows, on_conflict="title")
        .execute()
    )
    print(f"[Supabase] Upserted {len(result.data)} rows at {now}")


# ── Output: terminal (--test mode) ────────────────────────────────────────────

def print_results(stories: list[dict]) -> None:
    """Pretty-print enriched stories to stdout."""
    divider = "─" * 72
    print(f"\n{'═' * 72}")
    print(f"  SPORTS STORYLINE TRACKER  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {len(stories)} top stories  (--test mode, no DB write)")
    print(f"{'═' * 72}\n")

    for rank, s in enumerate(stories, start=1):
        virality = s.get("virality_score", "?")
        heat     = s.get("heat_score",     "?")
        sport    = s.get("sport",          "?")
        bar      = _heat_bar(heat if isinstance(heat, int) else 0)

        print(f"#{rank:<2}  [{sport}]  🔥 {virality}/100 virality  |  heat {bar} {heat}")
        print(f"    {s.get('title','')}")
        print()
        print(f"    VO HOOK  ▶  {s.get('vo_hook','')}")
        print()
        # Wrap historical context
        ctx = s.get("historical_context", "")
        for line in textwrap.wrap(f"    CONTEXT  ▶  {ctx}", width=72):
            print(line)
        print()
        print(f"    SOURCE: {s.get('source','')}  |  {s.get('url','')[:60]}...")
        print(divider)
        print()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _nba_result_set(data: dict, name: str) -> Optional[dict]:
    for rs in data.get("resultSets", []):
        if rs["name"] == name:
            return {"headers": rs["headers"], "rows": rs["rowSet"]}
    return None


def _infer_sport(text: str) -> str:
    text_upper = text.upper()
    mapping = {
        "NBA":    ["NBA", "BASKETBALL", "LAKERS", "CELTICS", "WARRIORS"],
        "NFL":    ["NFL", "FOOTBALL", "TOUCHDOWN", "QUARTERBACK"],
        "MLB":    ["MLB", "BASEBALL", "HOMERUN", "HOME RUN", "WORLD SERIES"],
        "NHL":    ["NHL", "HOCKEY", "STANLEY CUP"],
        "SOCCER": ["SOCCER", "FIFA", "PREMIER LEAGUE", "LA LIGA", "MLS"],
        "TENNIS": ["TENNIS", "WIMBLEDON", "US OPEN", "FRENCH OPEN", "ATP", "WTA"],
        "GOLF":   ["GOLF", "PGA", "MASTERS", "US OPEN PGA"],
        "MMA":    ["UFC", "MMA", "BOXING"],
    }
    for sport, keywords in mapping.items():
        if any(k in text_upper for k in keywords):
            return sport
    return "SPORTS"


def _heat_bar(score: int, width: int = 10) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(test_mode: bool = False, sport_filter: Optional[str] = None, no_ai: bool = False) -> None:
    print("[1/4] Fetching stories...", flush=True)

    raw_stories: list[dict] = []
    raw_stories.extend(fetch_espn(sport_filter))
    raw_stories.extend(fetch_google_news(sport_filter))

    # Only fetch NBA stats when relevant
    if not sport_filter or sport_filter.upper() == "NBA":
        raw_stories.extend(fetch_nba_stats())

    print(f"      Collected {len(raw_stories)} raw stories across all sources")

    print("[2/4] Deduplicating...", flush=True)
    unique = deduplicate(raw_stories)
    print(f"      {len(unique)} unique stories after deduplication")

    if not unique:
        print("[WARN] No stories found. Check network access and API keys.")
        return

    if no_ai:
        print("[3/4] Skipping AI enrichment (--no-ai mode)...", flush=True)
        enriched = mock_enrich(unique)
    else:
        print(f"[3/4] Sending {len(unique)} stories to Gemini for ranking & enrichment...", flush=True)
        enriched = enrich_with_gemini(unique)
        print(f"      Received {len(enriched)} enriched stories (top {TOP_N})")

    print("[4/4] Writing results...", flush=True)
    if test_mode:
        print_results(enriched)
    else:
        write_to_supabase(enriched)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sports Storyline Tracker — fetches, ranks, and enriches viral sports stories"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Print results to terminal instead of writing to Supabase",
    )
    parser.add_argument(
        "--sport",
        metavar="SPORT",
        default=None,
        help=f"Filter to one sport: {', '.join(ESPN_FEEDS.keys())}",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip Gemini enrichment — just fetch and print raw stories",
    )
    args = parser.parse_args()
    run(test_mode=args.test, sport_filter=args.sport, no_ai=args.no_ai)


if __name__ == "__main__":
    main()
