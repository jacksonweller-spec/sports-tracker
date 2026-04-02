#!/usr/bin/env python3
"""
Sports Storyline Tracker Agent
================================
Fetches viral sports stories from ESPN, CBS Sports, Yahoo Sports, Bleacher Report,
Google News RSS, and official stats APIs (NBA, NHL). Ranks and enriches them with
Groq/Llama, then writes results to Supabase — or prints them in --test mode.

Usage
-----
  python agent.py           # full run → writes to Supabase
  python agent.py --test    # dry run  → prints to terminal, no DB writes
  python agent.py --test --sport nba
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.parse
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
    "NHL":    "https://www.espn.com/espn/rss/nhl/news",
    "SOCCER": "https://www.espn.com/espn/rss/soccer/news",
    "NCAAB":  "https://www.espn.com/espn/rss/ncb/news",
}

CBS_FEEDS: dict[str, str] = {
    "NBA":    "https://www.cbssports.com/rss/headlines/nba/",
    "NFL":    "https://www.cbssports.com/rss/headlines/nfl/",
    "NHL":    "https://www.cbssports.com/rss/headlines/nhl/",
    "SOCCER": "https://www.cbssports.com/rss/headlines/soccer/",
    "NCAAB":  "https://www.cbssports.com/rss/headlines/college-basketball/",
}

YAHOO_FEEDS: dict[str, str] = {
    "NBA":    "https://sports.yahoo.com/nba/rss.xml",
    "NFL":    "https://sports.yahoo.com/nfl/rss.xml",
    "NHL":    "https://sports.yahoo.com/nhl/rss.xml",
    "SOCCER": "https://sports.yahoo.com/soccer/rss.xml",
    "NCAAB":  "https://sports.yahoo.com/college-basketball/rss.xml",
}

REDDIT_SUBREDDITS: dict[str, str] = {
    "NBA":    "nba",
    "NFL":    "nfl",
    "NHL":    "hockey",
    "SOCCER": "soccer",
    "NCAAB":  "CollegeBasketball",
}

BLEACHER_REPORT_FEED = "https://bleacherreport.com/articles/feed"

NBA_REQUEST_HEADERS: dict[str, str] = {
    "Accept":             "application/json",
    "Accept-Language":    "en-US,en;q=0.9",
    "Connection":         "keep-alive",
    "Referer":            "https://www.nba.com/",
    "User-Agent":         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
}

REDDIT_HEADERS: dict[str, str] = {
    "User-Agent": "SportsTrackerBot/1.0 (research tool)",
}

RSS_PER_FEED = 5    # stories pulled per RSS feed
TOP_N        = 10   # final ranked stories written to Supabase / printed


# ── Generic RSS fetcher ────────────────────────────────────────────────────────

def fetch_rss_feed(source_name: str, feeds: dict[str, str],
                   sport_filter: Optional[str] = None) -> list[dict]:
    """Fetch stories from any dict of sport→RSS-URL."""
    stories: list[dict] = []
    selected = (
        {sport_filter.upper(): feeds[sport_filter.upper()]}
        if sport_filter and sport_filter.upper() in feeds
        else feeds
    )
    for sport, url in selected.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:RSS_PER_FEED]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                stories.append({
                    "source":    source_name,
                    "sport":     sport,
                    "title":     title,
                    "summary":   _strip_html(entry.get("summary", "")),
                    "url":       entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception as exc:
            _warn(f"{source_name} {sport}: {exc}")
    return stories


# ── News source fetchers ───────────────────────────────────────────────────────

def fetch_espn(sport_filter: Optional[str] = None) -> list[dict]:
    return fetch_rss_feed("ESPN", ESPN_FEEDS, sport_filter)


def fetch_cbs_sports(sport_filter: Optional[str] = None) -> list[dict]:
    return fetch_rss_feed("CBS Sports", CBS_FEEDS, sport_filter)


def fetch_yahoo_sports(sport_filter: Optional[str] = None) -> list[dict]:
    return fetch_rss_feed("Yahoo Sports", YAHOO_FEEDS, sport_filter)


def fetch_bleacher_report(sport_filter: Optional[str] = None) -> list[dict]:
    stories: list[dict] = []
    try:
        feed = feedparser.parse(BLEACHER_REPORT_FEED)
        for entry in feed.entries[:20]:
            title = entry.get("title", "").strip()
            if not title:
                continue
            sport = _infer_sport(title + " " + entry.get("summary", ""))
            if sport_filter and sport != sport_filter.upper():
                continue
            if sport not in ESPN_FEEDS:
                continue
            stories.append({
                "source":    "Bleacher Report",
                "sport":     sport,
                "title":     title,
                "summary":   _strip_html(entry.get("summary", "")),
                "url":       entry.get("link", ""),
                "published": entry.get("published", ""),
            })
    except Exception as exc:
        _warn(f"Bleacher Report: {exc}")
    return stories


def fetch_google_news_rss(sport_filter: Optional[str] = None) -> list[dict]:
    """Pull stories from Google News RSS — no API key required."""
    sport_queries: dict[str, str] = {
        "NBA":    "NBA basketball",
        "NFL":    "NFL football",
        "NHL":    "NHL hockey",
        "SOCCER": "soccer football MLS Premier League",
        "NCAAB":  "college basketball NCAA",
    }
    queries = (
        {sport_filter.upper(): sport_queries[sport_filter.upper()]}
        if sport_filter and sport_filter.upper() in sport_queries
        else sport_queries
    )
    stories: list[dict] = []
    for sport, q in queries.items():
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:RSS_PER_FEED]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                stories.append({
                    "source":    "Google News",
                    "sport":     sport,
                    "title":     title,
                    "summary":   _strip_html(entry.get("summary", "")),
                    "url":       entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception as exc:
            _warn(f"Google News RSS {sport}: {exc}")
    return stories


def fetch_google_news_serper(sport_filter: Optional[str] = None) -> list[dict]:
    """Query Serper's Google News API (requires SERPER_API_KEY)."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        _warn("SERPER_API_KEY not set — skipping Serper Google News")
        return []
    queries = (
        [f"{sport_filter} news highlights"]
        if sport_filter
        else [
            "viral sports moment today",
            "NBA breaking news",
            "NFL trade rumor",
            "sports record broken",
            "athlete injury update",
        ]
    )
    stories: list[dict] = []
    for query in queries:
        try:
            resp = requests.post(
                "https://google.serper.dev/news",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": 3},
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
            _warn(f"Serper '{query}': {exc}")
    return stories


# ── Stats API fetchers ─────────────────────────────────────────────────────────

def fetch_nba_scoreboard() -> list[dict]:
    """Today's NBA games from stats.nba.com."""
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
        game_headers = _nba_result_set(data, "GameHeader")
        line_scores  = _nba_result_set(data, "LineScore")
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
                stories.append({
                    "source":    "NBA Stats API",
                    "sport":     "NBA",
                    "title":     f"NBA: {away} vs {home} — {status}",
                    "summary":   f"Today's game ({today}). Status: {status}",
                    "url":       f"https://www.nba.com/game/{gid}",
                    "published": r.get("GAME_DATE_EST", today),
                })
    except Exception as exc:
        _warn(f"NBA scoreboard: {exc}")
    return stories


def fetch_nba_leaders() -> str:
    """
    Fetch current NBA scoring, assist, and rebound leaders.
    Returns a formatted string for inclusion in the AI prompt.
    """
    season = _current_nba_season()
    categories = [
        ("PTS", "Scoring leaders (PPG)"),
        ("AST", "Assist leaders (APG)"),
        ("REB", "Rebound leaders (RPG)"),
    ]
    lines: list[str] = ["NBA CURRENT SEASON LEADERS:"]
    for stat, label in categories:
        try:
            resp = requests.get(
                "https://stats.nba.com/stats/leagueleaders",
                headers=NBA_REQUEST_HEADERS,
                params={
                    "LeagueID":   "00",
                    "PerMode":    "PerGame",
                    "Scope":      "S",
                    "Season":     season,
                    "SeasonType": "Regular Season",
                    "StatCategory": stat,
                },
                timeout=12,
            )
            resp.raise_for_status()
            rs = resp.json().get("resultSet", {})
            headers = rs.get("headers", [])
            rows    = rs.get("rowSet", [])[:5]
            name_i  = headers.index("PLAYER") if "PLAYER" in headers else None
            stat_i  = headers.index(stat)     if stat in headers     else None
            if name_i is None or stat_i is None:
                continue
            entries = [f"{dict(zip(headers, r))['PLAYER']} {dict(zip(headers, r))[stat]}" for r in rows]
            lines.append(f"  {label}: {', '.join(entries)}")
            time.sleep(0.5)
        except Exception as exc:
            _warn(f"NBA leaders ({stat}): {exc}")
    return "\n".join(lines) if len(lines) > 1 else ""


def fetch_nhl_stats() -> tuple[list[dict], str]:
    """
    Fetch NHL scoring leaders from the official NHL API.
    Returns (stories_list, stats_context_string).
    """
    stories: list[dict] = []
    context_lines: list[str] = ["NHL CURRENT SEASON LEADERS:"]

    categories = [
        ("goals",   "Goal leaders"),
        ("assists", "Assist leaders"),
        ("points",  "Point leaders"),
    ]
    for cat, label in categories:
        try:
            resp = requests.get(
                f"https://api-web.nhle.com/v1/skater-stats-leaders/current?categories={cat}&limit=5",
                timeout=10,
            )
            resp.raise_for_status()
            leaders = resp.json().get(cat, [])
            entries = []
            for p in leaders:
                name  = f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}".strip()
                value = p.get("value", "")
                entries.append(f"{name} {value}")
                if cat == "goals" and int(value) >= 30:
                    stories.append({
                        "source":    "NHL Stats API",
                        "sport":     "NHL",
                        "title":     f"NHL: {name} has {value} goals this season",
                        "summary":   f"{name} is among the NHL's top goal scorers with {value} goals.",
                        "url":       "https://www.nhl.com/stats",
                        "published": datetime.now().strftime("%Y-%m-%d"),
                    })
            context_lines.append(f"  {label}: {', '.join(entries)}")
        except Exception as exc:
            _warn(f"NHL stats ({cat}): {exc}")

    return stories, "\n".join(context_lines) if len(context_lines) > 1 else ""


# ── Social engagement ──────────────────────────────────────────────────────────

def fetch_reddit_engagement(title: str, sport: str) -> dict:
    """
    Search the relevant sports subreddit for posts matching the story title.
    Returns dict with post_count, top_score, top_comments.
    """
    subreddit = REDDIT_SUBREDDITS.get(sport.upper(), "sports")
    result = {"reddit_posts": 0, "reddit_top_score": 0, "reddit_top_comments": 0}
    try:
        query = urllib.parse.quote(title[:80])
        url   = f"https://www.reddit.com/r/{subreddit}/search.json?q={query}&sort=top&t=week&limit=3&restrict_sr=1"
        resp  = requests.get(url, headers=REDDIT_HEADERS, timeout=8)
        if resp.status_code == 429:
            return result
        resp.raise_for_status()
        posts = resp.json().get("data", {}).get("children", [])
        result["reddit_posts"] = len(posts)
        if posts:
            top = posts[0]["data"]
            result["reddit_top_score"]    = top.get("score", 0)
            result["reddit_top_comments"] = top.get("num_comments", 0)
    except Exception:
        pass
    return result


# ── Google News coverage count ─────────────────────────────────────────────────

def fetch_coverage_count(title: str) -> int:
    """Number of Google News RSS results for a story title."""
    try:
        query = urllib.parse.quote(title)
        url   = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed  = feedparser.parse(url)
        return len(feed.entries)
    except Exception:
        return 0


# ── Deduplication ──────────────────────────────────────────────────────────────

def deduplicate(stories: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for s in stories:
        key = re.sub(r"[^a-z0-9 ]", "", s["title"].lower())
        key = " ".join(key.split())
        if key and key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ── AI enrichment ──────────────────────────────────────────────────────────────

ENRICHMENT_PROMPT = """\
You are a sports media analyst at a major broadcast network.
Evaluate these sports news stories for virality potential and craft broadcast content.

SIGNAL GUIDE — use all of these when scoring:
• coverage_count     — Google News articles covering this story. Higher = more mainstream traction.
• reddit_top_score   — Upvotes on the top Reddit post about this story. Measures fan engagement.
• reddit_top_comments — Comments on the top Reddit post. High comments = divisive or exciting story.
• reddit_posts       — Number of Reddit posts found. More posts = broader discussion.

STATS CONTEXT (use to identify historically rare performances):
{stats_context}

For any story involving a player or team in the stats context, actively look for historically
rare angles — e.g. "first player since Larry Bird in 1983 to do X", "only the 4th time in
NHL history", "no rookie has done this since". Be specific with years, player names, and
comparisons. Put these findings in historical_context.

Return a JSON array (same length, same order) where every element has EXACTLY these fields:
  title              — original title, unchanged
  sport              — sport name in ALL CAPS (NBA, NFL, NHL, SOCCER, NCAAB, etc.)
  virality_score     — integer 1-100 (100 = all-time viral potential)
  heat_score         — integer 1-100 (current buzz/momentum, may differ from virality)
  historical_context — 1-2 sentences with specific historical comparison or stat rarity
  vo_hook            — punchy VO opening line a TV anchor would read cold (≤25 words)

Respond with ONLY a valid JSON array. No markdown fences, no extra text.

Stories:
{stories_json}
"""


def enrich_with_ai(stories: list[dict], stats_context: str) -> list[dict]:
    """Rank and enrich stories using Groq/Llama with real data signals."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        sys.exit("[ERROR] GROQ_API_KEY not set in .env")

    client     = Groq(api_key=api_key)
    candidates = stories[:TOP_N * 2]

    print("      Fetching coverage counts and Reddit engagement...", flush=True)
    slim: list[dict] = []
    for i, s in enumerate(candidates):
        coverage = fetch_coverage_count(s["title"])
        reddit   = fetch_reddit_engagement(s["title"], s.get("sport", ""))
        slim.append({
            "id":                  i,
            "title":               s["title"],
            "sport":               s["sport"],
            "summary":             s["summary"],
            "coverage_count":      coverage,
            "reddit_top_score":    reddit["reddit_top_score"],
            "reddit_top_comments": reddit["reddit_top_comments"],
            "reddit_posts":        reddit["reddit_posts"],
        })
        time.sleep(0.3)  # gentle rate limiting

    prompt = ENRICHMENT_PROMPT.format(
        stats_context=stats_context or "No stats context available.",
        stories_json=json.dumps(slim, indent=2),
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        enriched: list[dict] = json.loads(raw)
    except json.JSONDecodeError as exc:
        _warn(f"AI returned non-JSON — falling back to placeholders. {exc}")
        enriched = [
            {
                "title":              s["title"],
                "sport":              s["sport"],
                "virality_score":     50,
                "heat_score":         50,
                "historical_context": "(enrichment unavailable)",
                "vo_hook":            s["title"],
            }
            for s in candidates
        ]

    title_to_original = {s["title"]: s for s in stories}
    merged: list[dict] = []
    for e in enriched:
        base = title_to_original.get(e.get("title", ""), {})
        merged.append({**base, **e})

    merged.sort(key=lambda x: x.get("virality_score", 0), reverse=True)
    return merged[:TOP_N]


def mock_enrich(stories: list[dict]) -> list[dict]:
    return [
        {
            **s,
            "virality_score":     0,
            "heat_score":         0,
            "historical_context": "(AI enrichment skipped)",
            "vo_hook":            s["title"],
        }
        for s in stories[:TOP_N]
    ]


# ── Output: Supabase ───────────────────────────────────────────────────────────

def write_to_supabase(stories: list[dict]) -> None:
    try:
        from supabase import create_client
    except ImportError:
        sys.exit("[ERROR] supabase-py not installed.")

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        sys.exit("[ERROR] SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in .env")

    db  = create_client(url, key)
    now = datetime.now(timezone.utc).isoformat()
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
    result = db.table("storylines").upsert(rows, on_conflict="title").execute()
    print(f"[Supabase] Upserted {len(result.data)} rows at {now}")


# ── Output: terminal ───────────────────────────────────────────────────────────

def print_results(stories: list[dict]) -> None:
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
        reddit_score = s.get("reddit_top_score", 0)
        coverage     = s.get("coverage_count", 0)

        print(f"#{rank:<2}  [{sport}]  🔥 {virality}/100 virality  |  heat {bar} {heat}")
        print(f"    {s.get('title','')}")
        print()
        print(f"    VO HOOK  ▶  {s.get('vo_hook','')}")
        print()
        ctx = s.get("historical_context", "")
        for line in textwrap.wrap(f"    CONTEXT  ▶  {ctx}", width=72):
            print(line)
        print()
        print(f"    📰 {coverage} news articles  |  👍 {reddit_score} Reddit upvotes")
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


def _current_nba_season() -> str:
    now = datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _infer_sport(text: str) -> str:
    text_upper = text.upper()
    mapping = {
        "NBA":    ["NBA", "LAKERS", "CELTICS", "WARRIORS", "KNICKS", "HEAT", "BUCKS",
                   "SUNS", "NUGGETS", "SPURS", "NETS", "SIXERS", "CLIPPERS"],
        "NFL":    ["NFL", "FOOTBALL", "TOUCHDOWN", "QUARTERBACK", "SUPER BOWL",
                   "PATRIOTS", "CHIEFS", "COWBOYS", "EAGLES", "49ERS"],
        "NHL":    ["NHL", "HOCKEY", "STANLEY CUP", "OVECHKIN", "MCDAVID",
                   "MAPLE LEAFS", "BRUINS", "RANGERS", "PENGUINS"],
        "SOCCER": ["SOCCER", "FOOTBALL", "FIFA", "PREMIER LEAGUE", "LA LIGA",
                   "MLS", "CHAMPIONS LEAGUE", "BUNDESLIGA", "SERIE A", "MESSI",
                   "RONALDO", "HAALAND", "MBAPPÉ"],
        "NCAAB":  ["COLLEGE BASKETBALL", "NCAA", "MARCH MADNESS", "FINAL FOUR",
                   "DUKE", "KENTUCKY", "KANSAS", "GONZAGA", "UNC", "UCONN"],
    }
    for sport, keywords in mapping.items():
        if any(k in text_upper for k in keywords):
            return sport
    return "SPORTS"


def _heat_bar(score: int, width: int = 10) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(test_mode: bool = False, sport_filter: Optional[str] = None,
        no_ai: bool = False) -> None:

    print("[1/4] Fetching stories...", flush=True)

    raw: list[dict] = []
    raw.extend(fetch_espn(sport_filter))
    raw.extend(fetch_cbs_sports(sport_filter))
    raw.extend(fetch_yahoo_sports(sport_filter))
    raw.extend(fetch_bleacher_report(sport_filter))
    raw.extend(fetch_google_news_rss(sport_filter))
    raw.extend(fetch_google_news_serper(sport_filter))

    # Stats APIs
    stats_context_parts: list[str] = []

    if not sport_filter or sport_filter.upper() == "NBA":
        raw.extend(fetch_nba_scoreboard())
        nba_ctx = fetch_nba_leaders()
        if nba_ctx:
            stats_context_parts.append(nba_ctx)

    if not sport_filter or sport_filter.upper() == "NHL":
        nhl_stories, nhl_ctx = fetch_nhl_stats()
        raw.extend(nhl_stories)
        if nhl_ctx:
            stats_context_parts.append(nhl_ctx)

    stats_context = "\n\n".join(stats_context_parts)

    print(f"      Collected {len(raw)} raw stories across all sources")

    print("[2/4] Deduplicating...", flush=True)
    unique = deduplicate(raw)
    print(f"      {len(unique)} unique stories after deduplication")

    if not unique:
        print("[WARN] No stories found.")
        return

    if no_ai:
        print("[3/4] Skipping AI enrichment (--no-ai)...", flush=True)
        enriched = mock_enrich(unique)
    else:
        print(f"[3/4] Enriching top stories with AI + real signals...", flush=True)
        enriched = enrich_with_ai(unique, stats_context)
        print(f"      Received {len(enriched)} enriched stories (top {TOP_N})")

    print("[4/4] Writing results...", flush=True)
    if test_mode:
        print_results(enriched)
    else:
        write_to_supabase(enriched)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sports Storyline Tracker"
    )
    parser.add_argument("--test",    action="store_true",
                        help="Print to terminal instead of writing to Supabase")
    parser.add_argument("--sport",   metavar="SPORT", default=None,
                        help=f"Filter to one sport: {', '.join(ESPN_FEEDS.keys())}")
    parser.add_argument("--no-ai",   action="store_true",
                        help="Skip AI enrichment")
    args = parser.parse_args()
    run(test_mode=args.test, sport_filter=args.sport, no_ai=args.no_ai)


if __name__ == "__main__":
    main()
