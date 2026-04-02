import streamlit as st
from datetime import datetime, timezone
from agent import (
    ESPN_FEEDS,
    fetch_espn,
    fetch_cbs_sports,
    fetch_yahoo_sports,
    fetch_bleacher_report,
    fetch_google_news_rss,
    fetch_google_news_serper,
    fetch_nba_scoreboard,
    fetch_nba_leaders,
    fetch_nhl_stats,
    deduplicate,
    enrich_with_ai,
)

st.set_page_config(
    page_title="Sports Storyline Tracker",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 Sports Storyline Tracker")
st.caption("Top viral sports stories — ranked by AI using real news coverage, Reddit engagement, and live stats. Refreshes every hour.")

sport_options = ["All"] + list(ESPN_FEEDS.keys())
sport_filter = st.selectbox("Filter by sport", sport_options)

@st.cache_data(ttl=3600, show_spinner="Fetching and ranking stories...")
def get_stories(sport: str) -> list[dict]:
    filter_val = None if sport == "All" else sport

    raw = []
    raw.extend(fetch_espn(filter_val))
    raw.extend(fetch_cbs_sports(filter_val))
    raw.extend(fetch_yahoo_sports(filter_val))
    raw.extend(fetch_bleacher_report(filter_val))
    raw.extend(fetch_google_news_rss(filter_val))
    raw.extend(fetch_google_news_serper(filter_val))

    stats_context_parts = []
    if not filter_val or filter_val == "NBA":
        raw.extend(fetch_nba_scoreboard())
        nba_ctx = fetch_nba_leaders()
        if nba_ctx:
            stats_context_parts.append(nba_ctx)
    if not filter_val or filter_val == "NHL":
        nhl_stories, nhl_ctx = fetch_nhl_stats()
        raw.extend(nhl_stories)
        if nhl_ctx:
            stats_context_parts.append(nhl_ctx)

    unique = deduplicate(raw)
    return enrich_with_ai(unique, "\n\n".join(stats_context_parts))

try:
    stories = get_stories(sport_filter)
except Exception as e:
    st.error(f"Failed to fetch stories: {e}")
    st.stop()

st.markdown(f"**{len(stories)} stories** · Last updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
st.divider()

for i, s in enumerate(stories, 1):
    virality = s.get("virality_score", 0)
    heat     = s.get("heat_score", 0)
    sport    = s.get("sport", "")
    title    = s.get("title", "")
    vo_hook  = s.get("vo_hook", title)
    context  = s.get("historical_context", "")
    url      = s.get("url", "")
    source   = s.get("source", "")
    coverage = s.get("coverage_count", 0)
    reddit   = s.get("reddit_top_score", 0)

    with st.container():
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            st.markdown(f"### #{i} &nbsp; `{sport}` &nbsp; [{title}]({url})")
        with col2:
            st.metric("Virality", f"{virality}/100")
        with col3:
            st.metric("Heat", f"{heat}/100")

        st.markdown(f"**VO Hook:** {vo_hook}")

        if context:
            st.markdown(f"**Historical Context:** {context}")

        col4, col5 = st.columns(2)
        with col4:
            st.caption(f"📰 {coverage} news articles covering this story")
        with col5:
            st.caption(f"👍 {reddit} Reddit upvotes on top post")

        st.caption(f"Source: {source}")
        st.divider()
