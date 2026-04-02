import streamlit as st
from datetime import datetime, timezone
from agent import (
    ESPN_FEEDS,
    fetch_espn,
    fetch_nba_stats,
    deduplicate,
    enrich_with_gemini,
)

st.set_page_config(
    page_title="Sports Storyline Tracker",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 Sports Storyline Tracker")
st.caption("Top viral sports stories — ranked and enriched by AI. Refreshes every hour.")

sport_options = ["All"] + list(ESPN_FEEDS.keys())
sport_filter = st.selectbox("Filter by sport", sport_options)

@st.cache_data(ttl=3600, show_spinner="Fetching and ranking stories...")
def get_stories(sport: str) -> list[dict]:
    raw = []
    filter_val = None if sport == "All" else sport
    for s, url in ESPN_FEEDS.items():
        if filter_val and s != filter_val:
            continue
        raw.extend(fetch_espn(s, url))
    if not filter_val or filter_val == "NBA":
        raw.extend(fetch_nba_stats())
    unique = deduplicate(raw)
    return enrich_with_gemini(unique)

try:
    stories = get_stories(sport_filter)
except Exception as e:
    st.error(f"Failed to fetch stories: {e}")
    st.stop()

st.markdown(f"**{len(stories)} stories** · Last updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
st.divider()

for i, s in enumerate(stories, 1):
    virality = s.get("virality_score", 0)
    heat = s.get("heat_score", 0)
    sport = s.get("sport", "")
    title = s.get("title", "")
    vo_hook = s.get("vo_hook", title)
    context = s.get("historical_context", "")
    url = s.get("url", "")
    source = s.get("source", "")

    with st.container():
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"### #{i} &nbsp; `{sport}` &nbsp; [{title}]({url})")
        with col2:
            st.metric("Virality", f"{virality}/100")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"**VO Hook:** {vo_hook}")
        with col4:
            st.markdown(f"**Heat score:** {heat}/100")

        if context:
            st.markdown(f"**Context:** {context}")

        st.caption(f"Source: {source}")
        st.divider()
