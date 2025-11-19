"""
scrape_chanakya.py
Scrapes videos from a given channel using yt-dlp and builds a raw CSV.
Configurable: CHANNEL_URL, MAX_SCRAPE, MIN_AGE_DAYS, OUTPUT_RAW_CSV
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import yt_dlp

# ---------- USER CONFIG ----------
CHANNEL_URL = 'https://www.youtube.com/@THECHANAKYADIALOGUESHINDI/videos'
# Try to fetch more than needed, then filter by age. 
MAX_SCRAPE = 2500 
# Keep videos older than this many days so that view_count approximates long-term views
MIN_AGE_DAYS = 90
OUTPUT_RAW_CSV = 'raw_chanakya.csv'
# ---------------------------------

# Keyword classification maps (extend as needed)
GUEST_TYPE_KEYWORDS = {
    'Diplomat': ['Ambassador', 'Diplomat', 'IFS', 'Foreign Secretary', 'High Commissioner', 'S Jaishankar'],
    'Military_Veteran': ['General', 'Admiral', 'Air Marshal', 'Colonel', 'Major', 'Brigadier', 'MARCOS', 'Para SF', 'Veteran'],
    'Journalist': ['Journalist', 'Editor', 'Anchor'],
    'Academic': ['Professor', 'Historian', 'Author', 'Scholar'],
    'Politician': ['Minister', 'MP', 'MLA', 'Chief Minister', 'Prime Minister', 'PM Modi', 'Modi'],
}

TOPIC_CATEGORY_KEYWORDS = {
    'Geopolitics': ['Geopolitics', 'China', 'Pakistan', 'USA', 'Russia', 'World Order', 'Global'],
    'Defense_Analysis': ['Defense', 'Indian Army', 'Indian Navy', 'IAF', 'Military', 'Weapon', 'Missile'],
    'Indian_History': ['History', 'Mughal', 'Maratha', 'Ancient India', 'Medieval'],
    'Foreign_Policy': ['Foreign Policy', 'MEA', 'Diplomacy', 'Bilateral', 'UN'],
    'Internal_Affairs': ['Election', 'Politics', 'Economy', 'UCC', 'Article 370', 'GST'],
}

FORMAT_KEYWORDS = {
    'Panel_Discussion': ['Panel', 'Discussion', 'Roundtable'],
    'Interview': ['Interview', 'In Conversation With', 'With', 'Interview with', 'साक्षात्कार'],
    'Monologue_Explainer': ['Explained', 'Analysis', 'Decoded', 'Talk'],
}


def classify_by_keywords(text: str, keyword_dict: Dict[str, List[str]], default='Other'):
    if not text:
        return default
    text_l = text.lower()
    for category, keywords in keyword_dict.items():
        for kw in keywords:
            if kw.lower() in text_l:
                return category
    return default


def extract_channel_videos(channel_url: str, max_videos: int) -> List[Dict[str, Any]]:
    # Use yt-dlp to extract playlist entries (channel uploads list)
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
        'force_generic_extractor': True,
        'skip_download': True,
        'playlistend': max_videos
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"[scrape] extracting list from channel: {channel_url} ...")
        info = ydl.extract_info(channel_url, download=False)
    entries = info.get('entries', []) if info else []
    return entries


def fetch_video_details(video_url: str, ydl: yt_dlp.YoutubeDL):
    try:
        # Use the ydl instance passed in to avoid reinitializing each time
        video_info = ydl.extract_info(video_url, download=False)
        return video_info
    except Exception as e:
        print(f"[warn] failed to extract {video_url}: {e}")
        return None


def main():
    entries = extract_channel_videos(CHANNEL_URL, MAX_SCRAPE)
    print(f"[scrape] found {len(entries)} entries (may include unavailable videos).")

    rows = []
    ydl_opts = {'quiet': True, 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for idx, entry in enumerate(tqdm(entries, desc="listing videos")):
            # Each entry may be a dict with 'id' and 'url' or 'url' might be relative
            video_url = entry.get('url') or entry.get('id')
            if not video_url:
                continue
            # Normalize URL if needed
            if not video_url.startswith('http'):
                video_url = f"https://www.youtube.com/watch?v={video_url}"

            v = fetch_video_details(video_url, ydl)
            if v is None:
                continue

            # parse upload date if available
            upload_date = v.get('upload_date')  # YYYYMMDD string
            upload_datetime = None
            if upload_date:
                try:
                    upload_datetime = datetime.strptime(upload_date, "%Y%m%d")
                except:
                    upload_datetime = None

            view_count = v.get('view_count') or 0
            like_count = v.get('like_count')
            duration = v.get('duration') or 0
            description = v.get('description') or ''
            title = v.get('title') or ''
            tags = v.get('tags') or []
            comment_count = v.get('comment_count')
            thumbnails = v.get('thumbnails') or []

            # classification
            full_text = f"{title} {description}"
            guest_type = classify_by_keywords(full_text, GUEST_TYPE_KEYWORDS, default='Other')
            topic_category = classify_by_keywords(title, TOPIC_CATEGORY_KEYWORDS, default='General')
            format_type = classify_by_keywords(title, FORMAT_KEYWORDS, default='Interview')

            row = {
                'video_id': v.get('id'),
                'video_url': video_url,
                'title': title,
                'description': description,
                'uploader': v.get('uploader'),
                'upload_date': upload_date,
                'upload_datetime': upload_datetime.isoformat() if upload_datetime else None,
                'view_count': int(view_count) if view_count is not None else None,
                'like_count': int(like_count) if like_count is not None else None,
                'duration': int(duration),
                'tags': json.dumps(tags, ensure_ascii=False),
                'comment_count': int(comment_count) if comment_count is not None else None,
                'thumbnails': json.dumps(thumbnails, ensure_ascii=False),
                'guest_type': guest_type,
                'topic_category': topic_category,
                'format_type': format_type
            }
            rows.append(row)

            # polite pause to avoid throttling
            time.sleep(0.1)

    df = pd.DataFrame(rows)
    # compute age in days where possible
    now = datetime.now()
    def age_days_from_iso(iso):
        if not iso:
            return None
        try:
            dt = datetime.fromisoformat(iso)
            return (now - dt).days
        except:
            return None
    df['age_days'] = df['upload_datetime'].apply(age_days_from_iso)

    # save raw CSV
    df.to_csv(OUTPUT_RAW_CSV, index=False)
    print(f"[scrape] saved {len(df)} rows to {OUTPUT_RAW_CSV}")

    # filter to older than MIN_AGE_DAYS for stable targets
    df_filtered = df[df['age_days'].notnull() & (df['age_days'] >= MIN_AGE_DAYS)].copy()
    print(f"[scrape] after filtering for age >= {MIN_AGE_DAYS} days: {len(df_filtered)} rows available.")

    # If not enough rows, tell the user
    if len(df_filtered) < 1000: # Updated the check to 1000
        print("\nWARNING: After filtering for age we have fewer than 1000 rows.")
        print("Options:")
        print(f" - reduce MIN_AGE_DAYS (currently {MIN_AGE_DAYS})")
        print(f" - increase MAX_SCRAPE (currently {MAX_SCRAPE})")
        print("You can still use the dataset, but target stability may vary.")

    # Save filtered version too
    df_filtered.to_csv('raw_chanakya_filtered_by_age.csv', index=False)
    print("[scrape] also saved filtered CSV: raw_chanakya_filtered_by_age.csv")

if __name__ == '__main__':
    main()