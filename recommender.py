import math
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (NUM_CATEGORIES, GAMES_PER_CATEGORY, MIN_PLAYTIME, CHILL_TAGS)
from database import get_metadata


def extract_tags(tag_str):
    """Extracts tag names from a string containing 'tag:weight' tokens."""
    if not tag_str:
        return set()
    tags = set()
    for token in tag_str.split():
        if ":" in token:
            tags.add(token.rsplit(":", 1)[0])
        else:
            tags.add(token)
    return tags


def extract_tags_with_counts(tag_str):
    """Extracts tag names and their counts from a string containing 'tag:count' tokens."""
    if not tag_str:
        return {}
    tags = {}
    for token in tag_str.split():
        if ":" in token:
            parts = token.rsplit(":", 1)
            try:
                tags[parts[0]] = float(parts[1])
            except ValueError:
                tags[parts[0]] = 1.0
        else:
            tags[token] = 1.0
    return tags


def get_game_weighted_tags(conn, appid):
    """
    Fetch tags for a game and calculate their weights directly in logic.
    Returns (list_of_tags, dict_of_weights).
    """
    rows = conn.execute("""
        SELECT t.name, gt.count
        FROM game_tags gt
        JOIN tags t ON gt.tag_id = t.id
        WHERE gt.appid = ?
    """, (appid,)).fetchall()

    tags = []
    raw_counts = {}
    for r in rows:
        tags.append(r['name'])
        raw_counts[r['name']] = r['count']

    if not tags:
        return [], {}

    # 1. Position weight (Steam tags are ordered by popularity)
    pos_weights = {}
    for i, tag in enumerate(tags):
        # 1.0 for first tag, down to 0.1 for last tag
        pos_weights[tag] = max(0.1, 1.0 - (i / len(tags)) * 0.9)

    # 2. Relative count weight
    count_weights = {}
    max_count = max(raw_counts.values()) if raw_counts else 1
    default_weight = 0.5
    for tag in tags:
        count_weights[tag] = (raw_counts[tag] / max_count) * (1.0 - default_weight) + default_weight

    # 3. Combine
    weights = {}
    for tag in tags:
        combined = pos_weights[tag] * count_weights[tag]
        weights[tag] = round(max(0.01, min(1.0, combined)), 4)

    return tags, weights


def generate_recommendations(conn, stop_event=None):
    """Generate ML-based game recommendations."""
    now = int(time.time())

    def check_stop():
        if stop_event and stop_event.is_set():
            raise InterruptedError("Training cancelled")

    # 2. Rated games (profile)
    # Include games with either a permanent rating or an active temporary rating
    rated_db_games = [dict(r) for r in conn.execute("""
        SELECT * FROM games 
        WHERE ignored = 0 
        AND (rating > 0 OR (temp_rating IS NOT NULL AND temp_rating_until > ?))
    """, (now,)).fetchall()]

    # Apply temporary ratings to the profile
    for g in rated_db_games:
        if g.get('temp_rating') and g.get('temp_rating_until', 0) > now:
            g['rating'] = g['temp_rating']

    # 3. Candidate pool
    min_playtime = int(get_metadata('MIN_PLAYTIME', MIN_PLAYTIME))
    all_candidates = [dict(r) for r in conn.execute("""
                                                    SELECT *
                                                    FROM games
                                                    WHERE ignored = 0
                                                      AND (ignore_until = 0 OR ignore_until < ?)
                                                      AND playtime >= ?
                                                      AND (tags IS NOT NULL AND tags != '')
                                                    ORDER BY (temp_rating IS NOT NULL AND temp_rating_until > ?) DESC, 
                                                             (rating > 0 AND finished = 0) DESC,
                                                             playtime DESC
                                                    """, (now, min_playtime, now)).fetchall()]

    # Separate backlog (unfinished) from finished games for recommendation logic
    # We take enough games to satisfy the requested number of categories
    num_categories = int(get_metadata('NUM_CATEGORIES', NUM_CATEGORIES))
    games_per_category = int(get_metadata('GAMES_PER_CATEGORY', GAMES_PER_CATEGORY))
    limit = max(300, num_categories * games_per_category + 50)
    
    backlog = [g for g in all_candidates if not g['finished']][:limit]
    finished_candidates = [g for g in all_candidates if g['finished']]

    if not rated_db_games:
        return None, None, None, None, None

    # Custom token pattern to keep tags like "action_rpg" and "1990's" intact
    vectorizer = TfidfVectorizer(stop_words='english', max_features=300, token_pattern=r"(?u)\S+")

    # We will build a list of space-separated tags (without weights) to fit the vectorizer
    all_tags_list = []

    # We will also keep track of the weight dictionary for each game
    game_weights = []

    # Set to track metadata tags (dev/pub) to exclude from cluster names
    meta_tags = set()

    # Profile building
    for g in rated_db_games:
        check_stop()
        tags, weights = get_game_weighted_tags(conn, g['appid'])
        if not tags:
            # Fallback for games with missing normalized tags but having the tags string
            # This shouldn't happen much with the new filter, but good for robustness
            tags_dict = extract_tags_with_counts(g.get('tags', ''))
            tags = list(tags_dict.keys())
            weights = {t: c/100.0 for t, c in tags_dict.items()}

        s_db = g.get('steam_score')
        if s_db is None:
            s_db = 5.0
        
        rating = float(g.get('rating') or 0)
        
        # Give higher weight to Up Next games in the profile
        if g.get('temp_rating') and g.get('temp_rating_until', 0) > now:
            weight = rating * 1.5
        else:
            diff = -math.fabs(rating - float(s_db))
            weight = rating * 1.1 + diff

        # Add metadata as tags with max weight
        if g.get('developer'):
            devs = [d.strip() for d in g['developer'].split(',')]
            for dev in devs:
                dev_tag = dev.replace(' ', '_').lower()
                if dev_tag not in tags:
                    tags.append(dev_tag)
                weights[dev_tag] = 1.0
                meta_tags.add(dev_tag)
        if g.get('publisher'):
            pubs = [p.strip() for p in g['publisher'].split(',')]
            for pub in pubs:
                pub_tag = pub.replace(' ', '_').lower()
                if pub_tag not in tags:
                    tags.append(pub_tag)
                weights[pub_tag] = 1.0
                meta_tags.add(pub_tag)

        # Reconstruct tag string for recommender functions and tfidf
        # Multiply tags based on their weights to give them more importance in TF-IDF
        weighted_tags = []
        for t in tags:
            w = weights.get(t, 0.5)
            # Repeat the tag based on weight to influence TF-IDF
            count = max(1, int(w * 10))
            weighted_tags.extend([t] * count)

        g['tags'] = " ".join([f"{t}:{weights.get(t, 0.5)*100}" for t in tags]) # Re-add counts for build_explanation compatibility
        g['weight'] = weight
        all_tags_list.append(" ".join(weighted_tags))
        game_weights.append({"weight": weight, "tag_weights": weights})

    # Backlog prep
    for g in backlog + finished_candidates:
        check_stop()
        tags, weights = get_game_weighted_tags(conn, g['appid'])
        if not tags:
            tags_dict = extract_tags_with_counts(g.get('tags', ''))
            tags = list(tags_dict.keys())
            weights = {t: c/100.0 for t, c in tags_dict.items()}

        if g['difficulty'] and g['difficulty'] != 'Easy':
            diff_tag = str(g['difficulty']).replace(' ', '_').lower()
            if diff_tag not in tags:
                tags.append(diff_tag)
            weights[diff_tag] = 0.8 # High weight but not absolute

        # Reconstruct tag string for recommender functions and tfidf
        weighted_tags = []
        for t in tags:
            w = weights.get(t, 0.5)
            count = max(1, int(w * 10))
            weighted_tags.extend([t] * count)

        g['tags'] = " ".join([f"{t}:{weights.get(t, 0.5)*100}" for t in tags])
        all_tags_list.append(" ".join(weighted_tags))

    # Vectorize
    check_stop()
    tfidf_matrix = vectorizer.fit_transform(all_tags_list)
    
    rated_count = len(rated_db_games)
    rated_matrix = tfidf_matrix[:rated_count]
    candidate_matrix = tfidf_matrix[rated_count:]

    # Weight the profile matrix
    weighted_profile_rows = []
    total_profile_weight = 0
    for i in range(rated_count):
        row = rated_matrix[i].toarray()[0]
        # Multiply by user rating weight
        w = max(0, game_weights[i]['weight'])
        weighted_row = row * w
        weighted_profile_rows.append(weighted_row)
        total_profile_weight += w
    
    if total_profile_weight > 0:
        user_profile_vector = np.sum(weighted_profile_rows, axis=0).reshape(1, -1)
        # Normalize sum by weight instead of simple mean to preserve stronger signals
        user_profile_vector = user_profile_vector / total_profile_weight
    else:
        user_profile_vector = np.mean(weighted_profile_rows, axis=0).reshape(1, -1)
    
    # Calculate similarity
    check_stop()
    similarities = cosine_similarity(user_profile_vector, candidate_matrix)[0]
    
    # Apply a small boost for games the user has already played (replay value)
    # Replay value is high if the user rated it high but hasn't finished it
    # Map back to candidates
    for i, g in enumerate(backlog + finished_candidates):
        score = float(similarities[i] * 100)
        
        # Boost for highly rated unfinished games (replays)
        if g['rating'] >= 7 and not g['finished']:
            score += 5.0
            
        g['match_score'] = score

    # Sort backlog by match score
    backlog.sort(key=lambda x: x['match_score'], reverse=True)
    df_backlog = pd.DataFrame(backlog)
    df_finished = pd.DataFrame(finished_candidates)

    return df_backlog, df_finished, rated_db_games, vectorizer, tfidf_matrix, rated_count, meta_tags


def build_recommendations_html(conn, show_finished=False, stop_event=None):
    """Complete pipeline to generate recommendation HTML."""
    try:
        df_backlog, df_finished, rated_db_games, vectorizer, tfidf_matrix, rated_count, meta_tags = generate_recommendations(conn, stop_event=stop_event)
    except InterruptedError:
        return None
    
    if df_backlog is None:
        return "<h2>Please rate some games first!</h2>"

    # Persistent columns
    html, shown_appids = build_persistent_sections(
        df_backlog, df_finished, rated_db_games, vectorizer, tfidf_matrix, rated_count, 
        show_finished=show_finished
    )

    # Clustering for remaining games
    remaining = df_backlog[~df_backlog['appid'].isin(shown_appids)].copy()
    num_categories = int(get_metadata('NUM_CATEGORIES', NUM_CATEGORIES))
    
    if not remaining.empty:
        # If we have fewer games than requested categories, reduce category count
        actual_num_clusters = max(1, min(num_categories, len(remaining)))
        
        # Re-vectorize remaining to get better clusters
        if stop_event and stop_event.is_set(): return None
        rem_tags = [" ".join(extract_tags(t)) for t in remaining['tags']]
        rem_tfidf = vectorizer.transform(rem_tags)
        
        if stop_event and stop_event.is_set(): return None
        kmeans = KMeans(n_clusters=actual_num_clusters, n_init=10, random_state=42)
        remaining['cluster'] = kmeans.fit_predict(rem_tfidf)
        
        # Get top terms for each cluster to name them
        feature_names = vectorizer.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_
        
        for i in range(actual_num_clusters):
            cluster_data = remaining[remaining['cluster'] == i].head(int(get_metadata('GAMES_PER_CATEGORY', GAMES_PER_CATEGORY)))
            if cluster_data.empty:
                continue
            
            # Name the cluster based on top TF-IDF terms (excluding meta_tags)
            top_indices = cluster_centers[i].argsort()[::-1]
            top_terms = []
            for idx in top_indices:
                term = feature_names[idx]
                if term not in meta_tags and term not in ['game', 'games', 'indie', 'singleplayer']:
                    top_terms.append(term.replace('_', ' ').title())
                if len(top_terms) >= 2:
                    break
            
            title = " & ".join(top_terms) if top_terms else f"Collection {i+1}"
            
            html += f'<div class="column"><div class="col-title">{title}</div>'
            for _, r in cluster_data.iterrows():
                html += render_game_card(r.to_dict(), rated_db_games, vectorizer, tfidf_matrix, rated_count)
            html += '</div>'

    return html


def build_explanation(candidate, rated_db_games, vectorizer, tfidf_matrix, rated_start_idx):
    """
    Build a human-readable explanation for why a game was recommended.
    Returns a list of short reason strings.
    """
    reasons = []

    cand_tags = extract_tags(candidate.get('tags', ''))
    steam_score = candidate.get('steam_score')
    difficulty = candidate.get('difficulty', 'Easy')
    rating = candidate.get('rating', 0)
    finished = candidate.get('finished', 0)
    playtime = candidate.get('playtime', 0)

    # 1. Similar to games you liked
    similar_to = []
    for rg in rated_db_games:
        if rg['rating'] < 5:
            continue
        rg_tags = extract_tags(rg.get('tags', ''))
        overlap = cand_tags & rg_tags
        if len(overlap) >= 2:
            similar_to.append((rg['name'], rg['rating'], len(overlap)))

    similar_to.sort(key=lambda x: (x[2], x[1]), reverse=True)
    if similar_to:
        top = similar_to[:2]
        names = " & ".join(f"<b>{g[0]}</b>" for g in top)
        reasons.append(f"Similar to {names}")

    # 2. Steam score signal
    if steam_score is not None:
        if steam_score >= 8.5:
            reasons.append(f"Highly rated on Steam ({steam_score:.1f}/10)")
        elif steam_score >= 7.0:
            reasons.append(f"Well received on Steam ({steam_score:.1f}/10)")
        elif steam_score < 5.0:
            reasons.append(f"Mixed reviews ({steam_score:.1f}/10)")

    # 3. Challenge / difficulty signal
    if difficulty and difficulty != 'Easy':
        reasons.append(f"Challenging: {difficulty}")

    # 4. You've played it before but never finished
    if rating > 0 and not finished:
        now = int(time.time())
        if candidate.get('temp_rating') and candidate.get('temp_rating_until', 0) > now:
            reasons.append("Marked as <b>Up Next</b>")
        else:
            reasons.append(f"You played this (rated {rating}/10) but never finished it")

    # 5. Long playtime already invested
    if playtime > 600 and not finished:
        hours = round(playtime / 60, 1)
        reasons.append(f"You have {hours}h in this - worth finishing?")

    # 6. Tag highlights (top shared tags with your profile)
    liked_tags = {}
    for rg in rated_db_games:
        if rg['rating'] >= 7:
            rg_tag_counts = extract_tags_with_counts(rg.get('tags', ''))
            for t, count in rg_tag_counts.items():
                # Weighted contribution to profile
                liked_tags[t] = liked_tags.get(t, 0) + (count / 100.0) # Assume 100 is a "meaningful" vote count

    highlight_tags = [t for t in cand_tags if liked_tags.get(t, 0) >= 0.5]
    
    highlight_tags.sort(key=lambda t: liked_tags.get(t, 0), reverse=True)
    if highlight_tags:
        # Filter out common/uninteresting tags if needed, but for now just show top ones
        tag_str = ", ".join(t.replace("_", " ").title() for t in highlight_tags[:5])
        reasons.append(f"Matches your taste in: {tag_str}")

    if not reasons:
        reasons.append("Matches your overall taste profile")

    return reasons


def render_game_card(r, rated_db_games, vectorizer, tfidf, rated_start_idx):
    """Render a single game card HTML (used by both persistent and dynamic columns)."""
    replay_flag = " (replay)" if (r['rating'] > 0 and not r['finished']) else ""
    reasons = build_explanation(r, rated_db_games, vectorizer, tfidf, rated_start_idx)
    why_html = "".join(f"<span>{reason}</span>" for reason in reasons)

    # Determine which finish button to show
    if not r['finished']:
        finish_btn = f'<button class="icon-btn btn-finish" title="Finish" onclick="finishGame({r["appid"]}, this)">Done</button>'
    else:
        finish_btn = f'<button class="icon-btn btn-unfinish" title="Unfinish" onclick="unfinishGame({r["appid"]}, this)">Revive</button>'

    up_next_badge = ""
    now = int(time.time())
    if r.get('temp_rating') and r.get('temp_rating_until', 0) > now:
        up_next_badge = '<span class="up-next-badge" title="High priority for recommendations">Up Next</span>'

    rating_val = int(r['rating'] or 0)
    return f'''
        <div class="game-card" data-appid="{r['appid']}">
            <div class="btn-group">
                {finish_btn}
                <button class="icon-btn btn-up-next" title="Up Next" onclick="updateGame({r['appid']}, 'up_next', this)">Next</button>
                <button class="icon-btn btn-ignore" title="Ignore" onclick="updateGame({r['appid']}, 'ignore', this)">Ignore</button>
                <button class="icon-btn btn-ban" title="Ban" onclick="updateGame({r['appid']}, 'ban', this)">Ban</button>
                <a href="https://store.steampowered.com/app/{r['appid']}" target="_blank" class="icon-btn btn-steam" title="Steam Page" style="text-decoration: none; text-align: center;">Steam</a>
            </div>
            <img src="https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{r['appid']}/header.jpg">
            <div style="padding: 2px;">
                <div style="margin-bottom: 5px; min-height: 2.2em; display: flex; align-items: flex-start;">
                    <b style="color: white; font-size: 0.85em; line-height: 1.2;">{r['name']}{replay_flag}</b>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="match-score" style="font-size: 0.75em; opacity: 0.8;">
                        {round(r['match_score'], 1)}% Match
                    </span>
                    {up_next_badge}
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 6px; border-radius: 6px; margin-bottom: 8px;">
                    <div style="display:flex; align-items:center; gap:6px;">
                        <input type="range" min="0" max="10" value="{rating_val}"
                               style="flex:1; accent-color:var(--accent); cursor: pointer; height: 4px;"
                               autocomplete="off"
                               onchange="rateCard({r['appid']}, this)">
                        <span style="font-weight:800; color:var(--accent); min-width:14px; font-size: 0.8em;">{rating_val}</span>
                    </div>
                </div>
                <button class="why-toggle" style="font-size: 0.7em; padding: 4px 8px;" onclick="toggleWhy(this)">Why?</button>
                <div class="why-box">{why_html}</div>
            </div>
        </div>'''


def build_persistent_sections(df_backlog, df_finished, rated_db_games, vectorizer, tfidf, rated_start_idx, show_finished=False):
    """
    Build the HTML for the always-present categories:
    Top Games, Hard Games, Chill Games, Recently Played Unfinished, Finished Games, Forgotten Games.
    Returns (html_string, set_of_appids_already_shown).
    """
    sections = []
    shown = set()
    now = time.time()

    def make_column(title, rows):
        if rows.empty:
            return ""
        html = f'<div class="column"><div class="col-title">{title}</div>'
        for _, r in rows.iterrows():
            html += render_game_card(r.to_dict(), rated_db_games, vectorizer, tfidf, rated_start_idx)
            shown.add(int(r['appid']))
        html += '</div>'
        return html

    # Top Games - best overall matches
    games_per_cat = int(get_metadata('GAMES_PER_CATEGORY', GAMES_PER_CATEGORY))
    top_rows = df_backlog.head(games_per_cat)
    sections.append(make_column("Top Games", top_rows))

    # Hard Games - anything not 'Easy' difficulty
    hard_mask = df_backlog['difficulty'].fillna('Easy').str.lower() != 'easy'
    hard_rows = df_backlog[hard_mask].head(games_per_cat)
    sections.append(make_column("Hard Games", hard_rows))

    # Chill Games - tag-based
    def is_chill(tag_str):
        if not tag_str:
            return False
        tokens = extract_tags(tag_str)
        return bool(tokens & CHILL_TAGS)

    chill_mask = df_backlog['tags'].apply(is_chill)
    chill_rows = df_backlog[chill_mask].head(games_per_cat)
    sections.append(make_column("Chill Games", chill_rows))

    # Recently Played - unfinished games played within last 30 days
    recent_mask = (df_backlog['finished'] == 0) & (df_backlog['last_played'] > now - 30 * 24 * 3600)
    recent_rows = df_backlog[recent_mask].sort_values('last_played', ascending=False).head(games_per_cat)
    sections.append(make_column("Recently Played", recent_rows))

    # Finished Games - to allow marking as unfinished
    # These were separated in app.py to prevent them from appearing in other columns
    if show_finished:
        finished_rows = df_finished.sort_values('last_played', ascending=False).head(games_per_cat)
        sections.append(make_column("Finished Games", finished_rows))

    # Forgotten Games - unfinished games not played in over 1 year
    forgotten_mask = (df_backlog['finished'] == 0) & (df_backlog['last_played'] < now - 365 * 24 * 3600)
    forgotten_rows = df_backlog[forgotten_mask].sort_values('last_played', ascending=True).head(games_per_cat)
    sections.append(make_column("Forgotten Games", forgotten_rows))

    return "".join(sections), shown
