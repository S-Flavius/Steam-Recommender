import time

from config import GAMES_PER_CATEGORY, CHILL_TAGS


def build_explanation(candidate, rated_db_games, vectorizer, tfidf_matrix, rated_start_idx):
    """
    Build a human-readable explanation for why a game was recommended.
    Returns a list of short reason strings.
    """
    reasons = []

    cand_tags = set(candidate.get('tags', '').split())
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
        rg_tags = set(rg.get('tags', '').split())
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
        reasons.append(f"You played this (rated {rating}/10) but never finished it")

    # 5. Long playtime already invested
    if playtime > 600 and not finished:
        hours = round(playtime / 60, 1)
        reasons.append(f"You have {hours}h in this - worth finishing?")

    # 6. Tag highlights (top shared tags with your profile)
    liked_tags = {}
    for rg in rated_db_games:
        if rg['rating'] >= 7:
            for t in rg.get('tags', '').split():
                liked_tags[t] = liked_tags.get(t, 0) + 1

    highlight_tags = [t for t in cand_tags if liked_tags.get(t, 0) >= 2]
    highlight_tags.sort(key=lambda t: liked_tags.get(t, 0), reverse=True)
    if highlight_tags:
        tag_str = ", ".join(t.replace("_", " ").title() for t in highlight_tags[:3])
        reasons.append(f"Matches your taste in: {tag_str}")

    if not reasons:
        reasons.append("Matches your overall taste profile")

    return reasons


def render_game_card(r, rated_db_games, vectorizer, tfidf, rated_start_idx):
    """Render a single game card HTML (used by both persistent and dynamic columns)."""
    replay_flag = " (replay)" if (r.get('rating', 0) > 0 and not r.get('finished', 0)) else ""
    reasons = build_explanation(r, rated_db_games, vectorizer, tfidf, rated_start_idx)
    why_html = "".join(f"<span>{reason}</span>" for reason in reasons)

    # Determine which finish button to show
    if not r.get('finished', 0):
        finish_btn = f'<button class="icon-btn btn-finish" onclick="finishGame({r["appid"]}, this)">Finish</button>'
    else:
        finish_btn = f'<button class="icon-btn btn-unfinish" onclick="updateGame({r["appid"]}, \'unfinish\', this)">Unfinish</button>'

    return f'''
        <div class="game-card">
            <div class="btn-group">
                {finish_btn}
                <button class="icon-btn btn-up-next" onclick="updateGame({r['appid']}, 'up_next', this)">Up Next</button>
                <button class="icon-btn btn-ban" onclick="updateGame({r['appid']}, 'ban', this)">Ban</button>
                <button class="icon-btn btn-ignore" onclick="updateGame({r['appid']}, 'ignore', this)">Ignore</button>
            </div>
            <img src="https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{r['appid']}/header.jpg">
            <div style="padding: 5px;">
                <b>{r['name']}{replay_flag}</b><br>
                <span class="match-score">
                    Match: {round(r['match_score'], 1)}%
                </span>
                <div style="display:flex; align-items:center; gap:6px; margin-top:5px;">
                    <span style="font-size:0.75em; color:#8899aa;">Rating:</span>
                    <input type="range" min="0" max="10" value="{int(r['rating'])}"
                           style="flex:1; accent-color:#66c0f4;"
                           onchange="rateCard({r['appid']}, this)">
                    <span style="font-weight:bold; color:#66c0f4; min-width:14px;">{int(r['rating'])}</span>
                </div>
                <button class="why-toggle" onclick="toggleWhy(this)">Why?</button>
                <div class="why-box">{why_html}</div>
            </div>
        </div>'''


def build_persistent_sections(df_backlog, rated_db_games, vectorizer, tfidf, rated_start_idx):
    """
    Build the HTML for the always-present categories:
    Top Games, Hard Games, Chill Games, Recently Played Unfinished, Forgotten Games.
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
    top_rows = df_backlog.head(GAMES_PER_CATEGORY)
    sections.append(make_column("Top Games", top_rows))

    # Hard Games - anything not 'Easy' difficulty
    hard_mask = df_backlog['difficulty'].fillna('Easy').str.lower() != 'easy'
    hard_rows = df_backlog[hard_mask].head(GAMES_PER_CATEGORY)
    sections.append(make_column("Hard Games", hard_rows))

    # Chill Games - tag-based
    def is_chill(tag_str):
        if not tag_str:
            return False
        tokens = {t.lower() for t in tag_str.split()}
        return bool(tokens & CHILL_TAGS)

    chill_mask = df_backlog['tags'].apply(is_chill)
    chill_rows = df_backlog[chill_mask].head(GAMES_PER_CATEGORY)
    sections.append(make_column("Chill Games", chill_rows))

    # Recently Played Unfinished - unfinished games played within last 30 days
    recent_mask = (df_backlog['finished'] == 0) & (df_backlog['last_played'] > now - 30 * 24 * 3600)
    recent_rows = df_backlog[recent_mask].sort_values('last_played', ascending=False).head(GAMES_PER_CATEGORY)
    sections.append(make_column("Recently Played Unfinished", recent_rows))

    # Forgotten Games - unfinished games not played in over 1 year
    forgotten_mask = (df_backlog['finished'] == 0) & (df_backlog['last_played'] < now - 365 * 24 * 3600)
    forgotten_rows = df_backlog[forgotten_mask].sort_values('last_played', ascending=True).head(GAMES_PER_CATEGORY)
    sections.append(make_column("Forgotten Games", forgotten_rows))

    return "".join(sections), shown
