import time
from database import get_metadata
from config import CAROUSEL_SIZE

def get_unrated_count(conn):
    """Get the total count of unrated games that are not ignored."""
    now = int(time.time())
    row = conn.execute("""
        SELECT COUNT(*) as count
        FROM games
        WHERE rating = 0
          AND ignored = 0
          AND (ignore_until = 0 OR ignore_until < ?)
          AND (temp_rating IS NULL OR temp_rating_until < ?)
    """, (now, now)).fetchone()
    return row['count'] if row else 0


def get_carousel_html(conn):
    """Build HTML for the rating carousel of unrated games."""
    carousel_size_meta = get_metadata('CAROUSEL_SIZE', CAROUSEL_SIZE)
    carousel_size = int(carousel_size_meta)
    
    now = int(time.time())
    games = conn.execute("""
                         SELECT *
                         FROM games
                         WHERE rating = 0
                           AND ignored = 0
                           AND (ignore_until = 0 OR ignore_until < ?)
                           AND (temp_rating IS NULL OR temp_rating_until < ?)
                         ORDER BY CASE WHEN finished = 1 THEN 0 ELSE 1 END,
                                  playtime DESC
                         LIMIT ?
                         """, (now, now, carousel_size)).fetchall()

    html_parts = []
    for g in games:
        rating = g['rating'] or 0
        flag = " (Finished)" if g['finished'] else ""
        
        # Determine which finish button to show
        if not g['finished']:
            finish_btn = f'<button class="icon-btn btn-finish" title="Finish" onclick="finishGame({g["appid"]}, this)">Done</button>'
        else:
            finish_btn = f'<button class="icon-btn btn-unfinish" title="Unfinish" onclick="unfinishGame({g["appid"]}, this)">Revive</button>'

        part = f'''
        <div class="rate-card" data-appid="{g['appid']}">
            <div class="btn-group">
                {finish_btn}
                <button class="icon-btn btn-up-next" title="Up Next" onclick="updateGame({g['appid']}, 'up_next', this)">Next</button>
                <button class="icon-btn btn-ignore" title="Ignore" onclick="updateGame({g['appid']}, 'ignore', this)">Ignore</button>
                <button class="icon-btn btn-ban" title="Ban" onclick="updateGame({g['appid']}, 'ban', this)">Ban</button>
                <a href="https://store.steampowered.com/app/{g['appid']}" target="_blank" class="icon-btn btn-steam" title="Steam Page" style="text-decoration: none; text-align: center;">Steam</a>
            </div>
            <img src="https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{g['appid']}/header.jpg">
            <div style="margin-bottom: 5px; min-height: 2.2em; display: flex; align-items: flex-start; justify-content: center;">
                <b style="color: white; font-size: 0.85em; line-height: 1.2;">{g['name']}{flag}</b>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 6px; border-radius: 6px;">
                <div style="display:flex; align-items:center; gap:6px;">
                    <input type="range" class="rate-slider" data-appid="{g['appid']}" min="0" max="10" value="{rating}" 
                           style="flex:1; accent-color:var(--accent); cursor: pointer; height: 4px;"
                           oninput="this.nextElementSibling.innerText = this.value">
                    <span style="font-weight: 800; color: var(--accent); min-width: 14px; font-size: 0.8em;">{rating}</span>
                </div>
            </div>
        </div>'''
        html_parts.append(part)
    return "".join(html_parts)
