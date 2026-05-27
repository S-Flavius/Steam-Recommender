import time
from database import get_metadata, Game
from config import CAROUSEL_SIZE
from sqlmodel import select, func, or_

def get_unrated_count(session):
    """Get the total count of unrated games that are not ignored."""
    now = int(time.time())
    statement = select(func.count(Game.appid)).where(
        Game.rating == 0,
        Game.ignored == False,
        or_(Game.ignore_until == 0, Game.ignore_until < now),
        or_(Game.temp_rating == None, Game.temp_rating_until < now)
    )
    return session.exec(statement).one()


def get_carousel_html(session):
    """Build HTML for the rating carousel of unrated games."""
    carousel_size_meta = get_metadata('CAROUSEL_SIZE', CAROUSEL_SIZE)
    carousel_size = int(carousel_size_meta)
    
    now = int(time.time())
    statement = select(Game).where(
        Game.rating == 0,
        Game.ignored == False,
        or_(Game.ignore_until == 0, Game.ignore_until < now),
        or_(Game.temp_rating == None, Game.temp_rating_until < now)
    ).order_by(Game.finished.desc(), Game.playtime.desc()).limit(carousel_size)
    
    games = session.exec(statement).all()

    html_parts = []
    for g_obj in games:
        g = g_obj.model_dump()
        rating = g['rating'] or 0
        flag = " (Finished)" if g['finished'] else ""
        
        up_next_badge = ""
        if g['temp_rating'] and (g['temp_rating_until'] or 0) > now:
             up_next_badge = '<span class="up-next-badge" style="margin-left: 5px; vertical-align: middle;">Next</span>'

        # Determine which finish button to show
        if not g['finished']:
            finish_btn = f'<button class="icon-btn btn-finish" title="Finish" onclick="finishGame({g["appid"]}, this)">Done</button>'
        else:
            finish_btn = f'<button class="icon-btn btn-unfinish" title="Unfinish" onclick="unfinishGame({g["appid"]}, this)">Revive</button>'

        part = f'''
        <div class="rate-card" data-appid="{g['appid']}" x-data="{{ open: false }}">
            <img src="https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{g['appid']}/header.jpg">
            <div class="game-title-container">
                <div class="game-title">{g['name']}{flag}{up_next_badge}</div>
            </div>
            <div class="slider-container">
                <div class="slider-wrapper">
                    <input type="range" class="rate-slider" data-appid="{g['appid']}" min="0" max="10" value="{rating}" 
                           autocomplete="off"
                           oninput="this.nextElementSibling.innerText = this.value"
                           onchange="rateCard({g['appid']}, this)">
                    <span class="rating-value">{rating}</span>
                </div>
            </div>
            <div class="btn-group">
                {finish_btn}
                <button class="icon-btn btn-up-next" title="Up Next" onclick="updateGame({g['appid']}, 'up_next', this)">Next</button>
                <button class="icon-btn btn-ignore" title="Ignore" onclick="updateGame({g['appid']}, 'ignore', this)">Ignore</button>
                <button class="icon-btn btn-ban" title="Ban" onclick="updateGame({g['appid']}, 'ban', this)">Ban</button>
                <a href="https://store.steampowered.com/app/{g['appid']}" target="_blank" class="icon-btn btn-steam" title="Steam Page">Steam</a>
            </div>
        </div>'''
        html_parts.append(part)
    return "".join(html_parts)
