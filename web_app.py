import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
from sift.engine import SiftRecommender

# Initialize engine globally for instant API responses
engine = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sift — Multi-Stage Book Recommendation System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #243247;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.25);
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #334155;
            --radius-lg: 16px;
            --radius-md: 10px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            padding-bottom: 60px;
        }

        header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-bottom: 1px solid var(--border);
            padding: 32px 24px;
            text-align: center;
        }
        .header-badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 999px;
            padding: 4px 14px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        h1 { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 8px; }
        .subtitle { color: var(--text-secondary); max-width: 650px; margin: 0 auto; font-size: 0.98rem; }

        .container {
            max-width: 1180px;
            margin: 32px auto;
            padding: 0 20px;
        }

        /* 3-Stage Pipeline Ribbon */
        .pipeline-ribbon {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }
        .stage-pill {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .stage-num {
            background: var(--accent);
            color: #0f172a;
            font-weight: 800;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            flex-shrink: 0;
        }
        .stage-title { font-weight: 700; font-size: 0.92rem; }
        .stage-desc { font-size: 0.78rem; color: var(--text-secondary); }

        /* Search & Controls */
        .control-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-bottom: 32px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1fr auto;
            gap: 16px;
            align-items: end;
        }
        @media (max-width: 900px) {
            .form-grid { grid-template-columns: 1fr; }
        }
        .form-group label {
            display: block;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        input, select {
            width: 100%;
            padding: 12px 14px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: border 0.2s;
        }
        input:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .btn-submit {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: #ffffff;
            border: none;
            border-radius: var(--radius-md);
            padding: 13px 28px;
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .btn-submit:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px var(--accent-glow);
        }

        .quick-picks {
            margin-top: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            font-size: 0.82rem;
            color: var(--text-secondary);
        }
        .chip {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 4px 10px;
            border-radius: 999px;
            cursor: pointer;
            color: var(--text-primary);
            transition: all 0.15s;
        }
        .chip:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* Results Area */
        #results-container { display: none; }
        .meta-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px 24px;
            margin-bottom: 24px;
        }
        .meta-stat { display: flex; align-items: center; gap: 8px; font-size: 0.88rem; }
        .stat-value { font-weight: 700; color: var(--accent); }

        .recommendations-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
        }
        .book-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 20px;
            display: flex;
            flex-direction: column;
            position: relative;
            transition: transform 0.2s, border-color 0.2s;
        }
        .book-card:hover {
            transform: translateY(-4px);
            border-color: var(--accent);
        }
        .rank-badge {
            position: absolute;
            top: 14px;
            left: 14px;
            background: #0284c7;
            color: #fff;
            font-weight: 800;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        .book-cover {
            width: 110px;
            height: 160px;
            object-fit: cover;
            border-radius: 8px;
            margin: 0 auto 16px auto;
            background: #1e293b;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            border: 1px solid var(--border);
        }
        .book-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 4px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            min-height: 2.6rem;
        }
        .book-author {
            color: var(--text-secondary);
            font-size: 0.86rem;
            margin-bottom: 12px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-pill-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: auto;
            padding-top: 14px;
            border-top: 1px solid var(--border);
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.84rem;
        }
        .metric-val { font-weight: 700; }
        .pred-score { color: #38bdf8; font-size: 1.05rem; font-weight: 800; }
        .prob-bar-container {
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 4px;
        }
        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #38bdf8 100%);
            border-radius: 999px;
        }

        .tag-pill {
            display: inline-block;
            background: rgba(255,255,255,0.05);
            color: var(--text-secondary);
            border-radius: 4px;
            font-size: 0.72rem;
            padding: 2px 6px;
            margin-top: 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
        }

        /* Loading Spinner */
        #loader {
            display: none;
            text-align: center;
            padding: 60px 0;
        }
        .spinner {
            width: 44px;
            height: 44px;
            border: 4px solid var(--border);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 16px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div class="header-badge">Google Deepmind Pair Programming</div>
        <h1>Sift Book Recommender</h1>
        <p class="subtitle">A full 3-Stage collaborative & content recommendation funnel powered by TF-IDF tag clustering, preference classification, and rating regression.</p>
    </header>

    <div class="container">
        <!-- 3-Stage Architecture Summary -->
        <div class="pipeline-ribbon">
            <div class="stage-pill">
                <div class="stage-num">1</div>
                <div>
                    <div class="stage-title">Candidate Retrieval (Keya)</div>
                    <div class="stage-desc">TF-IDF & K-Means (20 Clusters) + KNN Cosine Similarity</div>
                </div>
            </div>
            <div class="stage-pill">
                <div class="stage-num">2</div>
                <div>
                    <div class="stage-title">Preference Classifier (Satya)</div>
                    <div class="stage-desc">Calibrated Random Forest: P(Liked &ge; 4★)</div>
                </div>
            </div>
            <div class="stage-pill">
                <div class="stage-num">3</div>
                <div>
                    <div class="stage-title">Rating Ranking (Sneha)</div>
                    <div class="stage-desc">Regularized Regressor: Predicts exact 1.0–5.0★ rating</div>
                </div>
            </div>
        </div>

        <!-- Controls Form -->
        <div class="control-panel">
            <form id="rec-form" class="form-grid" onsubmit="event.preventDefault(); fetchRecommendations();">
                <div class="form-group">
                    <label>Seed Book Title or Author</label>
                    <input type="text" id="book-query" value="The Hobbit" placeholder="e.g. The Hunger Games, 1984..." required>
                </div>
                <div class="form-group">
                    <label>User ID</label>
                    <input type="number" id="user-id" value="1" min="1" max="60000">
                </div>
                <div class="form-group">
                    <label>Top Results</label>
                    <select id="top-n">
                        <option value="5" selected>Top 5 Books</option>
                        <option value="8">Top 8 Books</option>
                        <option value="12">Top 12 Books</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Stage 2 Min Likelihood</label>
                    <select id="min-prob">
                        <option value="0.30">30% (Permissive)</option>
                        <option value="0.40" selected>40% (Balanced)</option>
                        <option value="0.60">60% (Strict)</option>
                    </select>
                </div>
                <button type="submit" class="btn-submit">
                    <span>Explore</span> &rarr;
                </button>
            </form>

            <div class="quick-picks">
                <span>Quick Picks:</span>
                <span class="chip" onclick="setQuery('The Hobbit', 1)">The Hobbit</span>
                <span class="chip" onclick="setQuery('Harry Potter', 42)">Harry Potter</span>
                <span class="chip" onclick="setQuery('Pride and Prejudice', 100)">Pride and Prejudice</span>
                <span class="chip" onclick="setQuery('The Hunger Games', 15)">The Hunger Games</span>
                <span class="chip" onclick="setQuery('To Kill a Mockingbird', 999)">Mockingbird (Cold-Start)</span>
            </div>
        </div>

        <!-- Loading State -->
        <div id="loader">
            <div class="spinner"></div>
            <p style="color: var(--text-secondary);">Traversing 3-Stage Recommender Funnel...</p>
        </div>

        <!-- Results Section -->
        <div id="results-container">
            <div class="meta-bar">
                <div class="meta-stat">
                    <span>Seed Book:</span>
                    <span id="meta-seed" class="stat-value">-</span>
                </div>
                <div class="meta-stat">
                    <span>User Profile:</span>
                    <span id="meta-user" class="stat-value">-</span>
                </div>
                <div class="meta-stat">
                    <span>Funnel Reduction:</span>
                    <span id="meta-funnel" class="stat-value">-</span>
                </div>
            </div>

            <div class="recommendations-grid" id="recs-grid">
                <!-- Dynamically rendered book cards -->
            </div>
        </div>
    </div>

    <script>
        function setQuery(title, userId) {
            document.getElementById('book-query').value = title;
            document.getElementById('user-id').value = userId;
            fetchRecommendations();
        }

        async function fetchRecommendations() {
            const query = document.getElementById('book-query').value.trim();
            const userId = document.getElementById('user-id').value.trim() || 1;
            const topN = document.getElementById('top-n').value;
            const minProb = document.getElementById('min-prob').value;

            if (!query) return;

            document.getElementById('loader').style.display = 'block';
            document.getElementById('results-container').style.display = 'none';

            try {
                const url = `/api/recommend?book=${encodeURIComponent(query)}&user_id=${userId}&top_n=${topN}&min_prob=${minProb}`;
                const resp = await fetch(url);
                const data = await resp.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    document.getElementById('loader').style.display = 'none';
                    return;
                }

                // Render Metadata Bar
                const seed = data.seed_book;
                document.getElementById('meta-seed').innerText = `${seed.title} (${seed.average_rating}★)`;
                const user = data.user_profile;
                document.getElementById('meta-user').innerText = user.is_cold_start ? 
                    `Cold-Start User (New)` : 
                    `User #${data.user_id} (${user.user_rating_count} reads, ${user.user_mean_rating.toFixed(2)}★ avg)`;
                
                const funnel = data.funnel_stats;
                document.getElementById('meta-funnel').innerText = 
                    `${funnel.stage1_candidates} Candidates -> ${funnel.stage2_survivors} Classified -> ${funnel.stage3_final_ranked} Ranked`;

                // Render Cards
                const grid = document.getElementById('recs-grid');
                grid.innerHTML = '';

                data.recommendations.forEach(r => {
                    const fallbackImg = 'https://images.gr-assets.com/books/1447303603m/2767052.jpg';
                    const imgUrl = (r.image_url && r.image_url.startsWith('http')) ? r.image_url : fallbackImg;

                    const card = document.createElement('div');
                    card.className = 'book-card';
                    card.innerHTML = `
                        <div class="rank-badge">${r.rank}</div>
                        <img class="book-cover" src="${imgUrl}" alt="${r.title}" onerror="this.src='${fallbackImg}'">
                        <div class="book-title" title="${r.title}">${r.title}</div>
                        <div class="book-author">${r.authors}</div>
                        <span class="tag-pill">${r.top_tags || 'fiction, literature'}</span>
                        <div class="metric-pill-group">
                            <div class="metric-row">
                                <span style="color: var(--text-secondary);">Stage 2 Likelihood:</span>
                                <span class="metric-val" style="color: #10b981;">${r.like_probability}%</span>
                            </div>
                            <div class="prob-bar-container">
                                <div class="prob-bar" style="width: ${r.like_probability}%;"></div>
                            </div>
                            <div class="metric-row" style="margin-top: 6px;">
                                <span style="color: var(--text-secondary);">Stage 3 Pred Rating:</span>
                                <span class="pred-score">${r.predicted_rating.toFixed(2)} ★</span>
                            </div>
                            <div class="metric-row">
                                <span style="color: var(--text-secondary); font-size: 0.76rem;">Goodreads Global:</span>
                                <span style="font-size: 0.78rem; color: #94a3b8;">${r.average_rating}★</span>
                            </div>
                        </div>
                    `;
                    grid.appendChild(card);
                });

                document.getElementById('loader').style.display = 'none';
                document.getElementById('results-container').style.display = 'block';
            } catch (err) {
                alert('Request failed: ' + err);
                document.getElementById('loader').style.display = 'none';
            }
        }

        // Run default search on load
        window.addEventListener('DOMContentLoaded', () => {
            fetchRecommendations();
        });
    </script>
</body>
</html>
"""


class SiftRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean logging
        return

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif path == "/api/recommend":
            params = urllib.parse.parse_qs(parsed.query)
            book_query = params.get("book", ["The Hobbit"])[0]
            user_id = int(params.get("user_id", ["1"])[0])
            top_n = int(params.get("top_n", ["5"])[0])
            min_prob = float(params.get("min_prob", ["0.40"])[0])

            try:
                res = engine.recommend(
                    user_id=user_id,
                    seed_title=book_query,
                    top_n=top_n,
                    min_like_prob=min_prob
                )
                response_json = json.dumps(res)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(response_json.encode("utf-8"))
            except Exception as e:
                err_json = json.dumps({"error": str(e)})
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err_json.encode("utf-8"))
            return

        else:
            self.send_response(404)
            self.end_headers()


def run_web_dashboard(port: int = 8501, auto_open: bool = True):
    global engine
    if engine is None:
        engine = SiftRecommender()

    server = HTTPServer(("127.0.0.1", port), SiftRequestHandler)
    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f"  Sift Web Dashboard is live at: {url}")
    print(f"  Press Ctrl+C in this terminal to stop.")
    print(f"=======================================================\n")
    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Sift Web Dashboard.")
        server.server_close()


if __name__ == "__main__":
    run_web_dashboard(port=8501)
