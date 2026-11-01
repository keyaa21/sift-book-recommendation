import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import webbrowser
from sift.engine import SiftRecommender


def print_banner():
    banner = """
======================================================================
  SIFT: Multi-Stage Hybrid Book Recommendation System
  Stage 1: Content & Tag Retrieval (Keya)
  Stage 2: Preference Classification (Satya)
  Stage 3: Rating Regression & Ranking (Sneha)
======================================================================
"""
    print(banner)


def format_table(results: dict):
    seed = results["seed_book"]
    user = results["user_profile"]
    funnel = results["funnel_stats"]
    recs = results["recommendations"]

    print("-" * 75)
    print(f"📖 SEED BOOK: {seed['title']} (Cluster {seed['cluster_id']})")
    print(f"   Author(s): {seed['authors']} | Goodreads Rating: {seed['average_rating']}★")
    if user['is_cold_start']:
        user_info = "Cold-Start User"
    else:
        user_info = f"Rated {user['user_rating_count']} books, Avg: {user['user_mean_rating']:.2f}★"

    print(f"👤 USER ID: {results['user_id']} ({user_info})")
    print(f"⚙️  FUNNEL: {funnel['stage1_candidates']} Candidates (Stage 1) "
          f"-> {funnel['stage2_survivors']} Passed Classifier (Stage 2) "
          f"-> Top {funnel['stage3_final_ranked']} Ranked (Stage 3)")
    print("-" * 75)
    print(f"{'#':<3} {'Title':<38} {'Authors':<18} {'P(Like)':<9} {'Pred ★':<7}")
    print("-" * 75)

    for r in recs:
        title = (r['title'][:35] + '...') if len(r['title']) > 35 else r['title']
        authors = (r['authors'][:16] + '..') if len(r['authors']) > 16 else r['authors']
        print(f"{r['rank']:<3} {title:<38} {authors:<18} {r['like_probability']:>5.1f}%   {r['predicted_rating']:>4.2f}★")
    print("-" * 75)


def run_demo(engine: SiftRecommender):
    print("\n>>> Running Automated Demonstration Across Multiple Genres & Users <<<\n")
    demos = [
        {"user_id": 1, "book": "The Hobbit", "top_n": 5},
        {"user_id": 42, "book": "The Hunger Games", "top_n": 5},
        {"user_id": 100, "book": "Pride and Prejudice", "top_n": 5},
        {"user_id": 999999, "book": "Harry Potter and the Sorcerer's Stone", "top_n": 5},  # Cold start
    ]
    for d in demos:
        res = engine.recommend(
            user_id=d["user_id"],
            seed_title=d["book"],
            top_n=d["top_n"]
        )
        format_table(res)
        print()


def run_interactive(engine: SiftRecommender):
    print("\n--- Interactive Recommendation Mode ---")
    print("Type 'exit' or 'q' to quit at any time.\n")

    while True:
        try:
            user_input = input("Enter User ID (default: 1): ").strip()
            if user_input.lower() in ("exit", "q"):
                break
            user_id = int(user_input) if user_input else 1

            book_query = input("Enter Seed Book Title (e.g. 'Hobbit', '1984', 'Twilight'): ").strip()
            if book_query.lower() in ("exit", "q"):
                break
            if not book_query:
                print("Book query cannot be empty. Try again.\n")
                continue

            top_input = input("Number of recommendations (default: 5): ").strip()
            top_n = int(top_input) if top_input else 5

            res = engine.recommend(user_id=user_id, seed_title=book_query, top_n=top_n)
            format_table(res)
            print()
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def start_web_server(port: int = 8501):
    from web_app import run_web_dashboard
    run_web_dashboard(port=port)


def main():
    parser = argparse.ArgumentParser(description="Sift Multi-Stage Book Recommendation System")
    parser.add_argument("--user", type=int, default=1, help="User ID (1 to 53424, or any ID for cold-start)")
    parser.add_argument("--book", type=str, default=None, help="Seed book title to search for")
    parser.add_argument("--book-id", type=int, default=None, help="Seed book ID (1 to 10000)")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations to return (default: 5)")
    parser.add_argument("--min-prob", type=float, default=0.40, help="Stage 2 minimum like probability threshold")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prompt")
    parser.add_argument("--demo", action="store_true", help="Run automated multi-user demonstration")
    parser.add_argument("--web", action="store_true", help="Launch interactive web browser dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Web dashboard port (default: 8501)")

    args = parser.parse_args()

    print_banner()

    if args.web:
        start_web_server(port=args.port)
        return

    engine = SiftRecommender()

    if args.demo:
        run_demo(engine)
    elif args.interactive:
        run_interactive(engine)
    elif args.book or args.book_id:
        res = engine.recommend(
            user_id=args.user,
            seed_title=args.book,
            seed_book_id=args.book_id,
            top_n=args.top,
            min_like_prob=args.min_prob
        )
        format_table(res)
    else:
        # Default behavior if run without arguments: run a quick showcase
        print("Tip: Run with --interactive, --web, or --demo. Running default demo with 'The Hobbit'...\n")
        res = engine.recommend(user_id=args.user, seed_title="The Hobbit", top_n=args.top)
        format_table(res)


if __name__ == "__main__":
    main()
