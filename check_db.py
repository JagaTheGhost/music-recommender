import sqlite3

# Connect to the database
conn = sqlite3.connect('feedback.db')
cursor = conn.cursor()

try:
    # Select everything from the feedback table
    cursor.execute("SELECT * FROM feedback")
    rows = cursor.fetchall()

    print(f"\n✅ Total Feedback Entries: {len(rows)}\n")
    print(f"{'ID':<5} | {'Sentiment':<10} | {'Track Name'}")
    print("-" * 50)

    for row in rows:
        # row structure: (id, track_name, sentiment)
        # Note: Depending on your exact model, sentiment might be index 1 or 2
        # We assume: ID is 0, Track is 1, Sentiment is 2 based on previous code
        entry_id = row[0]
        track = row[1]
        sentiment = "❤️ Like" if row[2] == 1 else "💔 Dislike"
        
        print(f"{entry_id:<5} | {sentiment:<10} | {track}")

except sqlite3.OperationalError:
    print("❌ Error: Could not find table 'feedback'. Run the app once to create it.")

conn.close()