import pandas as pd
import re

def load_titles(file_path):
    """Load titles from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} titles from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def assign_genre(title):
    """Assign a genre based on keywords in the title."""
    title_lower = title.lower()
    
    # Define genre keywords
    genre_keywords = {
        "Fantasy": ["fantasy", "dragon", "magic", "wizard", "elf", "fairy", "myth", "legend", "mythical", "kingdom"],
        "Science Fiction": ["sci-fi", "science fiction", "space", "alien", "robot", "future", "galaxy", "starship", "dystopian"],
        "Mystery": ["mystery", "detective", "crime", "thriller", "suspense", "murder", "clue", "investigation"],
        "Romance": ["romance", "love", "passion", "heart", "marriage", "wedding", "relationship", "desire"],
        "Horror": ["horror", "scary", "ghost", "monster", "fear", "terror", "nightmare", "haunted", "creepy"],
        "Biography": ["biography", "memoir", "autobiography", "life", "journal", "diary", "true story"],
        "History": ["history", "historical", "ancient", "medieval", "century", "empire", "civilization", "dynasty"],
        "Self-Help": ["self-help", "motivational", "inspiration", "success", "happiness", "habit", "productivity"],
        "Adventure": ["adventure", "journey", "quest", "expedition", "explore", "discovery", "survival"],
        "Young Adult": ["young adult", "teen", "coming of age", "high school", "adolescent", "youth"],
        "Children": ["children", "kid", "picture book", "illustrated", "bedtime", "fairy tale"],
        "Poetry": ["poetry", "poem", "verse", "rhyme", "sonnet", "lyric", "stanza"],
        "Philosophy": ["philosophy", "philosophical", "ethics", "metaphysics", "logic", "consciousness"],
        "Religion": ["religion", "spiritual", "faith", "god", "bible", "church", "divine", "prayer"],
        "Cooking": ["cooking", "recipe", "food", "baking", "cuisine", "cookbook", "culinary", "kitchen"],
        "Art": ["art", "painting", "drawing", "photography", "sculpture", "artistic", "museum"],
        "Business": ["business", "finance", "management", "entrepreneur", "investing", "leadership", "economics"],
        "Science": ["science", "scientific", "biology", "chemistry", "physics", "research", "experiment"],
        "Travel": ["travel", "journey", "guide", "destination", "tour", "expedition", "explore"],
        "Health": ["health", "medical", "fitness", "diet", "nutrition", "exercise", "wellness"],
        "Thriller": ["thriller", "suspense", "action", "psychological", "conspiracy", "espionage", "spy"],
        "Drama": ["drama", "emotional", "relationship", "family", "conflict", "tragedy", "theatrical"],
        "Western": ["western", "cowboy", "ranch", "frontier", "wilderness", "wild west"],
        "Political": ["political", "politics", "government", "election", "democracy", "leader", "campaign"]
    }
    
    # Check for matches with genre keywords
    matched_genres = []
    for genre, keywords in genre_keywords.items():
        for keyword in keywords:
            if keyword in title_lower:
                matched_genres.append(genre)
                break
    
    # If multiple genres match, return all of them
    if matched_genres:
        return ", ".join(matched_genres)
    
    # Use simple heuristics for some common patterns
    if "how to" in title_lower or "guide" in title_lower:
        return "Self-Help"
    if "novel" in title_lower:
        return "Fiction"
    
    # If no genre matches, assign a default
    return "Fiction"

def main():
    # File paths
    input_path = 'unique_titles.csv'
    output_path = 'titles_with_genres.csv'
    
    # Load the titles
    titles_df = load_titles(input_path)
    if titles_df is None:
        return
    
    # Process titles and assign genres
    print("Assigning genres to titles...")
    titles_df['genre'] = titles_df['normalized_title'].apply(assign_genre)
    
    # Display a sample of the results
    print("\nSample of titles with assigned genres:")
    print(titles_df.head(10))
    
    # Report genre distribution
    genre_counts = titles_df['genre'].value_counts()
    print("\nGenre distribution (top 10):")
    print(genre_counts.head(10))
    
    # Save the results
    try:
        titles_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved {len(titles_df)} records to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    print("\nProcess completed successfully.")

if __name__ == "__main__":
    main()