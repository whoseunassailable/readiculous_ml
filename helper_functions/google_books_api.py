#!/usr/bin/env python3
import csv
import json
import requests
import os
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genre_finder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_BOOKS_API_KEY = os.getenv('GOOGLE_BOOKS_API_KEY')
MAX_REQUESTS = int(os.getenv('MAX_REQUESTS', '1000'))
REQUEST_DELAY = float(os.getenv('REQUEST_DELAY', '0.1'))  # Delay between API requests

# Paths Configuration
INPUT_FILE = os.getenv('INPUT_FILE', 'unique_titles.csv')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'book_genres.csv')
TEMP_FOLDER = os.getenv('TEMP_FOLDER', 'temp')
MODEL_FOLDER = os.getenv('MODEL_FOLDER', 'genre_model')
DATASET_PATH = os.getenv('DATASET_PATH', 'datasets/best_books_prepared.csv')

# Create necessary folders
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Global tracking variables
request_counter = 0

def clean_title(title: str) -> str:
    """Clean and normalize a book title for better matching"""
    if not isinstance(title, str):
        return ""
    
    # Convert to lowercase and strip whitespace
    title = title.lower().strip()
    
    # Remove common subtitle indicators
    for sep in [' - ', ' : ', ' – ', ': ', ' -- ']:
        if sep in title:
            title = title.split(sep)[0].strip()
    
    # Remove series indicators in parentheses
    title = title.split(' (')[0].strip()
    
    # Remove common words that don't help with matching
    common_words = ['a novel', 'novel', 'the novel', 'a memoir', 'memoir', 'book', 
                   'volume', 'edition', 'revised', 'series', 'complete', 'collection']
    for word in common_words:
        title = title.replace(f" {word}", "")
    
    # Remove special characters and extra spaces
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title

def load_book_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the book dataset containing genres"""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file {dataset_path} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} entries")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

def build_title_vectors(dataset_df: pd.DataFrame):
    """Create TF-IDF vectors for the book titles in the dataset"""
    # Make sure clean_title column exists
    if 'clean_title' not in dataset_df.columns:
        dataset_df['clean_title'] = dataset_df['title'].apply(clean_title)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2)
    title_vectors = vectorizer.fit_transform(dataset_df['clean_title'])
    
    return vectorizer, title_vectors

def match_with_dataset(titles: List[str], dataset_df: pd.DataFrame, 
                      vectorizer, title_vectors,
                      threshold: float = 0.65) -> Dict[str, Dict]:
    """Match input titles with books in the dataset based on title similarity"""
    results = {}
    
    # Clean the input titles
    cleaned_titles = [clean_title(title) for title in titles]
    
    # Transform input titles to TF-IDF vectors
    input_vectors = vectorizer.transform(cleaned_titles)
    
    # Calculate similarity
    similarities = cosine_similarity(input_vectors, title_vectors)
    
    # Get matches above threshold
    for i, title in enumerate(titles):
        max_idx = similarities[i].argmax()
        max_sim = similarities[i][max_idx]
        
        result = {
            'original_title': title,
            'matched': False,
            'similarity': max_sim,
            'matched_title': '',
            'genres': []
        }
        
        if max_sim >= threshold:
            result['matched'] = True
            result['matched_title'] = dataset_df.iloc[max_idx]['title']
            
            # Get genres
            genres = dataset_df.iloc[max_idx]['genres']
            if isinstance(genres, str):
                # Handle comma-separated genre strings
                result['genres'] = [g.strip() for g in genres.split(',')]
            elif isinstance(genres, list):
                result['genres'] = genres
            
        results[title] = result
    
    return results

def search_book_genre(title: str) -> Dict[str, Any]:
    """Search for a book title on Google Books API and extract genre information"""
    global request_counter
    
    if not GOOGLE_BOOKS_API_KEY:
        logger.error("No Google Books API key found")
        return {'title': title, 'genres': [], 'error': 'No API key'}
    
    # Increment request counter
    request_counter += 1
    
    try:
        # Encode title for URL
        encoded_title = title.replace(' ', '+')
        
        # Create the API URL
        url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{encoded_title}&key={GOOGLE_BOOKS_API_KEY}"
        
        # Make the API request
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"API Request #{request_counter} for title: \"{title}\"")
        
        # Check if we got any results
        if data.get('totalItems', 0) == 0:
            return {'title': title, 'genres': [], 'status': 'No results found'}
        
        # Get the first book result
        book = data['items'][0]
        
        # Extract volume info
        volume_info = book.get('volumeInfo', {})
        
        # Get categories (genres) if available
        genres = volume_info.get('categories', [])
        
        # Return results
        return {
            'title': title,
            'google_title': volume_info.get('title', ''),
            'genres': genres,
            'authors': volume_info.get('authors', []),
            'publisher': volume_info.get('publisher', ''),
            'published_date': volume_info.get('publishedDate', ''),
            'description': volume_info.get('description', ''),
            'page_count': volume_info.get('pageCount', 0),
            'book_id': book.get('id', ''),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error searching for \"{title}\": {str(e)}")
        return {'title': title, 'genres': [], 'error': str(e)}

def process_titles_with_api(titles: List[str], max_requests: int = MAX_REQUESTS,
                          output_file: str = None) -> Dict[str, Dict]:
    """Process a list of titles using Google Books API with limits"""
    global request_counter
    request_counter = 0  # Reset counter
    
    results = {}
    limit_reached = False
    
    logger.info(f"Starting API processing of {len(titles)} titles (limit: {max_requests})")
    
    # Sequential processing
    for title in titles:
        if request_counter >= max_requests:
            limit_reached = True
            break
        
        result = search_book_genre(title)
        results[title] = result
        
        # Add a small delay between requests
        time.sleep(REQUEST_DELAY)
    
    # Save results if output file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    if limit_reached:
        logger.warning(f"API request limit of {max_requests} reached. Processed {len(results)}/{len(titles)} titles.")
    else:
        logger.info(f"Completed API processing of {len(results)} titles.")
    
    return results, limit_reached

def train_genre_predictor(matched_titles: Dict[str, Dict]) -> Dict:
    """Train a simple genre predictor based on title words"""
    # Extract titles with genres
    train_data = []
    for title_data in matched_titles.values():
        if title_data['matched'] or 'genres' in title_data and title_data['genres']:
            train_data.append({
                'title': title_data['original_title'],
                'genres': title_data['genres'] if isinstance(title_data['genres'], list) else []
            })
    
    if not train_data:
        logger.warning("No training data available for genre predictor")
        return {}
    
    # Create a simple word-to-genre map
    word_genre_map = {}
    genre_counts = {}
    
    for item in train_data:
        title = item['title'].lower()
        genres = item['genres']
        
        # Count genres
        for genre in genres:
            if genre not in genre_counts:
                genre_counts[genre] = 0
            genre_counts[genre] += 1
        
        # Map words to genres
        words = title.split()
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
                
            if word not in word_genre_map:
                word_genre_map[word] = {}
            
            for genre in genres:
                if genre not in word_genre_map[word]:
                    word_genre_map[word][genre] = 0
                word_genre_map[word][genre] += 1
    
    # Normalize the word-genre associations
    for word, genre_map in word_genre_map.items():
        total = sum(genre_map.values())
        for genre in genre_map:
            genre_map[genre] = genre_map[genre] / total
    
    # Save the model
    model = {
        'word_genre_map': word_genre_map,
        'genre_counts': genre_counts,
        'total_books': len(train_data)
    }
    
    with open(os.path.join(MODEL_FOLDER, 'simple_genre_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    return model

def predict_genres(titles: List[str], model: Dict, top_n: int = 3) -> Dict[str, Dict]:
    """Predict genres for titles without matches using the trained model"""
    if not model or not model.get('word_genre_map'):
        logger.error("No valid model for genre prediction")
        return {}
    
    word_genre_map = model['word_genre_map']
    genre_counts = model['genre_counts']
    total_books = model['total_books']
    
    # Calculate prior probabilities for each genre
    genre_priors = {genre: count/total_books for genre, count in genre_counts.items()}
    
    results = {}
    for title in titles:
        title_lower = title.lower()
        words = title_lower.split()
        
        # Score each genre based on words in the title
        genre_scores = {genre: genre_priors[genre] for genre in genre_priors}
        
        for word in words:
            if word in word_genre_map:
                for genre, score in word_genre_map[word].items():
                    if genre in genre_scores:
                        # Boost the score based on word-genre association
                        genre_scores[genre] *= (1 + score)
        
        # Sort genres by score
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        top_genres = [genre for genre, score in sorted_genres[:top_n]]
        
        results[title] = {
            'title': title,
            'predicted_genres': top_genres,
            'source': 'inference'
        }
    
    return results

def process_book_titles(input_file: str, output_file: str, dataset_path: str = None,
                      max_requests: int = MAX_REQUESTS):
    """Process all book titles to get their genres using the multi-phase approach"""
    # Load titles
    titles = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                titles.append(row['normalized_title'])
        
        logger.info(f"Loaded {len(titles)} titles from {input_file}")
    except Exception as e:
        logger.error(f"Error loading titles: {str(e)}")
        return
    
    all_results = {}
    remaining_titles = titles.copy()
    
    # Phase 1: Match with existing dataset if available
    if dataset_path and os.path.exists(dataset_path):
        logger.info(f"Phase 1: Matching with existing dataset {dataset_path}")
        
        dataset_df = load_book_dataset(dataset_path)
        if not dataset_df.empty:            
            # Build title vectors
            vectorizer, title_vectors = build_title_vectors(dataset_df)
            
            # Match in batches to avoid memory issues
            batch_size = 5000
            dataset_matches = {}
            
            for i in range(0, len(titles), batch_size):
                batch_titles = titles[i:i+batch_size]
                batch_results = match_with_dataset(
                    batch_titles, dataset_df, vectorizer, title_vectors
                )
                dataset_matches.update(batch_results)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(titles) + batch_size - 1)//batch_size}")
            
            # Update results and remaining titles
            matched_titles = []
            for title, result in dataset_matches.items():
                if result['matched']:
                    all_results[title] = {
                        'title': title,
                        'genres': result['genres'],
                        'source': 'dataset',
                        'matched_title': result['matched_title'],
                        'similarity': result['similarity']
                    }
                    matched_titles.append(title)
            
            remaining_titles = [t for t in remaining_titles if t not in matched_titles]
            logger.info(f"Matched {len(matched_titles)} titles with dataset. {len(remaining_titles)} remaining.")
    
    # Phase 2: Use Google Books API for some remaining titles
    if remaining_titles and GOOGLE_BOOKS_API_KEY:
        logger.info(f"Phase 2: Using Google Books API (limit: {max_requests} requests)")
        
        # Process a subset of remaining titles with Google Books API
        api_titles = remaining_titles[:max_requests]
        api_results, limit_reached = process_titles_with_api(
            api_titles, 
            max_requests=max_requests,
            output_file=os.path.join(TEMP_FOLDER, 'api_results.json')
        )
        
        # Update results and remaining titles
        processed_titles = []
        for title, result in api_results.items():
            if 'genres' in result and result['genres']:
                all_results[title] = {
                    'title': title,
                    'genres': result['genres'],
                    'source': 'google_api',
                    'google_title': result.get('google_title', ''),
                    'book_id': result.get('book_id', '')
                }
                processed_titles.append(title)
        
        remaining_titles = [t for t in remaining_titles if t not in processed_titles]
        logger.info(f"Processed {len(processed_titles)} titles with API. {len(remaining_titles)} remaining.")
    
    # Phase 3: Use genre inference for remaining titles
    if remaining_titles:
        logger.info("Phase 3: Using genre inference for remaining titles")
        
        # Train a genre predictor using the titles we have genres for
        model = train_genre_predictor(all_results)
        
        if model:
            # Predict genres for remaining titles
            predicted_results = predict_genres(remaining_titles, model)
            
            # Update results
            for title, result in predicted_results.items():
                all_results[title] = {
                    'title': title,
                    'genres': result['predicted_genres'],
                    'source': 'inference'
                }
            
            logger.info(f"Predicted genres for {len(predicted_results)} remaining titles.")
    
    # Save all results
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['title', 'genres', 'source', 'matched_title', 'similarity', 
                     'google_title', 'book_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for title in titles:  # Keep original order
            if title in all_results:
                result = all_results[title]
                if isinstance(result['genres'], list):
                    result['genres'] = ', '.join(result['genres'])
                writer.writerow(result)
            else:
                writer.writerow({'title': title, 'genres': '', 'source': 'unknown'})
    
    logger.info(f"All results saved to {output_file}")
    
    # Generate a summary
    total = len(titles)
    dataset_count = sum(1 for r in all_results.values() if r['source'] == 'dataset')
    api_count = sum(1 for r in all_results.values() if r['source'] == 'google_api')
    inference_count = sum(1 for r in all_results.values() if r['source'] == 'inference')
    unknown_count = total - dataset_count - api_count - inference_count
    
    logger.info("\nSummary:")
    logger.info(f"Total titles: {total}")
    logger.info(f"Titles matched with dataset: {dataset_count} ({dataset_count/total*100:.1f}%)")
    logger.info(f"Titles processed with Google API: {api_count} ({api_count/total*100:.1f}%)")
    logger.info(f"Titles with inferred genres: {inference_count} ({inference_count/total*100:.1f}%)")
    logger.info(f"Titles with unknown genres: {unknown_count} ({unknown_count/total*100:.1f}%)")

# Entry point
if __name__ == "__main__":
    logger.info("Starting Book Genre Finder")
    
    input_file = os.getenv('INPUT_FILE', 'unique_titles.csv')
    output_file = os.getenv('OUTPUT_FILE', 'book_genres.csv')
    dataset_path = os.getenv('DATASET_PATH', 'datasets/best_books_prepared.csv')
    max_requests = int(os.getenv('MAX_REQUESTS', '1000'))
    
    process_book_titles(input_file, output_file, dataset_path, max_requests)