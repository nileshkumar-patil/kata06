import pandas as pd
import chromadb
import uuid
import sys

def main():
    print("Initializing ChromaDB Ephemeral Client...")
    # Initialize an ephemeral client (in-memory)
    chroma_client = chromadb.EphemeralClient()
    
    # Create a collection
    collection = chroma_client.create_collection(name="movie_recommendations")
    
    print("Loading data from 'Movies Dataset.xlsx'...")
    try:
        df = pd.read_excel('Movies Dataset.xlsx')
    except Exception as e:
        print(f"Error loading excel file: {e}")
        sys.exit(1)
        
    documents = []
    metadatas = []
    ids = []
    
    # Iterate through the dataframe and prepare data for Chroma
    for index, row in df.iterrows():
        title = str(row.get('Title', 'Unknown Title'))
        description = str(row.get('Description', ''))
        genre = str(row.get('Genre', 'Unknown'))
        director = str(row.get('Director', 'Unknown'))
        
        # Skip if there's no description
        if not description or description == 'nan':
            continue
            
        documents.append(description)
        metadatas.append({
            "Title": title,
            "Genre": genre,
            "Director": director,
            "Genre_lower": genre.lower(),
            "Director_lower": director.lower()
        })
        ids.append(str(uuid.uuid4()))
        
    print(f"Adding {len(documents)} movies to the vector database. Generating embeddings (this may take a moment)...")
    # Add data to the collection
    # Chroma's default embedding function will automatically embed the 'documents'
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("\n[SUCCESS] Database loaded and ready!")
    print("-" * 50)
    
    # Interactive loop
    while True:
        print("\n--- Movie Recommendation System ---")
        try:
            query_text = input("Enter your movie preference/query (or type 'quit' to exit): ").strip()
        except EOFError:
            break
            
        if query_text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
            
        if not query_text:
            print("Please enter a valid query.")
            continue
            
        print("\nOptional Filters (press Enter to skip)")
        try:
            filter_genre = input("Filter by Genre: ").strip()
            filter_director = input("Filter by Director: ").strip()
        except EOFError:
            break
        
        # Build the 'where' clause for metadata filtering
        where_clause = {}
        if filter_genre and filter_director:
            where_clause = {
                "$and": [
                    {"Genre_lower": {"$eq": filter_genre.lower()}},
                    {"Director_lower": {"$eq": filter_director.lower()}}
                ]
            }
        elif filter_genre:
            where_clause = {"Genre_lower": {"$eq": filter_genre.lower()}}
        elif filter_director:
            where_clause = {"Director_lower": {"$eq": filter_director.lower()}}
            
        print("\nSearching for recommendations...")
        
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=5,
                where=where_clause if where_clause else None
            )
            
            # Print the results
            if not results['ids'] or not results['ids'][0]:
                print("No movies found matching your query and filters.")
            else:
                print("\nTop 5 Recommendations:")
                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    dist = results['distances'][0][i] if 'distances' in results and results['distances'] else 'N/A'
                    print(f"{i+1}. {meta['Title']} (Genre: {meta['Genre']}, Director: {meta['Director']})")
                    print(f"   Distance: {dist:.4f}" if isinstance(dist, float) else f"   Distance: {dist}")
        except Exception as e:
            print(f"An error occurred during query: {e}")

if __name__ == "__main__":
    main()
