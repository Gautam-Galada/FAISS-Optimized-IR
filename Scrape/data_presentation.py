import json

def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def present_documents(data):
    for topic, articles in data.items():
        print(f"\nTopic: {topic}, Number of Documents: {len(articles)}\n")
        for article in articles[:5]: 
            title = article.get('title', 'Title Unavailable')
            url = article.get('url', 'URL Unavailable')
            summary = article.get('summary', 'Summary Unavailable')[:200]
            print(f"Title: {title}\nURL: {url}\nSummary: {summary}\n---")

if __name__ == "__main__":
    data_path = 'wiki_scraped_articles_final.json'
    data = load_data(data_path)
    present_documents(data)
