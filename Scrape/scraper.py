import aiohttp
import asyncio
import nest_asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os

API_BASE_URL = "https://en.wikipedia.org/w/api.php"
ARTICLES_PER_TOPIC = 100
ARTICLES_BEFORE_CHECKPOINT = 10
MAX_DEPTH = 2
CHECKPOINT_FILE = "scraper_checkpoint.json"

topic_keywords = {
    "Health": ["health", "disease", "medical", "nutrition", "therapy"],
    "Environment": ["climate", "biodiversity", "pollution", "green", "energy"],
}

topics_dictionary = {
    "Health": ["Global health", "Mental health", "Healthcare systems"],
    "Environment": ["Climate change", "Conservation", "Sustainability"],
}

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as file:
            data = json.load(file)
            return data['articles'], set(data['urls'])
    return {}, set()

def save_checkpoint(documented_articles, unique_article_urls):
    checkpoint_data = {'articles': documented_articles, 'urls': list(unique_article_urls)}
    with open(CHECKPOINT_FILE, 'w') as file:
        json.dump(checkpoint_data, file, indent=4)

async def extract_article_summary(session, url):
    async with session.get(url) as response:
        soup = BeautifulSoup(await response.text(), 'html.parser')
        return ' '.join(p.text for p in soup.select('p'))

def is_relevant(summary, topic):
    keywords = topic_keywords[topic]
    return sum(1 for keyword in keywords if keyword in summary.lower()) >= 3

async def retrieve_articles(session, topic, subtopic):
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": subtopic,
        "format": "json",
        "srlimit": 10
    }
    async with session.get(API_BASE_URL, params=search_params) as response:
        data = await response.json()
        return data.get("query", {}).get("search", [])

async def execute_scraping():
    async with aiohttp.ClientSession() as session:
        documented_articles, unique_article_urls = load_checkpoint()
        for topic, subtopics in topics_dictionary.items():
            documented_articles.setdefault(topic, [])
            for subtopic in subtopics:
                articles = await retrieve_articles(session, topic, subtopic)
                for article in articles:
                    url = f"https://en.wikipedia.org/wiki/{article['title'].replace(' ', '_')}"
                    if url in unique_article_urls:
                        continue
                    unique_article_urls.add(url)
                    summary = await extract_article_summary(session, url)
                    if is_relevant(summary, topic):
                        documented_articles[topic].append({
                            "title": article['title'],
                            "url": url,
                            "summary": summary
                        })
                save_checkpoint(documented_articles, unique_article_urls)

        with open('wiki_scraped_articles_final.json', 'w') as file:
            json.dump(documented_articles, file, indent=4)

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(execute_scraping())
