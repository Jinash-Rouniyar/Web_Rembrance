from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from openai import OpenAI
import ujson
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, 'venv', 'Lib', 'site-packages')
sys.path.append(relative_path)

try:
    from readability import Document
except:
    print("Error importing Document from readability")

COMPLETION_MODEL = "gpt-4o"
SOURCE_COUNT = 5
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_search_query(text: str, model="gpt-4o") -> str:
    response = client.chat.completions.create(
        model=model,
        #use json return so that it can give three exact queries
        messages=[
            {"role": "system", "content": "Given a query, respond with 3 similar Google search query that would best help to answer the query. Don't use search operators. Respond with only the Google queries and nothing else."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def get_google_search_links(query: str, source_count: int = SOURCE_COUNT, proxies: dict = None) -> list[str]:
    """
    Scrapes the official Google search page using the `requests` module and returns the first `source_count` links.
    """
    url = f"https://www.google.com/search?q={query}"
    if proxies:
        response = requests.get(url, proxies=proxies)
    else:
        response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    link_tags = soup.find_all("a")
    
    links = []
    for link in link_tags:
        href = link.get("href")
        if href and href.startswith("/url?q="):
            cleaned_href = parse_qs(href)["/url?q"][0]
            if cleaned_href not in links:
                links.append(cleaned_href)

    filtered_links = []
    exclude_list = ["google", "facebook", "twitter", "instagram", "youtube", "tiktok", "reddit"]
    for link in links:
        parsed_url = urlparse(link)
        domain = parsed_url.hostname
        if domain:  # Check if domain is not None
            if not any(site in domain for site in exclude_list):
                if not any(urlparse(l).hostname == domain for l in filtered_links):
                    filtered_links.append(link)
    
    return filtered_links[:source_count]

def scrape_text_from_links(links: list, proxies: dict = None) -> list[dict]:   
    with ThreadPoolExecutor(max_workers=len(links)) as executor:
        results = list(executor.map(lambda l: scrape_text_from_link(l, proxies), links))
    
    for i, result in enumerate(results, start=1):
        result["result_number"] = i

    return results
    
def scrape_text_from_link(link: str, proxies: dict = None) -> dict:
    try:
        response = requests.get(link, proxies=proxies) if proxies else requests.get(link)
        doc = Document(response.text)
        parsed = doc.summary()
        soup = BeautifulSoup(parsed, "html.parser")
        source_text = soup.get_text()
        summarized_text = summarize_text(source_text[:50000])
        return {"url": link, "text": summarized_text}
    except Exception as e:
        print(f"Failed to scrape {link}: {e}")
        return {"url": link, "text": "Failed to scrape"}

def summarize_text(text: str, model="gpt-4o") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Given text, respond with the summarized text (no more than 100 words) and nothing else."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in summarization: {e}")
        return text[:100] + "..."

def search(query: str, proxies: dict = None) -> tuple[list[str], list[dict]]:
    """
    This function takes a query as input, gets top Google search links for the query, and then scrapes the text from the links.
    It returns a tuple containing the list of links and a list of dictionaries. Each dictionary contains the URL and the summarized text from the link.
    """
    links = get_google_search_links(query, proxies=proxies)
    sources = scrape_text_from_links(links, proxies=proxies)

    return links, sources

def perplexity_clone(query: str, proxies: dict = None, verbose=False) -> str:
    """
    A clone of Perplexity AI's "Search" feature. This function takes a query as input and returns Markdown formatted text containing a response to the query with cited sources.
    """
    formatted_time = datetime.utcnow().strftime("%A, %B %d, %Y %H:%M:%S UTC")

    search_query = generate_search_query(query)
    if verbose:
        print(f"Searching \"{search_query}\"...")
    links, sources = search(search_query, proxies=proxies)

    result = openai.ChatCompletion.create(
        model=COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": "Generate a comprehensive and informative answer for a given question solely based on the provided web Search Results (URL and Summary). You must only use information from the provided search results. Use an unbiased and journalistic tone. Use this current date and time: " + formatted_time + ". Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${number}] notation, and don't link the citations. Only cite the most relevant results that answer the question accurately. If different results refer to different entities with the same name, write separate answers for each entity."},
            {"role": "user", "content": ujson.dumps(sources)},
            {"role": "user", "content": query}
        ]
    )["choices"][0]["message"]["content"]

    for i, link in enumerate(links, start=1):
        result = result.replace(f"[{i}]", f"<sup>[[{i}]]({link})</sup>")
        
    return result