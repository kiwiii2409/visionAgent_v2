import re
import json
import requests
import asyncio
from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import time
_search_wrapper = DuckDuckGoSearchAPIWrapper()

def _ddg_search(query:str, max_results:int = 5):
    """Retrieves the top 5 results of the ddg search with title, source and snippets"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            raw_result = _search_wrapper.results(query, max_results)
            print(raw_result)
            if not raw_result:
                return []
                
            result = []
            for r in raw_result:
                result.append({
                    "title": r.get('title'),
                    "source": r.get('link'),
                    "snippet": r.get('snippet')
                })
            return result
        
        except Exception as e:
            print(f"[WebSearch] Attempt {attempt + 1} for query \"{query}\" failed: {str(e)}")
            if attempt ==max_retries :
                return []

def _retrieve_website(url:str):
    """given an url, retrieve the cleaned_text (no header, footer, etc. tags)"""
    if not (url.startswith("http://") or url.startswith("https://")):
        print(f"[WebSearch] URL: \"{url}\" is invalid!")
    try:

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url,headers=headers, timeout=3.0) 
        response.raise_for_status() # throw error in case e.g. 404

        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                script.decompose()
        text = soup.get_text(separator=" ")
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        return cleaned_text
    
    except Exception as e:
        print(f"[WebSearch] Fetching URL: \"{url}\" failed with: \n{str(e)}")
    

async def asearch(query, max_results:int=5):
    """Asynchronously fetches the top n websites and snippets, returns a formatted string"""
    results = await asyncio.to_thread(_ddg_search, query, max_results)

    # formatted_result_string = "### Retrieved Websites:"
    # for idx, res in enumerate(results):
    #     formatted_result_string += f"\n\n{idx}: {res["title"]}\nURL: {res["source"]}\nSnippet: {res["snippet"][:5000]}"
    
    return results

async def aretrieve(url:str):
    """Asynchronously fetches the specified url, returns the cleaned text of the website"""
    return await asyncio.to_thread(_retrieve_website, url)
