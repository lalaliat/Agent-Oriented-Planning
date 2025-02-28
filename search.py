from prompt import rewrite_search_agent_prompt
from planner import agent

import json
import serpapi
import requests

class BingSearchAPI:
    def __init__(self):
        with open("keys/bingsearchapi_key.json", "r") as f:
            data = json.load(f)
        self.api_key = data["api_key"]
        self.mkt = data['mkt']
        self.endpoint = data['endpoint']
    def search(self, query: str):
        params = { 'q': query, 'mkt': self.mkt }
        headers = { 'Ocp-Apim-Subscription-Key': self.api_key }
        try:
            response = requests.get(self.endpoint, headers=headers, params=params)
            response.raise_for_status()
            search_result = [response.json()['webPages']['value'][i]['snippet'] for i in range(10)]
            # print(search_result)
            
            prompt = rewrite_search_agent_prompt % (query, search_result)
            for i in range(10):
                res = agent(prompt)
                if isinstance(res['data'], dict):
                    output = res['data']['response']['choices'][0]['message']['content']
                    prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                    completion_tokens = res['data']['response']['usage']['completion_tokens']
                    break
        except:
            output = "Search failed."
            prompt_tokens = 0
            completion_tokens = 0
        return output, prompt_tokens, completion_tokens

    
    def search(self, query: str, use_date=False):
        params = {
            "q": query,
            "location": self.location,
            "api_key": self.api_key
        }
        try:
            search = serpapi.search(params)
            result = search.as_dict()
            output = self.process_result(result, query, use_date=use_date)
        except:
            output = "Search failed."
        return output
