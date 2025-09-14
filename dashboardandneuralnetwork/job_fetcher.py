
import requests

ROLE_QUERY_MAP = {
    "qa engineer": "quality assurance",
    "ml engineer": "machine learning",
    "data scientist": "data science",
    "data analyst": "data analyst",
    "frontend developer": "frontend",
    "backend developer": "backend",
    "python developer": "python",
    "software tester": "testing",
    "software engineer": "software",
}

def fetch_jsearch_jobs(predicted_role):
    query = ROLE_QUERY_MAP.get(predicted_role.lower(), predicted_role)

    url = f"https://jsearch.p.rapidapi.com/search?query={query}&num_pages=1"
    headers = {
        "X-RapidAPI-Key": "2c8a80b56fmshb6bfb9bbfd3ae73p12c0b2jsn8a08b0db2606",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"

        
    }

    try:
        response = requests.get(url, headers=headers)
        print("Jsearch query:", query)
        print("Status code:", response.status_code)
        print("response JSON:", response.text[:300])

        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print("jSearch API Error:", response.status_code)
            return []
    except Exception as e:
        print("exception:", str(e))
        return []

