'''
Tool Calling Layer (tools.py)

Adding tool calling with API integration (bonus feature).

Example tool: fetch GitHub repository info.
'''
import requests


def get_github_repo_info(repo):

    url = f"https://api.github.com/repos/{repo}"

    response = requests.get(url)

    if response.status_code != 200:
        return "Repository not found."

    data = response.json()

    return {
        "name": data["name"],
        "stars": data["stargazers_count"],
        "description": data["description"]
    }