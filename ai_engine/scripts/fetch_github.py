"""
Script to fetch GitHub repository data and README files.
Fetches all public repositories from GitHub profile and extracts README content.
"""

import requests
import json
import base64
from typing import List, Dict, Optional

GITHUB_USERNAME = "adescofaj"
GITHUB_API_BASE = "https://api.github.com"


def fetch_user_repos(username: str) -> List[Dict]:
    """Fetch all public repositories for a GitHub user."""
    repos = []
    page = 1

    while True:
        url = f"{GITHUB_API_BASE}/users/{username}/repos"
        params = {
            "per_page": 100,
            "page": page,
            "sort": "updated",
            "direction": "desc"
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error fetching repos: {response.status_code}")
            break

        data = response.json()

        if not data:
            break

        repos.extend(data)
        page += 1

    return repos


def fetch_readme(username: str, repo_name: str) -> Optional[str]:
    """Fetch README content for a specific repository."""
    # Try common README file names
    readme_names = ["README.md", "README.MD", "readme.md", "Readme.md", "README"]

    for readme_name in readme_names:
        url = f"{GITHUB_API_BASE}/repos/{username}/{repo_name}/contents/{readme_name}"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            # README content is base64 encoded
            if "content" in data:
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content

    return None


def extract_repo_info(repo: Dict) -> Dict:
    """Extract relevant information from repo data."""
    return {
        "name": repo.get("name"),
        "description": repo.get("description") or "No description provided",
        "url": repo.get("html_url"),
        "language": repo.get("language"),
        "topics": repo.get("topics", []),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "homepage": repo.get("homepage"),
        "is_fork": repo.get("fork", False)
    }


def main():
    print(f"Fetching repositories for {GITHUB_USERNAME}...")

    # Fetch all repos
    repos = fetch_user_repos(GITHUB_USERNAME)

    print(f"Found {len(repos)} repositories")

    # Filter out forked repos (optional - focus on original projects)
    original_repos = [r for r in repos if not r.get("fork", False)]
    print(f"Found {len(original_repos)} original repositories (excluding forks)")

    # Extract info and fetch READMEs
    repo_data = []

    for repo in original_repos:
        print(f"\nProcessing: {repo['name']}")

        info = extract_repo_info(repo)

        # Fetch README
        readme = fetch_readme(GITHUB_USERNAME, repo["name"])

        if readme:
            info["readme"] = readme
            print(f"  [+] README found ({len(readme)} chars)")
        else:
            info["readme"] = None
            print(f"  [-] No README found")

        repo_data.append(info)

    # Save to JSON file
    output_file = "../knowledge_base/github_repos.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(repo_data, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Data saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total repos: {len(repos)}")
    print(f"  Original repos: {len(original_repos)}")
    print(f"  Repos with README: {sum(1 for r in repo_data if r['readme'])}")

    # Print repo list
    print("\n" + "="*60)
    print("Repository List:")
    print("="*60)
    for info in repo_data:
        print(f"\n• {info['name']}")
        print(f"  Language: {info['language'] or 'N/A'}")
        print(f"  Description: {info['description']}")
        print(f"  Stars: {info['stars']} | Forks: {info['forks']}")
        if info['topics']:
            print(f"  Topics: {', '.join(info['topics'])}")


if __name__ == "__main__":
    main()
