"""
Script to add GitHub project data to the knowledge base Markdown file.
Parses github_repos.json and appends projects to amirlahi_portfolio.md
"""

import json
from collections import defaultdict

# Load GitHub repos data
with open("../knowledge_base/github_repos.json", "r", encoding="utf-8") as f:
    repos = json.load(f)

# Deduplicate repos by name (handle Customer-Segmentation duplicates)
unique_repos = {}
for repo in repos:
    name = repo["name"]
    # Keep the first occurrence or the one with most stars/most recent
    if name not in unique_repos:
        unique_repos[name] = repo
    # For Customer Segmentation variants, keep only Customer-Segmentation
    if "customer" in name.lower() and "segmentation" in name.lower():
        if name == "Customer-Segmentation":
            unique_repos["Customer-Segmentation"] = repo
        elif "Customer-Segmentation" not in unique_repos:
            unique_repos["Customer-Segmentation"] = repo

# Remove duplicate customer segmentation entries
repos_to_process = []
customer_seg_added = False
for name, repo in unique_repos.items():
    if "customer" in name.lower() and "segmentation" in name.lower():
        if not customer_seg_added:
            repos_to_process.append(repo)
            customer_seg_added = True
    else:
        repos_to_process.append(repo)

# Sort by updated date (most recent first)
repos_to_process.sort(key=lambda x: x["updated_at"], reverse=True)

print(f"Processing {len(repos_to_process)} unique repositories\n")

# Generate Markdown content for Projects section
projects_md = "\n## Projects\n\n"

for repo in repos_to_process:
    name = repo["name"]
    description = repo["description"]
    url = repo["url"]
    language = repo["language"] or "N/A"
    topics = repo["topics"]
    readme = repo["readme"]

    # Skip if no README
    if not readme:
        print(f"Skipping {name} (no README)")
        continue

    print(f"Adding: {name}")

    # Add project subsection
    projects_md += f"### {name}\n\n"
    projects_md += f"**Repository:** [{name}]({url})\n\n"
    projects_md += f"**Description:** {description}\n\n"
    projects_md += f"**Primary Language:** {language}\n\n"

    if topics:
        projects_md += f"**Topics:** {', '.join(topics)}\n\n"

    # Add README content (truncate if too long)
    projects_md += f"**Project Details:**\n\n{readme}\n\n"
    projects_md += "---\n\n"

# Read existing knowledge base
with open("../knowledge_base/amirlahi_portfolio.md", "r", encoding="utf-8") as f:
    kb_content = f.read()

# Find the Projects section and replace it
projects_section_start = kb_content.find("## Projects")
additional_section_start = kb_content.find("## Additional Information")

if projects_section_start != -1 and additional_section_start != -1:
    # Replace Projects section content
    before_projects = kb_content[:projects_section_start]
    after_projects = kb_content[additional_section_start:]

    new_kb_content = before_projects + projects_md + after_projects

    # Write back to file
    with open("../knowledge_base/amirlahi_portfolio.md", "w", encoding="utf-8") as f:
        f.write(new_kb_content)

    print(f"\n✓ Successfully added {len([r for r in repos_to_process if r['readme']])} projects to knowledge base")
else:
    print("Error: Could not find Projects section in knowledge base")
