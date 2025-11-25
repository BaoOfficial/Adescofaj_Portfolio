"""
Clean duplicate projects from the knowledge base
"""

import re

# Read the knowledge base
with open("../knowledge_base/amirlahi_portfolio.md", "r", encoding="utf-8") as f:
    content = f.read()

# Find the Projects section
projects_start = content.find("## Projects")
additional_start = content.find("## Additional Information")

if projects_start == -1 or additional_start == -1:
    print("Error: Could not find Projects section")
    exit(1)

before_projects = content[:projects_start]
projects_section = content[projects_start:additional_start]
after_projects = content[additional_start:]

# Split projects by ### headers
project_pattern = r'(### .+?)(?=###|##|$)'
projects = re.findall(project_pattern, projects_section, re.DOTALL)

print(f"Found {len(projects)} project sections")

# Deduplicate projects by content similarity
unique_projects = []
seen_readmes = set()

for project in projects:
    # Extract README content
    readme_match = re.search(r'\*\*Project Details:\*\*\n\n(.+)', project, re.DOTALL)
    if readme_match:
        readme_content = readme_match.group(1).strip()
        # Use first 500 chars as fingerprint
        fingerprint = readme_content[:500]

        if fingerprint not in seen_readmes:
            seen_readmes.add(fingerprint)
            # Check if this is a customer segmentation duplicate
            if "Customer Segmentation" in project and "cs" in project.lower():
                # Only keep if it's the main Customer-Segmentation project
                if "### Customer-Segmentation" in project:
                    unique_projects.append(project)
                    print("Keeping: Customer-Segmentation")
                else:
                    print(f"Skipping duplicate: {project.split('**Repository:**')[0].strip()}")
            # Check if Routing Optimisation duplicates
            elif "Routing-Optimisation-for-Aeronautical-Networks" in project:
                # Keep the longer one (original, not -2 version)
                if "-2" not in project.split("### ")[1].split("\n")[0]:
                    unique_projects.append(project)
                    print("Keeping: Routing-Optimisation-for-Aeronautical-Networks (original)")
                else:
                    print("Skipping duplicate: Routing-Optimisation-for-Aeronautical-Networks-2")
            else:
                unique_projects.append(project)
                project_name = project.split("### ")[1].split("\n")[0]
                print(f"Keeping: {project_name}")
    else:
        unique_projects.append(project)

print(f"\nFinal count: {len(unique_projects)} unique projects")

# Reconstruct projects section
new_projects_section = "## Projects\n\n" + "".join(unique_projects)

# Write back
new_content = before_projects + new_projects_section + after_projects

with open("../knowledge_base/amirlahi_portfolio.md", "w", encoding="utf-8") as f:
    f.write(new_content)

print("\n[SUCCESS] Duplicates removed successfully!")
