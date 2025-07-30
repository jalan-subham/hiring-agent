"""
GitHub Integration Module

This module provides functionality to interact with GitHub API and extract
GitHub profile and repository information for resume evaluation.
"""

import os
import re
import json
from typing import Dict, List, Optional
import requests
from pydantic import BaseModel
import ollama
from prompts.template_manager import TemplateManager


class GitHubProfile(BaseModel):
    """Pydantic model for GitHub profile data."""
    username: str
    name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    avatar_url: Optional[str] = None
    blog: Optional[str] = None
    twitter_username: Optional[str] = None
    hireable: Optional[bool] = None


def _fetch_github_api(api_url, params=None):
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    response = requests.get(api_url, params, timeout=10, headers=headers)
    return response

def extract_github_username(github_url: str) -> Optional[str]:
    """
    Extract username from various GitHub URL formats.
    
    Args:
        github_url (str): GitHub URL or username
        
    Returns:
        str: Extracted username or None if not found
    """
    if not github_url:
        return None
    
    # Clean up the URL by removing all spaces
    github_url = github_url.replace(' ', '')
    
    # Remove common prefixes
    github_url = github_url.strip()
    
    # Handle different URL formats
    patterns = [
        r'https?://github\.com/([^/]+)',  # https://github.com/username
        r'github\.com/([^/]+)',           # github.com/username
        r'@([^/]+)',                      # @username
        r'^([a-zA-Z0-9-]+)$'             # Just username
    ]
    
    for pattern in patterns:
        match = re.search(pattern, github_url)
        if match:
            return match.group(1)
    
    return None


def fetch_github_profile(github_url: str) -> Optional[GitHubProfile]:
    """
    Fetch GitHub profile details using GitHub API.
    
    Args:
        github_url (str): GitHub URL or username
        
    Returns:
        GitHubProfile: GitHub profile data or None if failed
    """
    try:
        username = extract_github_username(github_url)
        if not username:
            print(f"Could not extract username from: {github_url}")
            return None
        
        # GitHub API endpoint
        api_url = f"https://api.github.com/users/{username}"
        
        # Make request to GitHub API
        response = _fetch_github_api(api_url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create GitHubProfile object
            profile = GitHubProfile(
                username=username,
                name=data.get('name'),
                bio=data.get('bio'),
                location=data.get('location'),
                company=data.get('company'),
                public_repos=data.get('public_repos'),
                followers=data.get('followers'),
                following=data.get('following'),
                created_at=data.get('created_at'),
                updated_at=data.get('updated_at'),
                avatar_url=data.get('avatar_url'),
                blog=data.get('blog'),
                twitter_username=data.get('twitter_username'),
                hireable=data.get('hireable')
            )
            
            return profile
            
        elif response.status_code == 404:
            print(f"GitHub user not found: {username}")
            return None
        else:
            print(f"GitHub API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GitHub profile: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching GitHub profile: {e}")
        return None


def fetch_repo_contributors(owner: str, repo_name: str) -> int:
    """
    Fetch the number of contributors for a specific repository.
    
    Args:
        owner (str): Repository owner username
        repo_name (str): Repository name
        
    Returns:
        int: Number of contributors
    """
    try:
        # GitHub API endpoint for repository contributors
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}/contributors"
        
        # Use authenticated request for better rate limiting
        response = _fetch_github_api(api_url)
        
        if response.status_code == 200:
            contributors_data = response.json()
            return len(contributors_data)
        else:
            # If we can't fetch contributors, assume it's a single-person project
            return 1
            
    except Exception as e:
        logger.error(f"Error fetching contributors for {owner}/{repo_name}: {e}")
        return 1


def fetch_all_github_repos(github_url: str, max_repos: int = 100) -> List[Dict]:
    """
    Fetch all repositories from a GitHub profile with contributor information.
    
    Args:
        github_url (str): GitHub URL or username
        max_repos (int): Maximum number of repositories to fetch
        
    Returns:
        List[Dict]: List of repository dictionaries with contributor counts
    """
    try:
        username = extract_github_username(github_url)
        if not username:
            print(f"Could not extract username from: {github_url}")
            return []
        
        # GitHub API endpoint for user repositories
        api_url = f"https://api.github.com/users/{username}/repos"
        
        # Parameters for the request
        params = {
            'sort': 'updated',  # Sort by last updated
            'per_page': min(max_repos, 100),  # GitHub API limit is 100 per page
            'type': 'owner'  # Only get repositories owned by the user
        }
        
        # Make request to GitHub API
        response = _fetch_github_api(api_url, params=params)
        
        if response.status_code == 200:
            repos_data = response.json()
            
            # Convert to our project format
            projects = []
            for repo in repos_data:
                # Skip forked repositories unless they have significant activity
                if repo.get('fork') and repo.get('forks_count', 0) < 5:
                    continue
                
                # Fetch contributor count for this repository
                repo_name = repo.get('name')
                contributor_count = fetch_repo_contributors(username, repo_name)
                
                # Classify project type based on contributor count
                project_type = "open_source" if contributor_count > 1 else "self_project"
                
                project = {
                    'name': repo.get('name'),
                    'description': repo.get('description'),
                    'github_url': repo.get('html_url'),
                    'live_url': repo.get('homepage') if repo.get('homepage') else None,
                    'technologies': [repo.get('language')] if repo.get('language') else [],
                    'project_type': project_type,  # New field to classify project type
                    'contributor_count': contributor_count,  # New field for contributor count
                    'github_details': {
                        'stars': repo.get('stargazers_count', 0),
                        'forks': repo.get('forks_count', 0),
                        'language': repo.get('language'),
                        'description': repo.get('description'),
                        'created_at': repo.get('created_at'),
                        'updated_at': repo.get('updated_at'),
                        'topics': repo.get('topics', []),
                        'open_issues': repo.get('open_issues_count', 0),
                        'size': repo.get('size', 0),
                        'fork': repo.get('fork', False),
                        'archived': repo.get('archived', False),
                        'default_branch': repo.get('default_branch'),
                        'contributors': contributor_count  # Add contributor count to github_details
                    }
                }
                projects.append(project)
            
            # Sort by stars (descending) to show most popular repos first
            projects.sort(key=lambda x: x['github_details']['stars'], reverse=True)
            
            # Count project types
            open_source_count = sum(1 for p in projects if p['project_type'] == 'open_source')
            self_project_count = sum(1 for p in projects if p['project_type'] == 'self_project')
            
            print(f"✅ Found {len(projects)} repositories")
            print(f"📊 Project classification: {open_source_count} open source, {self_project_count} self projects")
            return projects
            
        elif response.status_code == 404:
            print(f"GitHub user not found: {username}")
            return []
        else:
            print(f"GitHub API error: {response.status_code} - {response.text}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching GitHub repositories: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error fetching GitHub repositories: {e}")
        return []


def generate_profile_json(profile: GitHubProfile) -> Dict:
    """
    Generate JSON data for GitHub profile.
    
    Args:
        profile (GitHubProfile): GitHub profile data
        
    Returns:
        Dict: JSON-serializable dictionary of profile data
    """
    if not profile:
        return {}
    
    profile_data = {
        "username": profile.username,
        "name": profile.name,
        "bio": profile.bio,
        "location": profile.location,
        "company": profile.company,
        "public_repos": profile.public_repos,
        "followers": profile.followers,
        "following": profile.following,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "avatar_url": profile.avatar_url,
        "blog": profile.blog,
        "twitter_username": profile.twitter_username,
        "hireable": profile.hireable
    }
    
    return profile_data


def generate_projects_json(projects: List[Dict]) -> List[Dict]:
    """
    Generate JSON data for GitHub projects/repositories using LLM to select top 5 projects.
    
    Args:
        projects (List[Dict]): List of project dictionaries
        
    Returns:
        List[Dict]: JSON-serializable list of top 5 project data
    """
    if not projects:
        return []
    
    # If we have 5 or fewer projects, return all of them
    if len(projects) <= 5:
        projects_data = []
        for project in projects:
            project_data = {
                "name": project.get('name'),
                "description": project.get('description'),
                "github_url": project.get('github_url'),
                "live_url": project.get('live_url'),
                "technologies": project.get('technologies', []),
                "project_type": project.get('project_type', 'self_project'),  # Add project type
                "contributor_count": project.get('contributor_count', 1),  # Add contributor count
                "github_details": project.get('github_details', {})
            }
            projects_data.append(project_data)
        return projects_data
    
    # Use LLM to select top 5 projects
    try:
        # Prepare the projects data for the LLM
        projects_data = []
        for project in projects:
            project_data = {
                "name": project.get('name'),
                "description": project.get('description'),
                "github_url": project.get('github_url'),
                "live_url": project.get('live_url'),
                "technologies": project.get('technologies', []),
                "project_type": project.get('project_type', 'self_project'),  # Add project type
                "contributor_count": project.get('contributor_count', 1),  # Add contributor count
                "github_details": project.get('github_details', {})
            }
            projects_data.append(project_data)
        
        # Convert to JSON string for the prompt
        projects_json = json.dumps(projects_data, indent=2)
        
        # Prepare the prompt using template manager
        template_manager = TemplateManager()
        prompt = template_manager.render_github_project_selection_template(projects_data=projects_json)
        
        if not prompt:
            print("❌ Failed to render GitHub project selection template")
            # Fallback: return first 5 projects
            projects_data = []
            for project in projects[:5]:
                project_data = {
                    "name": project.get('name'),
                    "description": project.get('description'),
                    "github_url": project.get('github_url'),
                    "live_url": project.get('live_url'),
                    "technologies": project.get('technologies', []),
                    "project_type": project.get('project_type', 'self_project'),  # Add project type
                    "contributor_count": project.get('contributor_count', 1),  # Add contributor count
                    "github_details": project.get('github_details', {})
                }
                projects_data.append(project_data)
            
            return projects_data
        
        print(f"🤖 Using LLM to select top 5 projects from {len(projects)} repositories...")
        
        # Call Ollama API
        response = ollama.chat(
            model="gemma3:4b",  # Use the same model as in run.py
            messages=[
                {
                    'role': 'system',
                    'content': 'You are an expert technical recruiter analyzing GitHub repositories to identify the most impressive projects. CRITICAL: You must select exactly 5 UNIQUE projects - no duplicates allowed. Each project must be different from the others.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.1,
                'top_p': 0.9
            }
        )
        
        # Extract the response content
        response_text = response['message']['content']
        
        # Parse JSON from the response
        try:
            # Clean the response to extract JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Parse JSON
            selected_projects = json.loads(response_text)
            
            # Ensure uniqueness by removing duplicates
            unique_projects = []
            seen_names = set()
            
            for project in selected_projects:
                project_name = project.get('name', '')
                if project_name and project_name not in seen_names:
                    unique_projects.append(project)
                    seen_names.add(project_name)
            
            # If we have duplicates, add more projects to reach 5 unique ones
            if len(unique_projects) < 5:
                print(f"⚠️ LLM selected {len(selected_projects)} projects but {len(unique_projects)} are unique")
                
                # Add more unique projects from the original list
                for project in projects_data:
                    if len(unique_projects) >= 5:
                        break
                    project_name = project.get('name', '')
                    if project_name and project_name not in seen_names:
                        unique_projects.append(project)
                        seen_names.add(project_name)
            
            project_names = ', '.join([proj.get('name', 'N/A') for proj in unique_projects])
            print(f"✅ LLM selected {len(unique_projects)} unique top projects: {project_names}")
            return unique_projects
            
        except json.JSONDecodeError as e:
            
            debug_log(f"ERROR: Error parsing LLM response: {e}")
            debug_log(f"ERROR: Raw response: {response_text}")
            
            # Fallback: return first 5 projects
            print("🔄 Falling back to first 5 projects")
            return projects_data[:5]
            
    except Exception as e:
        print(f"Error using LLM for project selection: {e}")
        print("🔄 Falling back to first 5 projects")
        
        # Fallback: return first 5 projects
        projects_data = []
        for project in projects[:5]:
            project_data = {
                "name": project.get('name'),
                "description": project.get('description'),
                "github_url": project.get('github_url'),
                "live_url": project.get('live_url'),
                "technologies": project.get('technologies', []),
                "project_type": project.get('project_type', 'self_project'),  # Add project type
                "contributor_count": project.get('contributor_count', 1),  # Add contributor count
                "github_details": project.get('github_details', {})
            }
            projects_data.append(project_data)
        
        return projects_data


def fetch_and_display_github_info(github_url: str) -> Dict:
    """
    Fetch and display GitHub profile and all repository information.
    
    Args:
        github_url (str): GitHub URL or username
        
    Returns:
        Dict: JSON data containing profile and projects information
    """
    # Fetch GitHub profile
    github_profile = fetch_github_profile(github_url)
    if not github_profile:
        print("\n❌ Failed to fetch GitHub profile details.")
        return {}
    
    # Fetch all repository details
    print("🔍 Fetching all repository details...")
    projects = fetch_all_github_repos(github_url)
    
    if not projects:
        print("\n❌ No repositories found or failed to fetch repository details.")
    
    # Generate and return JSON data
    profile_json = generate_profile_json(github_profile)
    projects_json = generate_projects_json(projects)
    
    result = {
        "profile": profile_json,
        "projects": projects_json,
        "total_projects": len(projects_json)
    }
    
    return result


def main(github_url):
    # Fetch data and get JSON
    result = fetch_and_display_github_info(github_url)
    
    # Print JSON data
    print("\n" + "="*60)
    print("JSON DATA OUTPUT")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("="*60)
    
    return result

if __name__ == "__main__":
    main("https://github.com/sp2hari/")