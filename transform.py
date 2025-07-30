"""
Transform Module

This module contains all the transformation functions for processing parsed data
from LLM responses to conform to the JSON Resume schema.
"""

from typing import Dict, List, Optional


def transform_parsed_data(parsed_data: Dict) -> Dict:
    """
    Transform parsed data to handle common LLM response format issues.
    
    This method ensures that the parsed JSON data from the LLM response
    conforms to the expected JSON Resume schema by providing default values
    for missing fields and converting string arrays to proper object structures.
    
    Args:
        parsed_data (Dict): The parsed JSON data from LLM response
        
    Returns:
        Dict: Cleaned and transformed data conforming to JSON Resume schema
        
    Raises:
        Exception: For data transformation errors
    """
    try:
        # Handle common issues with LLM responses
        if isinstance(parsed_data, dict):
            # Check if this is a complete resume or individual section
            if 'basics' in parsed_data and len(parsed_data) > 1:
                # This appears to be complete resume data
                transformed = {
                    'basics': transform_basics(parsed_data.get('basics', {})),
                    'work': transform_work_experience(parsed_data.get('work_experience', parsed_data.get('work', parsed_data.get('experience', [])))),
                    'volunteer': transform_organizations(parsed_data.get('organizations', [])),
                    'education': transform_education(parsed_data.get('education', [])),
                    'awards': transform_achievements(parsed_data.get('achievements', parsed_data.get('awards', parsed_data.get('honors_and_awards', [])))),
                    'certificates': parsed_data.get('certificates', []),
                    'publications': parsed_data.get('publications', []),
                    'skills': transform_skills_comprehensive(parsed_data),
                    'languages': parsed_data.get('languages', []),
                    'interests': parsed_data.get('interests', []),
                    'references': parsed_data.get('references', []),
                    'projects': transform_projects_comprehensive(parsed_data),
                    'meta': parsed_data.get('meta', {})
                }
            else:
                # This appears to be individual section data
                # Apply section-specific transformations
                if 'basics' in parsed_data:
                    # For basics section, the data might be nested or direct
                    basics_data = parsed_data.get('basics', parsed_data)
                    transformed = {'basics': transform_basics(basics_data)}
                elif 'work' in parsed_data or 'work_experience' in parsed_data or 'experience' in parsed_data:
                    work_data = parsed_data.get('work', parsed_data.get('work_experience', parsed_data.get('experience', [])))
                    transformed = {'work': transform_work_experience(work_data)}
                elif 'education' in parsed_data:
                    transformed = {'education': transform_education(parsed_data.get('education', []))}
                elif 'skills' in parsed_data or 'librariesFrameworks' in parsed_data or 'toolsPlatforms' in parsed_data or 'databases' in parsed_data:
                    transformed = {'skills': transform_skills_comprehensive(parsed_data)}
                elif 'projects' in parsed_data or 'projectsOpenSource' in parsed_data:
                    transformed = {'projects': transform_projects_comprehensive(parsed_data)}
                elif 'awards' in parsed_data or 'achievements' in parsed_data or 'honors_and_awards' in parsed_data:
                    awards_data = parsed_data.get('awards', parsed_data.get('achievements', parsed_data.get('honors_and_awards', [])))
                    transformed = {'awards': transform_achievements(awards_data)}
                else:
                    # If we can't determine the section type, return as-is
                    transformed = parsed_data
            
            return transformed
        else:
            return parsed_data
            
    except Exception as e:
        print(f"Error transforming parsed data: {e}")
        return parsed_data


def transform_basics(basics_data: Dict) -> Dict:
    """Transform basics data and fix network field based on URL."""
    if not isinstance(basics_data, dict):
        return basics_data
        
    # Handle profiles/profiles array
    profiles = basics_data.get('profiles', [])
    
    transformed_profiles = []
    if isinstance(profiles, list):
        for i, profile in enumerate(profiles):
            if isinstance(profile, dict):
                # Create a copy of the profile to avoid modifying the original
                transformed_profile = profile.copy()
                url = transformed_profile.get('url', '')
                network = transformed_profile.get('network')
                
                if url and network is None:
                    if 'github.com' in url:
                        transformed_profile['network'] = 'GitHub'
                        # Extract username from GitHub URL
                        username = extract_username_from_url(url, 'github.com')
                        if username:
                            transformed_profile['username'] = username
                    elif 'linkedin.com' in url:
                        transformed_profile['network'] = 'LinkedIn'
                        # Extract username from LinkedIn URL
                        username = extract_username_from_url(url, 'linkedin.com')
                        if username:
                            transformed_profile['username'] = username
                    elif 'leetcode.com' in url:
                        transformed_profile['network'] = 'LeetCode'
                        # Extract username from LeetCode URL
                        username = extract_username_from_url(url, 'leetcode.com')
                        if username:
                            transformed_profile['username'] = username
                    elif 'stackoverflow.com' in url:
                        transformed_profile['network'] = 'Stack Overflow'
                        # Extract username from Stack Overflow URL
                        username = extract_username_from_url(url, 'stackoverflow.com')
                        if username:
                            transformed_profile['username'] = username
                    elif 'hackerrank.com' in url:
                        transformed_profile['network'] = 'HackerRank'
                        # Extract username from HackerRank URL
                        username = extract_username_from_url(url, 'hackerrank.com')
                        if username:
                            transformed_profile['username'] = username
                transformed_profiles.append(transformed_profile)
    
    # Update the basics_data with the transformed profiles
    basics_data['profiles'] = transformed_profiles
    return basics_data


def extract_username_from_url(url: str, domain: str) -> str:
    """Extract username from URL based on domain and optional prefix."""
    try:
        # Remove protocol and domain
        path = url.split(domain)[1] if domain in url else ''
        if not path:
            return ''
        path = path.lstrip('/')
        
        # Split by '/' and get the parts
        parts = [part for part in path.split('/') if part]
        
        if parts:
            if domain == 'linkedin.com':
                return parts[1]
            elif domain == 'stackoverflow.com':
                return parts[2]
            else:
                return parts[0]
        return ''
    except Exception:
        return ''


def transform_work_experience(work_list: List) -> List[Dict]:
    """Transform work_experience to work format."""
    transformed = []
    for item in work_list:
        if isinstance(item, dict):
            # Handle description as array or string
            description = item.get('description', '')
            if isinstance(description, list):
                description = ' '.join(description)
            
            transformed.append({
                'name': item.get('name', ''),
                'position': item.get('position', item.get('type', item.get('title', ''))),
                'url': item.get('url', None),
                'startDate': parse_date_range(item.get('years', '')) if 'years' in item else item.get('startDate'),
                'endDate': parse_end_date(item.get('years', '')) if 'years' in item else item.get('endDate'),
                'summary': item.get('summary', description),
                'highlights': item.get('highlights', [])
            })
    return transformed


def transform_organizations(org_list: List) -> List[Dict]:
    """Transform organizations to volunteer format."""
    transformed = []
    for item in org_list:
        if isinstance(item, dict):
            transformed.append({
                'organization': item.get('name', ''),
                'position': item.get('role', ''),
                'url': item.get('url', None),
                'startDate': None,
                'endDate': 'Present',
                'summary': None,
                'highlights': []
            })
    return transformed


def transform_education(edu_list: List) -> List[Dict]:
    """Transform education format."""
    transformed = []
    for item in edu_list:
        if isinstance(item, dict):
            # Handle different education formats
            if 'degree' in item:
                # New format from LLM
                score = item.get('gpa', item.get('percentage', None))
                if score is not None:
                    score = str(score)  # Ensure score is always a string
                
                transformed.append({
                    'institution': item.get('institution', ''),
                    'url': item.get('url', None),
                    'area': item.get('degree', '').split(', ')[-1] if ',' in item.get('degree', '') else None,
                    'studyType': item.get('degree', '').split(', ')[0] if ',' in item.get('degree', '') else item.get('degree', ''),
                    'startDate': parse_date_range(item.get('years', '')),
                    'endDate': parse_end_date(item.get('years', '')),
                    'score': score,
                    'courses': []
                })
            else:
                # Original format
                transformed.append(item)
    return transformed


def transform_achievements(achievements_list: List) -> List[Dict]:
    """Transform achievements to awards format."""
    transformed = []
    for item in achievements_list:
        if isinstance(item, dict):
            # Handle different award formats
            title = item.get('title', item.get('name', ''))
            awarder = item.get('awarder', item.get('organization', ''))
            summary = item.get('summary', item.get('description', None))
            
            transformed.append({
                'title': title,
                'date': f"{item.get('year', '')}-01" if item.get('year') else None,
                'awarder': awarder,
                'summary': summary
            })
    return transformed


def transform_skills(skills_list: List) -> List[Dict]:
    """Transform skills format."""
    transformed = []
    for item in skills_list:
        if isinstance(item, dict):
            if 'category' in item:
                # New format from LLM
                transformed.append({
                    'name': item.get('category', ''),
                    'level': None,
                    'keywords': item.get('keywords', [])
                })
            else:
                # Original format
                transformed.append(item)
    return transformed


def transform_projects(projects_list: List) -> List[Dict]:
    """Transform projects format."""
    transformed = []
    for item in projects_list:
        if isinstance(item, dict):
            # Extract skills from project name if it contains "|"
            skills = []
            project_name = item.get('name', '')
            if '|' in project_name:
                name_parts = project_name.split('|')
                if len(name_parts) > 1:
                    skills_part = name_parts[1].strip()
                    skills = [skill.strip() for skill in skills_part.split(',')]
                    # Update project name to remove skills part
                    item['name'] = name_parts[0].strip()
            
            # Handle technologies field (could be string or array)
            technologies = item.get('technologies', [])
            if isinstance(technologies, str):
                technologies = [tech.strip() for tech in technologies.split(',')]
            
            # If no skills extracted from name, use technologies as skills
            if not skills and technologies:
                skills = technologies
            
            transformed.append({
                'name': item.get('name', ''),
                'startDate': None,
                'endDate': None,
                'description': item.get('description', ''),
                'highlights': [item.get('type', '')] if item.get('type') else [],
                'url': item.get('url', None),
                'technologies': technologies,
                'skills': skills
            })
    return transformed


def transform_skills_comprehensive(parsed_data: Dict) -> List[Dict]:
    """Transform skills from various possible formats."""
    skills = []
    
    # Handle skills as array of strings
    if 'skills' in parsed_data and isinstance(parsed_data['skills'], list):
        if parsed_data['skills'] and isinstance(parsed_data['skills'][0], str):
            skills.append({
                'name': 'Programming Languages',
                'level': None,
                'keywords': parsed_data['skills']
            })
        else:
            skills.extend(transform_skills(parsed_data['skills']))
    
    # Handle separate skill categories
    skill_categories = {
        'librariesFrameworks': 'Libraries/Frameworks',
        'toolsPlatforms': 'Tools/Platforms', 
        'databases': 'Databases'
    }
    
    for field, category_name in skill_categories.items():
        if field in parsed_data and isinstance(parsed_data[field], list):
            skills.append({
                'name': category_name,
                'level': None,
                'keywords': parsed_data[field]
            })
    
    return skills


def transform_projects_comprehensive(parsed_data: Dict) -> List[Dict]:
    """Transform projects from various possible formats."""
    projects = []
    
    # Handle standard projects
    if 'projects' in parsed_data:
        projects.extend(transform_projects(parsed_data['projects']))
    
    # Handle projectsOpenSource
    if 'projectsOpenSource' in parsed_data:
        for item in parsed_data['projectsOpenSource']:
            if isinstance(item, dict):
                # Extract skills from project name if it contains "|"
                skills = []
                project_name = item.get('name', '')
                if '|' in project_name:
                    name_parts = project_name.split('|')
                    if len(name_parts) > 1:
                        skills_part = name_parts[1].strip()
                        skills = [skill.strip() for skill in skills_part.split(',')]
                        # Update project name to remove skills part
                        item['name'] = name_parts[0].strip()
                
                projects.append({
                    'name': item.get('name', ''),
                    'startDate': None,
                    'endDate': None,
                    'description': item.get('summary', ''),
                    'highlights': [],
                    'url': item.get('url', None),
                    'technologies': item.get('technologies', []),
                    'skills': skills
                })
    
    return projects


def parse_date_range(date_range: str) -> str:
    """Parse date ranges like 'Mar-May 2020' or '2007-2019'."""
    if not date_range:
        return None
    
    # Handle "Mar-May 2020" format
    if ' ' in date_range and any(month in date_range for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
        parts = date_range.split(' ')
        if len(parts) >= 2:
            year = parts[-1]
            month_map = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
            start_month = month_map.get(parts[0], '01')
            return f"{year}-{start_month}"
    
    # Handle "2007-2019" format
    if '-' in date_range and len(date_range.split('-')) == 2:
        start_year = date_range.split('-')[0]
        return f"{start_year}-01"
    
    return None


def parse_end_date(date_range: str) -> str:
    """Parse end date from date ranges."""
    if not date_range:
        return None
    
    # Handle "Feb-onwards 2021" format
    if 'onwards' in date_range:
        return 'Present'
    
    # Handle "Mar-May 2020" format
    if ' ' in date_range and any(month in date_range for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
        parts = date_range.split(' ')
        if len(parts) >= 3:
            year = parts[-1]
            month_map = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
            end_month = month_map.get(parts[1], '12')
            return f"{year}-{end_month}"
    
    # Handle "2007-2019" format
    if '-' in date_range and len(date_range.split('-')) == 2:
        end_year = date_range.split('-')[1]
        return f"{end_year}-12"
    
    return None 