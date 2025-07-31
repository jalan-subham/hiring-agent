from typing import Dict, List, Optional


def transform_parsed_data(parsed_data: Dict) -> Dict:
    try:
        if isinstance(parsed_data, dict):
            if 'basics' in parsed_data and len(parsed_data) > 1:
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
                if 'basics' in parsed_data:
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
                    transformed = parsed_data
            
            return transformed
        else:
            return parsed_data
            
    except Exception as e:
        print(f"Error transforming parsed data: {e}")
        return parsed_data


def extract_domain_from_url(url: str) -> str:
    try:
        if '://' in url:
            url = url.split('://')[1]
        domain = url.split('/')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return ''

def get_network_name(domain: str) -> str:
    domain_mapping = {
        'github.com': 'GitHub',
        'linkedin.com': 'LinkedIn',
        'leetcode.com': 'LeetCode',
        'stackoverflow.com': 'Stack Overflow',
        'hackerrank.com': 'HackerRank',
        'behance.net': 'Behance',
        'dev.to': 'DEV Community',
        'twitter.com': 'X',
        'x.com': 'X'
    }
    return domain_mapping.get(domain, '')

def transform_basics(basics_data: Dict) -> Dict:
    if not isinstance(basics_data, dict):
        return basics_data
        
    profiles = basics_data.get('profiles', [])
    
    transformed_profiles = []
    if isinstance(profiles, list):
        for i, profile in enumerate(profiles):
            if isinstance(profile, dict):
                transformed_profile = profile.copy()
                url = transformed_profile.get('url', '')
                network = transformed_profile.get('network')
                
                if url and network is None:
                    domain = extract_domain_from_url(url)
                    network_name = get_network_name(domain)
                    
                    if network_name:
                        transformed_profile['network'] = network_name
                        username = extract_username_from_url(url, domain)
                        if username:
                            transformed_profile['username'] = username
                transformed_profiles.append(transformed_profile)
    
    basics_data['profiles'] = transformed_profiles
    return basics_data


def extract_username_from_url(url: str, domain: str) -> str:
    try:
        path = url.split(domain)[1] if domain in url else ''
        if not path:
            return ''
        path = path.lstrip('/')
        
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
    transformed = []
    for item in work_list:
        if isinstance(item, dict):
            description = item.get('description', '')
            if isinstance(description, list):
                description = ' '.join(description)
            
            # Try to parse from 'startDate' if it contains a date range
            start_date_input = item.get('startDate', '')
            if start_date_input and any(month in start_date_input for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                start_date, end_date = parse_date_range(start_date_input)
            else:
                # Use existing startDate and endDate values
                start_date = item.get('startDate')
                end_date = item.get('endDate')
            
            transformed.append({
                'name': item.get('name', ''),
                'position': item.get('position', item.get('type', item.get('title', ''))),
                'url': item.get('url', None),
                'startDate': start_date,
                'endDate': end_date,
                'summary': item.get('summary', description),
                'highlights': item.get('highlights', [])
            })
    return transformed


def transform_organizations(org_list: List) -> List[Dict]:
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
    transformed = []
    for item in edu_list:
        if isinstance(item, dict):
            if 'degree' in item:
                score = item.get('gpa', item.get('percentage', None))
                if score is not None:
                    score = str(score)
                
                start_date, end_date = parse_date_range(item.get('years', ''))
                transformed.append({
                    'institution': item.get('institution', ''),
                    'url': item.get('url', None),
                    'area': item.get('degree', '').split(', ')[-1] if ',' in item.get('degree', '') else None,
                    'studyType': item.get('degree', '').split(', ')[0] if ',' in item.get('degree', '') else item.get('degree', ''),
                    'startDate': start_date,
                    'endDate': end_date,
                    'score': score,
                    'courses': []
                })
            else:
                transformed.append(item)
    return transformed


def transform_achievements(achievements_list: List) -> List[Dict]:
    transformed = []
    for item in achievements_list:
        if isinstance(item, dict):
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
    transformed = []
    for item in skills_list:
        if isinstance(item, dict):
            if 'category' in item:
                transformed.append({
                    'name': item.get('category', ''),
                    'level': None,
                    'keywords': item.get('keywords', [])
                })
            else:
                transformed.append(item)
    return transformed


def transform_projects(projects_list: List) -> List[Dict]:
    transformed = []
    for item in projects_list:
        if isinstance(item, dict):
            skills = []
            project_name = item.get('name', '')
            if '|' in project_name:
                name_parts = project_name.split('|')
                if len(name_parts) > 1:
                    skills_part = name_parts[1].strip()
                    skills = [skill.strip() for skill in skills_part.split(',')]
                    item['name'] = name_parts[0].strip()
            
            technologies = item.get('technologies', [])
            if isinstance(technologies, str):
                technologies = [tech.strip() for tech in technologies.split(',')]
            
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
    skills = []
    
    if 'skills' in parsed_data and isinstance(parsed_data['skills'], list):
        if parsed_data['skills'] and isinstance(parsed_data['skills'][0], str):
            skills.append({
                'name': 'Programming Languages',
                'level': None,
                'keywords': parsed_data['skills']
            })
        else:
            skills.extend(transform_skills(parsed_data['skills']))
    
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
    projects = []
    
    if 'projects' in parsed_data:
        projects.extend(transform_projects(parsed_data['projects']))
    
    if 'projectsOpenSource' in parsed_data:
        for item in parsed_data['projectsOpenSource']:
            if isinstance(item, dict):
                skills = []
                project_name = item.get('name', '')
                if '|' in project_name:
                    name_parts = project_name.split('|')
                    if len(name_parts) > 1:
                        skills_part = name_parts[1].strip()
                        skills = [skill.strip() for skill in skills_part.split(',')]
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


def parse_date_range(date_range: str) -> tuple:
    """
    Parse date range and return both start and end dates.
    For format like "Jan-Mar 2021", returns ("Jan 2021", "Mar 2021")
    """
    if not date_range:
        return None, None
    
    # Handle "onwards" case
    if 'onwards' in date_range:
        # Extract the start date from "onwards" format
        start_part = date_range.replace('onwards', '').strip()
        if start_part:
            return start_part, 'Present'
        return None, 'Present'
    
    # Handle format like "Jan-Mar 2021"
    if ' ' in date_range and any(month in date_range for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
        parts = date_range.split(' ')
        if len(parts) >= 2:
            year = parts[-1]
            month_map = {'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr', 'May': 'May', 'Jun': 'Jun',
                        'Jul': 'Jul', 'Aug': 'Aug', 'Sep': 'Sep', 'Oct': 'Oct', 'Nov': 'Nov', 'Dec': 'Dec'}
            
            # Check if it's a range like "Jan-Mar 2021"
            if '-' in parts[0] and len(parts[0].split('-')) == 2:
                start_month, end_month = parts[0].split('-')
                start_date = f"{month_map.get(start_month, start_month)} {year}"
                end_date = f"{month_map.get(end_month, end_month)} {year}"
                return start_date, end_date
            else:
                # Single month format like "Jan 2021"
                month = month_map.get(parts[0], parts[0])
                start_date = f"{month} {year}"
                return start_date, None
    
    # Handle year range like "2020-2021"
    if '-' in date_range and len(date_range.split('-')) == 2:
        start_year, end_year = date_range.split('-')
        start_date = f"{start_year}-01"
        end_date = f"{end_year}-12"
        return start_date, end_date
    
    return None, None


 