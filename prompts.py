"""
Prompts for Resume Evaluation System

This module contains all the prompts used by the resume evaluation system.
Centralizing prompts here makes them easier to maintain and update.
"""

# Constants
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_TEMPERATURE = 0.3

# JSON Resume extraction prompt
JSON_RESUME_EXTRACTION_PROMPT = """
Please analyze the following resume text and extract all information in JSON Resume format.

**CRITICAL INSTRUCTION**: You MUST extract ALL projects mentioned in the resume. Count every bullet point (●) in the projects section and create a separate project entry for each one. Do NOT skip any projects.

Resume Text:
{text_content}

**IMPORTANT INSTRUCTIONS FOR LINK EXTRACTION:**
1. Look for links formatted as "Text [URL]" in the resume (e.g., "Email [mailto:...]", "GitHub [https://...]")
2. Extract email addresses from "mailto:" links and put them in basics.email
3. Extract social profiles from URLs and put them in basics.profiles array:
   - GitHub links → network: "GitHub", username: extract username from URL. 
   - Github links -> URL: The URL SHOULD of the format "https://github.com/username". For Github URLS like "https://github.com/username/projectname", ignore the project name when generating profile URL.
   - LinkedIn links → network: "LinkedIn", username: extract username from URL  
   - Medium links → network: "Medium", username: extract username from URL
   - Blog links → network: "Blog", username: extract domain/username
   - Stack Overflow links → network: "Stack Overflow", username: extract username from URL
   - LeetCode links → network: "LeetCode", username: extract username from URL
4. Extract programming languages mentioned in the resume and put them in languages array
5. **CRITICAL**: Extract ALL projects mentioned in the resume and put them in projects array. Look for:
   - Projects under "PROJECTS" section
   - Projects under "PROJECTS AND GOOGLE CLOUD" section  
   - Projects mentioned in work experience
   - Projects mentioned in volunteer experience
   - Open source contributions and community projects
   - All projects with bullet points (●) or numbered lists
   - **MANDATORY**: You MUST extract EVERY project mentioned in the resume. Do NOT skip any projects.
   - **MANDATORY**: If you see bullet points (●) in the projects section, each bullet point is a separate project.
   - **MANDATORY**: Count the number of bullet points in the projects section and ensure you extract that many projects.
   - **MANDATORY**: Do not combine multiple projects into one - each bullet point is a separate project.

Please respond with a complete JSON Resume object containing all available information:
{{
    "basics": {{
        "name": "Full name",
        "email": "Email address (extract from mailto: links)",
        "phone": "Phone number",
        "url": "Personal website URL",
        "summary": "Professional summary",
        "location": {{
            "address": "Street address",
            "postalCode": "Postal code",
            "city": "City",
            "countryCode": "Country code",
            "region": "State/region"
        }},
        "profiles": [
            {{
                "network": "Network name (GitHub, LinkedIn, Medium, Blog, Stack Overflow, LeetCode)",
                "username": "Username extracted from URL",
                "url": "Profile URL"
            }}
        ]
    }},
    "work": [
        {{
            "name": "Company name (extract the FULL company name including programs like 'Google Summer of Code', 'Outreachy', 'Season of Docs')",
            "position": "Job title",
            "url": "Company website",
            "startDate": "Start date (YYYY-MM-DD)",
            "endDate": "End date (YYYY-MM-DD) or 'Present'",
            "summary": "Job description",
            "highlights": ["Key achievement 1", "Key achievement 2"]
        }}
    ],
    "volunteer": [
        {{
            "organization": "Organization name",
            "position": "Volunteer role",
            "url": "Organization website",
            "startDate": "Start date",
            "endDate": "End date",
            "summary": "Description",
            "highlights": ["Achievement 1", "Achievement 2"]
        }}
    ],
    "awards": [
        {{
            "title": "Award name",
            "date": "Award date",
            "awarder": "Awarding organization",
            "summary": "Award description"
        }}
    ],
    "certificates": [
        {{
            "name": "Certificate name",
            "date": "Issue date",
            "issuer": "Issuing organization",
            "url": "Certificate URL"
        }}
    ],
    "publications": [
        {{
            "name": "Publication title",
            "publisher": "Publisher name",
            "releaseDate": "Release date",
            "url": "Publication URL",
            "summary": "Publication description"
        }}
    ],
    "skills": [
        {{
            "name": "Skill category",
            "level": "Proficiency level",
            "keywords": ["Skill 1", "Skill 2", "Skill 3"]
        }}
    ],
    "languages": [
        {{
            "language": "Programming language name (Python, JavaScript, Java, etc.)",
            "fluency": "Proficiency level (Beginner, Intermediate, Advanced, Expert)"
        }}
    ],
    "projects": [
        {{
            "name": "Project name",
            "startDate": "Start date",
            "endDate": "End date",
            "description": "Project description",
            "highlights": ["Key feature 1", "Key feature 2"],
            "url": "Project URL (extract from resume links)"
        }}
    ]
    **MANDATORY**: Extract ALL projects from the resume. Each bullet point (●) in the projects section is a separate project. If you see multiple projects in the resume, you MUST extract ALL projects into separate projects, one for each bullet point.
}}

**CRITICAL REQUIREMENTS:**
1. Extract ALL links formatted as "Text [URL]" and properly categorize them
2. For GitHub URLs like "https://github.com/username", extract username as "username". 
3. For Github URLS like "https://github.com/username/projectname", extract ONLY username as "username". Ignore the project name from the URL. The URL SHOULD of the format "https://github.com/username"
3. For LinkedIn URLs like "https://www.linkedin.com/in/username", extract username as "username"
4. For Medium URLs like "https://medium.com/@username", extract username as "username"
5. **CRITICAL**: When extracting company names, extract the FULL company name including special programs:
   - "Google Summer of Code" should be extracted as "Google Summer of Code" (NOT just "Google")
   - "Girl Script Summer of Code" and "Google Summer of Code" are two different programs. Don't get confused between those two.
   - "Outreachy" should be extracted as "Outreachy"
   - "Season of Docs" should be extracted as "Season of Docs"
   - "Google Code-in" should be extracted as "Google Code-in"
   - Any other special programs should be extracted with their full names
6. Extract programming languages mentioned in skills section and put them in languages array
7. Extract hobbies and interests mentioned in the resume and put them in interests array. Look for:
   - Hobbies section
   - Interests section  
   - Personal interests mentioned anywhere in the resume
   - Activities and pastimes mentioned
   - **IMPORTANT**: Look for comma-separated lists of activities like "Contributing to Open Source, Traveling, Photography"
   - **IMPORTANT**: These are often found near the end of resumes or in personal sections
   - **IMPORTANT**: Split comma-separated hobbies into individual interest entries
7. **MOST IMPORTANT**: Extract ALL projects mentioned in the resume - do not skip any projects. Look for:
   - Every project with bullet points (●)
   - Projects in "PROJECTS" or "PROJECTS AND GOOGLE CLOUD" sections
   - Open source contributions and community projects
   - Work experience projects
   - Volunteer experience projects
   - All projects mentioned anywhere in the resume
   - **MANDATORY**: Count bullet points (●) in projects section and extract that many projects
   - **MANDATORY**: Each bullet point = one separate project
   - **MANDATORY**: Do not combine projects - extract each bullet point as individual project

Extract all available information from the resume. If a section is not present, omit it from the JSON.
Respond only with valid JSON, no additional text.
"""

JSON_RESUME_EXTRACTION_SYSTEM_MESSAGE = """You are an expert resume parser. Extract all information from resumes and format it according to the JSON Resume specification. Be thorough and accurate.

**CRITICAL FORMAT REQUIREMENTS:**
1. All arrays must contain objects, not strings
2. For interests: Use {"name": "interest_name", "keywords": null} format, not just strings
3. For skills: Use {"name": "skill_name", "level": "level", "keywords": ["keyword1", "keyword2"]} format
4. For languages: Use {"language": "language_name", "fluency": "level"} format
5. For projects: Use {"name": "project_name", "description": "description", "url": "url"} format
6. Always return valid JSON with proper object structures
7. Never return arrays of strings - always use object format with proper fields
8. **CRITICAL**: When extracting company names, extract the FULL company name including special programs:
   - "Google Summer of Code" should be extracted as "Google Summer of Code" (NOT just "Google")
   - "Girl Script Summer of Code" should be extracted as "Girl Script Summer of Code" (NOT just "Girl Script")
   - "Outreachy" should be extracted as "Outreachy"
   - "Season of Docs" should be extracted as "Season of Docs"
   - "Google Code-in" should be extracted as "Google Code-in"
   - Any other special programs should be extracted with their full names"""