"""
PDF Processing Module

This module provides functionality for extracting text and links from PDF files
using PyMuPDF (fitz) library.
"""

import os
import sys
import json
import time
import ollama
import logging
import pymupdf

from models import JSONResume
from pymupdf_rag import to_markdown
from typing import List, Optional, Dict
from prompt import (
    JSON_RESUME_EXTRACTION_PROMPT,
    JSON_RESUME_EXTRACTION_SYSTEM_MESSAGE,
    DEFAULT_MODEL,
    MODEL_PARAMETERS
)
from prompts.template_manager import TemplateManager

# Configure logging to debug level
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(lineno)d - %(funcName)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class PDFHandler:
    """
    Handler class for PDF processing operations.
    
    This class provides comprehensive functionality for extracting text and links
    from PDF files using PyMuPDF (fitz) library. It offers superior link extraction
    and text processing capabilities compared to other PDF libraries.
    
    Attributes:
        None (stateless class)
    
    Example:
        >>> handler = PDFHandler()
        >>> text = handler.extract_text_from_pdf("resume.pdf")
        >>> print(f"Extracted {len(text)} characters")
    """
    
    def __init__(self):
        """Initialize the PDFHandler with template manager."""
        self.template_manager = TemplateManager()

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text content from a PDF file with links preserved.
        
        This is the main entry point for PDF text extraction. It validates the file,
        extracts text and hyperlinks using PyMuPDF (fitz), and returns the combined content.
        
        Args:
            pdf_path (str): Path to the PDF file to process
            
        Returns:
            Optional[str]: Extracted text content with URLs preserved, or None if extraction failed
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the PDF file is corrupted or encrypted
            Exception: For other PDF processing errors
            
        Example:
            >>> handler = PDFHandler()
            >>> text = handler.extract_text_from_pdf("resume.pdf")
            >>> if text:
            ...     print(f"Successfully extracted {len(text)} characters")
            ... else:
            ...     print("Failed to extract text")
        """
        try:
            # Validate PDF file
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Extract text and links using PyMuPDF (fitz)
            doc = pymupdf.open(pdf_path)
            pages = range(doc.page_count)
            resume_text = to_markdown(
                doc,
                pages=pages,
            )
            logger.debug(f"Extracted text from PDF: {len(resume_text) if resume_text else 0} characters")
            return resume_text
        except Exception as e:
            print(f"An error occurred while reading the PDF: {e}")
            return None

    def _call_llm_for_section(self, section_name: str, text_content: str, prompt: str) -> Optional[Dict]:
        """
        Call LLM for a specific section extraction.
        
        Args:
            section_name (str): Name of the section being extracted
            text_content (str): The resume text content
            prompt (str): Section-specific prompt
            
        Returns:
            Optional[Dict]: Parsed JSON data for the section, or None if failed
        """
        try:
            start_time = time.time()
            print(f"🔄 Extracting {section_name} section using {DEFAULT_MODEL}...")
            
            # Get model-specific parameters
            model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL, {
                'temperature': 0.7,
                'top_p': 0.9
            })
            
            # Create a simplified system message for section-specific extraction
            section_system_message = f"""You are an expert resume parser. Extract ONLY the {section_name} section from resumes and format it according to the JSON Resume specification.

**CRITICAL: You must respond with ONLY valid JSON. Do not include any explanatory text, thinking process, markdown formatting, or <think> tags. Return ONLY the JSON object.**

Return ONLY the {section_name} section in JSON format."""
            
            response = ollama.chat(
                model=DEFAULT_MODEL,
                messages=[
                    {
                        'role': 'system',
                        'content': section_system_message
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'stream': False,
                    'temperature': model_params['temperature'],
                    'top_p': model_params['top_p']
                }
            )
            
            # Extract the response content
            response_text = response['message']['content']
            
            # Debug: Print the raw response for troubleshooting
            print(f"🔍 Raw response for {section_name}: {response_text[:200]}...")
            
            # Clean and parse JSON response
            try:
                # Clean the response to extract JSON
                response_text = response_text.strip()
                
                # Remove any <think> tags and content
                if '<think>' in response_text:
                    think_start = response_text.find('<think>')
                    think_end = response_text.find('</think>')
                    if think_start != -1 and think_end != -1:
                        response_text = response_text[:think_start] + response_text[think_end + 8:]
                
                # Remove markdown code blocks
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                elif response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                # Clean up any remaining markdown or explanatory text
                response_text = response_text.strip()
                
                # Try to find JSON object boundaries
                json_start = response_text.find('{')
                json_end = response_text.rfind('}')
                
                if json_start != -1 and json_end != -1:
                    response_text = response_text[json_start:json_end + 1]
                
                # Parse JSON
                parsed_data = json.loads(response_text)
                print(f"✅ Successfully extracted {section_name} section")

                end_time = time.time()
                total_time = end_time - start_time
                print(f"\n⏱️ Total time for separate section extraction: {total_time:.2f} seconds")

                return parsed_data
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON for {section_name} section: {e}")
                print(f"Raw response: {response_text}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling LLM for {section_name} section: {e}")
            return None

    def extract_basics_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract basic information section from resume text.
        
        Args:
            resume_text (str): The resume text content
            
        Returns:
            Optional[Dict]: Basic information data
        """
        prompt = self.template_manager.render_basics_template(resume_text)
        if not prompt:
            print("❌ Failed to render basics template")
            return None
        return self._call_llm_for_section("basics", resume_text, prompt)

    def extract_work_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract work experience section from resume text.
        
        Args:
            resume_text (str): The resume text content
            
        Returns:
            Optional[Dict]: Work experience data
        """
        prompt = self.template_manager.render_work_template(resume_text)
        if not prompt:
            print("❌ Failed to render work template")
            return None
        return self._call_llm_for_section("work", resume_text, prompt)

    def extract_education_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract education section from resume text.
        
        Args:
            resume_text (str): The resume text content
            
        Returns:
            Optional[Dict]: Education data
        """
        prompt = self.template_manager.render_education_template(resume_text)
        if not prompt:
            print("❌ Failed to render education template")
            return None
        return self._call_llm_for_section("education", resume_text, prompt)

    def extract_skills_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract skills section from resume text.
        
        Args:
            resume_text (str): The resume text content
            
        Returns:
            Optional[Dict]: Skills data
        """
        prompt = self.template_manager.render_skills_template(resume_text)
        if not prompt:
            print("❌ Failed to render skills template")
            return None
        return self._call_llm_for_section("skills", resume_text, prompt)

    def extract_projects_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract projects section from resume text.
        
        Args:
            resume_text (str): The resume text content
            
        Returns:
            Optional[Dict]: Projects data
        """
        prompt = self.template_manager.render_projects_template(resume_text)
        if not prompt:
            print("❌ Failed to render projects template")
            return None
        return self._call_llm_for_section("projects", resume_text, prompt)

    def extract_awards_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract awards section from resume text.
        
        Args:
            resume_text (str): The resume text content
            
        Returns:
            Optional[Dict]: Awards data
        """
        prompt = self.template_manager.render_awards_template(resume_text)
        if not prompt:
            print("❌ Failed to render awards template")
            return None
        return self._call_llm_for_section("awards", resume_text, prompt)

    def extract_json_from_text(self, resume_text: str) -> Optional[JSONResume]:
        """
        Extract comprehensive resume data in JSON Resume format using LLM.
        
        This method uses Ollama LLM to parse extracted text and convert it into
        structured JSON Resume format. It handles JSON parsing, data transformation,
        and error recovery for malformed LLM responses.
        
        Args:
            resume_text (str): Extracted text from PDF
            
        Returns:
            Optional[JSONResume]: Complete resume data in JSON Resume format, or None if extraction failed
            
        Raises:
            json.JSONDecodeError: If the LLM response cannot be parsed as JSON
            Exception: For other LLM processing errors
            
        Example:
            >>> handler = PDFHandler()
            >>> text = handler.extract_text_from_pdf("resume.pdf")
            >>> if text:
            ...     resume_data = handler.extract_json_from_text(text)
            ...     if resume_data:
            ...         print(f"Successfully extracted resume for: {resume_data.basics.name}")
        """
        try:
            return self._extract_all_sections_separately(resume_text)
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None

    def extract_json_from_pdf(self, pdf_path: str) -> Optional[JSONResume]:
        """
        Extract comprehensive resume data in JSON Resume format directly from a PDF file.
        
        This method combines text extraction and JSON conversion into a single convenient
        function call. It extracts all sections separately with individual LLM calls
        for better accuracy and reliability.
        
        Args:
            pdf_path (str): Path to the PDF file to process
            
        Returns:
            Optional[JSONResume]: Complete resume data in JSON Resume format, or None if extraction failed
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the PDF file is corrupted or encrypted
            Exception: For other PDF processing errors
            
        Example:
            >>> handler = PDFHandler()
            >>> resume_data = handler.extract_json_from_pdf("resume.pdf")
            >>> if resume_data:
            ...     print(f"Successfully extracted resume for: {resume_data.basics.name}")
        """
        try:
            # First extract text from the PDF
            print(f"📄 Extracting text from PDF: {pdf_path}")
            text_content = self.extract_text_from_pdf(pdf_path)
            
            if not text_content:
                print("❌ Failed to extract text from PDF")
                return None
            
            print(f"✅ Successfully extracted {len(text_content)} characters from PDF")
            
            print("🔄 Extracting all sections separately...")
            return self._extract_all_sections_separately(text_content)

                
        except Exception as e:
            print(f"❌ Error during PDF to JSON extraction: {e}")
            return None

    def _extract_single_section(self, text_content: str, section_name: str) -> Optional[Dict]:
        """
        Extract a single section from resume text.
        
        Args:
            text_content (str): The resume text content
            section_name (str): Name of the section to extract
            
        Returns:
            Optional[Dict]: Complete resume data with only the specified section populated
        """
        section_extractors = {
            'basics': self.extract_basics_section,
            'work': self.extract_work_section,
            'education': self.extract_education_section,
            'skills': self.extract_skills_section,
            'projects': self.extract_projects_section,
            'awards': self.extract_awards_section
        }
        
        if section_name not in section_extractors:
            print(f"❌ Invalid section name: {section_name}")
            print(f"Valid sections: {list(section_extractors.keys())}")
            return None
        
        section_data = section_extractors[section_name](text_content)
        if section_data:
            # Create a complete resume structure with only the specified section
            complete_resume = {
                'basics': None,
                'work': None,
                'volunteer': None,
                'education': None,
                'awards': None,
                'certificates': None,
                'publications': None,
                'skills': None,
                'languages': None,
                'interests': None,
                'references': None,
                'projects': None,
                'meta': None
            }
            
            # Merge the extracted section data
            complete_resume.update(section_data)
            return complete_resume
        
        return None

    def _extract_all_sections_separately(self, text_content: str) -> Optional[JSONResume]:
        """
        Extract all sections separately and merge them into a complete JSON Resume.
        
        Args:
            text_content (str): The resume text content
            
        Returns:
            Optional[JSONResume]: Complete resume data with all sections
        """
        start_time = time.time()
        
        # Define all sections to extract
        sections = [
            ('basics', self.extract_basics_section),
            ('work', self.extract_work_section),
            ('education', self.extract_education_section),
            ('skills', self.extract_skills_section),
            ('projects', self.extract_projects_section),
            ('awards', self.extract_awards_section)
        ]
        
        # Initialize complete resume structure
        complete_resume = {
            'basics': None,
            'work': None,
            'volunteer': None,
            'education': None,
            'awards': None,
            'certificates': None,
            'publications': None,
            'skills': None,
            'languages': None,
            'interests': None,
            'references': None,
            'projects': None,
            'meta': None
        }
        
        # Extract each section
        for section_name, extractor_func in sections:
            print(f"\n🔄 Extracting {section_name} section...")
            section_data = extractor_func(text_content)
            
            if section_data:
                # Merge the section data into the complete resume
                complete_resume.update(section_data)
                print(f"✅ Successfully extracted {section_name} section")
            else:
                print(f"⚠️ Failed to extract {section_name} section")
        
        # Transform the data to handle common LLM response format issues
        complete_resume = self._transform_parsed_data(complete_resume)
        
        # Create JSONResume object
        try:
            json_resume = JSONResume(**complete_resume)
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n⏱️ Total time for separate section extraction: {total_time:.2f} seconds")
            
            return json_resume
            
        except Exception as e:
            print(f"❌ Error creating JSONResume object: {e}")
            return None

    def _transform_parsed_data(self, parsed_data: Dict) -> Dict:
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
                # Map LLM response format to expected schema
                transformed = {
                    'basics': parsed_data.get('basics', {}),
                    'work': self._transform_work_experience(parsed_data.get('work_experience', parsed_data.get('work', parsed_data.get('experience', [])))),
                    'volunteer': self._transform_organizations(parsed_data.get('organizations', [])),
                    'education': self._transform_education(parsed_data.get('education', [])),
                    'awards': self._transform_achievements(parsed_data.get('achievements', parsed_data.get('awards', parsed_data.get('honors_and_awards', [])))),
                    'certificates': parsed_data.get('certificates', []),
                    'publications': parsed_data.get('publications', []),
                    'skills': self._transform_skills_comprehensive(parsed_data),
                    'languages': parsed_data.get('languages', []),
                    'interests': parsed_data.get('interests', []),
                    'references': parsed_data.get('references', []),
                    'projects': self._transform_projects_comprehensive(parsed_data),
                    'meta': parsed_data.get('meta', {})
                }
                
                return transformed
            else:
                return parsed_data
                
        except Exception as e:
            print(f"Error transforming parsed data: {e}")
            return parsed_data

    def _transform_work_experience(self, work_list: List) -> List[Dict]:
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
                    'startDate': self._parse_date_range(item.get('years', '')) if 'years' in item else item.get('startDate'),
                    'endDate': self._parse_end_date(item.get('years', '')) if 'years' in item else item.get('endDate'),
                    'summary': item.get('summary', description),
                    'highlights': item.get('highlights', [])
                })
        return transformed

    def _transform_organizations(self, org_list: List) -> List[Dict]:
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

    def _transform_education(self, edu_list: List) -> List[Dict]:
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
                        'startDate': self._parse_date_range(item.get('years', '')),
                        'endDate': self._parse_end_date(item.get('years', '')),
                        'score': score,
                        'courses': []
                    })
                else:
                    # Original format
                    transformed.append(item)
        return transformed

    def _transform_achievements(self, achievements_list: List) -> List[Dict]:
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

    def _transform_skills(self, skills_list: List) -> List[Dict]:
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

    def _transform_projects(self, projects_list: List) -> List[Dict]:
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

    def _transform_skills_comprehensive(self, parsed_data: Dict) -> List[Dict]:
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
                skills.extend(self._transform_skills(parsed_data['skills']))
        
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

    def _transform_projects_comprehensive(self, parsed_data: Dict) -> List[Dict]:
        """Transform projects from various possible formats."""
        projects = []
        
        # Handle standard projects
        if 'projects' in parsed_data:
            projects.extend(self._transform_projects(parsed_data['projects']))
        
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

    def _parse_date_range(self, date_range: str) -> str:
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

    def _parse_end_date(self, date_range: str) -> str:
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

def main():
    """
    Main function to test PDF processing functionality.
    
    This function provides a command-line interface for testing the PDF processing
    capabilities. It accepts a PDF file path as a command-line argument and
    demonstrates the text extraction functionality with detailed output.
    
    Usage:
        python pdf.py [pdf_file_path]
        
    Args:
        Command line arguments:
            pdf_file_path (optional): Path to the PDF file to test. 
                                    Defaults to "sample.pdf" if not provided.
    """
    # Create PDFHandler instance for testing
    pdf_handler = PDFHandler()
    
    # Get PDF path from command line arguments or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "sample.pdf"  # Default test file
        print(f"No PDF path provided. Using default: {pdf_path}")
    
    try:
        # Extract text from PDF and print it
        extracted_text = pdf_handler.extract_text_from_pdf(pdf_path)
        if extracted_text:
            print(extracted_text)
        else:
            print("❌ Failed to extract text from PDF")
   
        resume_data = pdf_handler.extract_json_from_pdf(pdf_path)
        if resume_data:
            print(json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False))
        else:
            print("❌ Failed to extract JSON resume data")
        
    except Exception as e:
        print(f"❌ Error during PDF processing: {e}")
        return


if __name__ == "__main__":
    main() 

    