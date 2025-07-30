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

from models import JSONResume, Basics, Work, Education, Skill, Project, Award, BasicsSection, WorkSection, EducationSection, SkillsSection, ProjectsSection, AwardsSection
from pymupdf_rag import to_markdown
from typing import List, Optional, Dict
from prompt import (
    DEFAULT_MODEL,
    MODEL_PARAMETERS
)
from prompts.template_manager import TemplateManager
from transform import transform_parsed_data

# Configure logging to debug level
logging.basicConfig(
    level=logging.INFO,
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

    def _call_llm_for_section(self, section_name: str, text_content: str, prompt: str, return_model=None) -> Optional[Dict]:
        try:
            start_time = time.time()
            print(f"🔄 Extracting {section_name} section using {DEFAULT_MODEL}...")
            
            # Get model-specific parameters
            model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL, {
                'temperature': 0.7,
                'top_p': 0.9
            })
            
            # Get system message from template
            section_system_message = self.template_manager.render_system_message_template(section_name)
            if not section_system_message:
                print(f"❌ Failed to render system message template for {section_name}")
                return None
            
            # Prepare chat parameters
            chat_params = {
                'model': DEFAULT_MODEL,
                'messages': [
                    {
                        'role': 'system',
                        'content': section_system_message
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'options': {
                    'stream': False,
                    'temperature': model_params['temperature'],
                    'top_p': model_params['top_p']
                }
            }
            
            # Add format parameter if return_model is provided
            if return_model:
                chat_params['format'] = return_model.model_json_schema()
            
            response = ollama.chat(**chat_params)
            
            # Extract the response content
            response_text = response['message']['content']
            
            # Debug: Print the raw response for troubleshooting
            # print(f"🔍 Raw response for {section_name}: {response_text}...")
            
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

                # Transform the parsed data before returning
                transformed_data = transform_parsed_data(parsed_data)
                
                end_time = time.time()
                total_time = end_time - start_time
                print(f"\n⏱️ Total time for separate section extraction: {total_time:.2f} seconds")

                return transformed_data
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON for {section_name} section: {e}")
                print(f"Raw response: {response_text}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling LLM for {section_name} section: {e}")
            return None

    def extract_basics_section(self, resume_text: str) -> Optional[Dict]:
        prompt = self.template_manager.render_basics_template(resume_text)
        if not prompt:
            print("❌ Failed to render basics template")
            return None
        return self._call_llm_for_section("basics", resume_text, prompt, BasicsSection)

    def extract_work_section(self, resume_text: str) -> Optional[Dict]:
        prompt = self.template_manager.render_work_template(resume_text)
        if not prompt:
            print("❌ Failed to render work template")
            return None
        return self._call_llm_for_section("work", resume_text, prompt, WorkSection)

    def extract_education_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract education section from resume text.
        
        Args:
            resume_text (str): The resume text content
            return_model: Pydantic model for response validation (optional)
            
        Returns:
            Optional[Dict]: Education data
        """
        prompt = self.template_manager.render_education_template(resume_text)
        if not prompt:
            print("❌ Failed to render education template")
            return None
        return self._call_llm_for_section("education", resume_text, prompt, EducationSection)

    def extract_skills_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract skills section from resume text.
        
        Args:
            resume_text (str): The resume text content
            return_model: Pydantic model for response validation (optional)
            
        Returns:
            Optional[Dict]: Skills data
        """
        prompt = self.template_manager.render_skills_template(resume_text)
        if not prompt:
            print("❌ Failed to render skills template")
            return None
        return self._call_llm_for_section("skills", resume_text, prompt, SkillsSection)

    def extract_projects_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract projects section from resume text.
        
        Args:
            resume_text (str): The resume text content
            return_model: Pydantic model for response validation (optional)
            
        Returns:
            Optional[Dict]: Projects data
        """
        prompt = self.template_manager.render_projects_template(resume_text)
        if not prompt:
            print("❌ Failed to render projects template")
            return None
        return self._call_llm_for_section("projects", resume_text, prompt, ProjectsSection)

    def extract_awards_section(self, resume_text: str) -> Optional[Dict]:
        """
        Extract awards section from resume text.
        
        Args:
            resume_text (str): The resume text content
            return_model: Pydantic model for response validation (optional)
            
        Returns:
            Optional[Dict]: Awards data
        """
        prompt = self.template_manager.render_awards_template(resume_text)
        if not prompt:
            print("❌ Failed to render awards template")
            return None
        return self._call_llm_for_section("awards", resume_text, prompt, AwardsSection)

    def extract_json_from_text(self, resume_text: str) -> Optional[JSONResume]:
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

    def _extract_single_section(self, text_content: str, section_name: str, return_model=None) -> Optional[Dict]:
        """
        Extract a single section from resume text.
        
        Args:
            text_content (str): The resume text content
            section_name (str): Name of the section to extract
            return_model: Pydantic model for response validation (optional)
            
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
        
        section_data = section_extractors[section_name](text_content, return_model)
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
            logger.info(f"\n🔄 Extracting {section_name} section...")
            section_data = extractor_func(text_content)
            
            if section_data:
                # Merge the section data into the complete resume
                complete_resume.update(section_data)
                logger.info(f"✅ Successfully extracted {section_name} section")
            else:
                logger.info(f"⚠️ Failed to extract {section_name} section")
        
        # Note: Individual sections are already properly formatted, no need for additional transformation
        
        # Create JSONResume object
        try:
            # Ensure basics is properly formatted as a Basics object
            if complete_resume.get('basics') and isinstance(complete_resume['basics'], dict):
                try:
                    complete_resume['basics'] = Basics(**complete_resume['basics'])
                except Exception as e:
                    logger.error(f"❌ Error creating Basics object: {e}")
                    complete_resume['basics'] = None
            
            json_resume = JSONResume(**complete_resume)
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n⏱️ Total time for separate section extraction: {total_time:.2f} seconds")
            
            return json_resume
            
        except Exception as e:
            print(f"❌ Error creating JSONResume object: {e}")
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

    