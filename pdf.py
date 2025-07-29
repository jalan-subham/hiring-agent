"""
PDF Processing Module

This module provides functionality for extracting text and links from PDF files
using PyMuPDF (fitz) library.
"""

import os
import sys
import json
import ollama
import logging
import pymupdf

from models import JSONResume
from pymupdf_rag import to_markdown
from typing import List, Optional, Dict
from prompts import (
    JSON_RESUME_EXTRACTION_PROMPT,
    JSON_RESUME_EXTRACTION_SYSTEM_MESSAGE,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE
)

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
            response = self._extract_with_fitz(pdf_path)
            logger.debug(f"Extracted text from PDF: {len(response) if response else 0} characters")
            return response
        except Exception as e:
            print(f"An error occurred while reading the PDF: {e}")
            return None
    
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
            # Prepare the comprehensive prompt for JSON Resume extraction
            prompt = JSON_RESUME_EXTRACTION_PROMPT.format(text_content=resume_text)
            
            print(f"🪪 Extracting comprehensive resume data using {DEFAULT_MODEL} model...")
            
            response = ollama.chat(
                model=DEFAULT_MODEL,
                messages=[
                    {
                        'role': 'system',
                        'content': JSON_RESUME_EXTRACTION_SYSTEM_MESSAGE
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': DEFAULT_TEMPERATURE,
                    'top_p': 0.9
                }
            )
            
            # Extract the response content
            response_text = response['message']['content']
            
            # Try to parse JSON from the response
            try:
                # Clean the response to extract JSON
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                # Parse JSON
                parsed_data = json.loads(response_text)
                
                # Transform the data to handle common LLM response format issues
                parsed_data = self._transform_parsed_data(parsed_data)
                
                # Create JSONResume object
                json_resume = JSONResume(**parsed_data)
                
                return json_resume
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Raw response: {response_text}")
                
                # Fallback: try to extract basic information using simple text analysis
                return None
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None

    def extract_json_from_pdf(self, pdf_path: str) -> Optional[JSONResume]:
        """
        Extract comprehensive resume data in JSON Resume format directly from a PDF file.
        
        This method combines text extraction and JSON conversion into a single convenient
        function call. It first extracts text from the PDF, then converts it to JSON
        Resume format using LLM processing.
        
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
            ... else:
            ...     print("Failed to extract resume data")
        """
        try:
            # First extract text from the PDF
            print(f"📄 Extracting text from PDF: {pdf_path}")
            text_content = self.extract_text_from_pdf(pdf_path)
            
            if not text_content:
                print("❌ Failed to extract text from PDF")
                return None
            
            print(f"✅ Successfully extracted {len(text_content)} characters from PDF")
            
            # Then convert the text to JSON Resume format
            json_resume = self.extract_json_from_text(text_content)
            
            if json_resume:
                print(f"✅ Successfully converted to JSON Resume format")
                return json_resume
            else:
                print("❌ Failed to convert text to JSON Resume format")
                return None
                
        except Exception as e:
            print(f"❌ Error during PDF to JSON extraction: {e}")
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
                # Ensure all required fields exist with default values
                transformed = {
                    'basics': parsed_data.get('basics', {}),
                    'work': parsed_data.get('work', []),
                    'volunteer': parsed_data.get('volunteer', []),
                    'education': parsed_data.get('education', []),
                    'awards': parsed_data.get('awards', []),
                    'certificates': parsed_data.get('certificates', []),
                    'publications': parsed_data.get('publications', []),
                    'skills': parsed_data.get('skills', []),
                    'languages': parsed_data.get('languages', []),
                    'interests': parsed_data.get('interests', []),
                    'references': parsed_data.get('references', []),
                    'projects': parsed_data.get('projects', []),
                    'meta': parsed_data.get('meta', {})
                }
                
                # Clean up nested structures and handle string-to-object conversions
                for key in ['work', 'volunteer', 'education', 'awards', 'certificates', 
                           'publications', 'skills', 'languages', 'interests', 'references', 'projects']:
                    if key in transformed:
                        if not isinstance(transformed[key], list):
                            transformed[key] = []
                        else:
                            # Handle cases where LLM returns strings instead of objects
                            if key == 'interests':
                                transformed[key] = self._convert_strings_to_interests(transformed[key])
                            elif key == 'skills':
                                transformed[key] = self._convert_strings_to_skills(transformed[key])
                            elif key == 'languages':
                                transformed[key] = self._convert_strings_to_languages(transformed[key])
                
                return transformed
            else:
                return parsed_data
                
        except Exception as e:
            print(f"Error transforming parsed data: {e}")
            return parsed_data

    def _convert_strings_to_interests(self, interests_list: List) -> List[Dict]:
        """
        Convert string interests to proper Interest objects.
        
        Args:
            interests_list (List): List of interests that may contain strings or dicts
            
        Returns:
            List[Dict]: List of properly formatted interest objects
        """
        converted = []
        for item in interests_list:
            if isinstance(item, str):
                converted.append({'name': item, 'keywords': []})
            elif isinstance(item, dict):
                converted.append(item)
        return converted

    def _convert_strings_to_skills(self, skills_list: List) -> List[Dict]:
        """
        Convert string skills to proper Skill objects.
        
        Args:
            skills_list (List): List of skills that may contain strings or dicts
            
        Returns:
            List[Dict]: List of properly formatted skill objects
        """
        converted = []
        for item in skills_list:
            if isinstance(item, str):
                converted.append({'name': item, 'level': None, 'keywords': []})
            elif isinstance(item, dict):
                converted.append(item)
        return converted

    def _convert_strings_to_languages(self, languages_list: List) -> List[Dict]:
        """
        Convert string languages to proper Language objects.
        
        Args:
            languages_list (List): List of languages that may contain strings or dicts
            
        Returns:
            List[Dict]: List of properly formatted language objects
        """
        converted = []
        for item in languages_list:
            if isinstance(item, str):
                converted.append({'language': item, 'fluency': None})
            elif isinstance(item, dict):
                converted.append(item)
        return converted
    
    def _extract_with_fitz(self, pdf_path: str) -> Optional[str]:
        """
        Extract text and links using PyMuPDF (fitz) for better accuracy.
        
        This method uses PyMuPDF to extract text content from PDF files with
        preserved formatting and structure. It leverages the to_markdown function
        for enhanced text extraction capabilities.
        
        Args:
            pdf_path (str): Path to the PDF file to process
            
        Returns:
            Optional[str]: Extracted text content in markdown format, or None if failed
            
        Raises:
            Exception: For PDF processing errors
        """
        try:
            doc = pymupdf.open(pdf_path)
            pages = range(doc.page_count)
            resume_text = to_markdown(
                doc,
                pages=pages,
            )
            return resume_text
                
        except Exception as e:
            print(f"Fitz extraction failed: {e}")
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
        
        # Uncomment the following lines to test JSON extraction
        # resume_data = pdf_handler.extract_json_from_pdf(pdf_path)
        # if resume_data:
        #     logger.info(json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False))
        # else:
        #     print("❌ Failed to extract JSON resume data")
        
    except Exception as e:
        print(f"❌ Error during PDF processing: {e}")
        return


if __name__ == "__main__":
    main() 

    