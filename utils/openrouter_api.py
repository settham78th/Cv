import os
import logging
import time
import json
import requests
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-3-opus:beta"  # Using a powerful model for accurate analysis

# Valid industries and job types for validation
VALID_INDUSTRIES = [
    "technology", "healthcare", "finance", "education", "manufacturing", 
    "retail", "hospitality", "construction", "energy", "media", 
    "telecommunications", "government", "nonprofit", "agriculture", 
    "transportation", "logistics", "legal", "consulting", "marketing",
    "real estate", "general"
]

VALID_JOB_TYPES = [
    "office", "remote", "hybrid", "physical", "field", "general"
]

def get_headers():
    """
    Get headers for OpenRouter API request including authentication.
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://cv-analyzer.example.com"  # Replace with your actual domain
    }

def call_openrouter_api(messages, temperature=0.3, max_retries=3, retry_delay=2):
    """
    Make a request to the OpenRouter API with retry mechanism.
    
    Args:
        messages (list): List of message objects for the chat completion
        temperature (float): Controls randomness of the output (0.0-1.0)
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        dict: API response or None if failed
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not found in environment variables")
        raise ValueError("OpenRouter API key not configured")
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    
    logger.debug("Sending request to OpenRouter API")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=get_headers(),
                json=payload,
                timeout=60  # 60 second timeout
            )
            
            response.raise_for_status()
            logger.debug("Received response from OpenRouter API")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                # Increase delay for each retry (exponential backoff)
                retry_delay *= 2
            else:
                logger.error(f"Failed to call OpenRouter API after {max_retries} attempts")
                raise Exception(f"OpenRouter API call failed: {str(e)}")

def analyze_seniority(text):
    """
    Analyze the CV to determine the seniority level.
    
    Args:
        text (str): Extracted text from CV
        
    Returns:
        str: Detected seniority level
    """
    messages = [
        {"role": "system", "content": "You are an expert CV and resume analyst. Your task is to determine the seniority level of the candidate based on their CV."},
        {"role": "user", "content": f"Analyze this CV and determine the seniority level. Only respond with one of these categories: entry-level, junior, mid-level, senior, executive, or unknown. Here's the CV text:\n\n{text[:4000]}"}
    ]
    
    try:
        response = call_openrouter_api(messages)
        seniority = response["choices"][0]["message"]["content"].strip().lower()
        
        # Validate and normalize response
        valid_seniorities = ["entry-level", "junior", "mid-level", "senior", "executive", "unknown"]
        
        if seniority not in valid_seniorities:
            logger.warning(f"Invalid seniority detected: \"{seniority}\". Using 'unknown' as default.")
            seniority = "unknown"
        
        logger.info(f"Detected seniority: {seniority}")
        return seniority
        
    except Exception as e:
        logger.error(f"Error analyzing seniority: {str(e)}")
        return "unknown"

def analyze_industry(text):
    """
    Analyze the CV to determine the industry.
    
    Args:
        text (str): Extracted text from CV
        
    Returns:
        str: Detected industry
    """
    messages = [
        {"role": "system", "content": "You are an expert CV and resume analyst. Your task is to determine the industry that the candidate is applying for based on their CV."},
        {"role": "user", "content": f"Analyze this CV and determine the industry. Only respond with one word from this list: technology, healthcare, finance, education, manufacturing, retail, hospitality, construction, energy, media, telecommunications, government, nonprofit, agriculture, transportation, logistics, legal, consulting, marketing, real estate. If you cannot determine, respond with 'general'. Here's the CV text:\n\n{text[:4000]}"}
    ]
    
    try:
        response = call_openrouter_api(messages)
        industry = response["choices"][0]["message"]["content"].strip().lower()
        
        # Validate the response against valid industries
        if industry not in VALID_INDUSTRIES:
            logger.warning(f"Invalid industry detected: \"{industry}\". Using 'general' as default.")
            industry = "general"
        
        logger.info(f"Detected industry: {industry}")
        return industry
        
    except Exception as e:
        logger.error(f"Error analyzing industry: {str(e)}")
        return "general"

def analyze_job_type(text):
    """
    Analyze the CV to determine the job type.
    
    Args:
        text (str): Extracted text from CV
        
    Returns:
        str: Detected job type
    """
    messages = [
        {"role": "system", "content": "You are an expert CV and resume analyst. Your task is to determine the job type the candidate is seeking based on their CV."},
        {"role": "user", "content": f"Analyze this CV and determine the job type. Only respond with one of these categories: office, remote, hybrid, physical, field. If you cannot determine, respond with 'general'. Here's the CV text:\n\n{text[:4000]}"}
    ]
    
    try:
        response = call_openrouter_api(messages)
        job_type = response["choices"][0]["message"]["content"].strip().lower()
        
        # Validate the response
        if job_type not in VALID_JOB_TYPES:
            logger.warning(f"Invalid job type detected: \"{job_type}\". Using 'general' as default.")
            job_type = "general"
        
        logger.info(f"Detected job type: {job_type}")
        return job_type
        
    except Exception as e:
        logger.error(f"Error analyzing job type: {str(e)}")
        return "general"

def analyze_specific_role(text):
    """
    Analyze the CV to determine the specific role.
    
    Args:
        text (str): Extracted text from CV
        
    Returns:
        str: Detected specific role
    """
    messages = [
        {"role": "system", "content": "You are an expert CV and resume analyst. Your task is to determine the specific job role or title the candidate is seeking based on their CV. Handle multilingual CVs and extract the job title in its original language if possible."},
        {"role": "user", "content": f"Analyze this CV and determine the specific job role or title the candidate is seeking. Respond with only the job title. If you're unsure, respond with 'unknown'. Here's the CV text:\n\n{text[:4000]}"}
    ]
    
    try:
        response = call_openrouter_api(messages)
        specific_role = response["choices"][0]["message"]["content"].strip()
        
        logger.info(f"Detected specific role: {specific_role}")
        return specific_role
        
    except Exception as e:
        logger.error(f"Error analyzing specific role: {str(e)}")
        return "unknown"
