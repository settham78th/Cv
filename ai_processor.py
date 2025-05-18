import os
import logging
import requests
import json
import time

logger = logging.getLogger(__name__)

# OpenRouter API configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/auto"  # OpenRouter will select the best available model

def process_with_openrouter(prompt, max_retries=3, retry_delay=2):
    """
    Process text with OpenRouter API.
    
    Args:
        prompt (str): The prompt to send to the AI
        max_retries (int): Maximum number of retry attempts for API calls
        retry_delay (int): Delay in seconds between retries
        
    Returns:
        str: The AI-generated response
        
    Raises:
        Exception: If the API call fails after maximum retries
    """
    if not API_KEY:
        logger.error("OPENROUTER_API_KEY environment variable is not set")
        raise Exception("OpenRouter API key is not configured. Please set the OPENROUTER_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com", # Replace with your actual site URL in production
        "X-Title": "PDF Text Processor"
    }
    
    data = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1000
    }
    
    attempt = 0
    
    while attempt < max_retries:
        attempt += 1
        logger.debug(f"Making OpenRouter API call, attempt {attempt}/{max_retries}")
        
        try:
            response = requests.post(API_URL, headers=headers, json=data)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Validate response structure
            if not result or "choices" not in result or not result["choices"]:
                logger.error(f"Invalid API response structure: {result}")
                raise Exception("Invalid response from OpenRouter API")
            
            # Extract the assistant's message
            message = result["choices"][0]["message"]
            if message and "content" in message:
                logger.info("Successfully received response from OpenRouter API")
                return message["content"]
            else:
                logger.error(f"Response missing content: {result}")
                raise Exception("Response from OpenRouter API is missing content")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt}/{max_retries}): {str(e)}")
            
            if attempt >= max_retries:
                logger.error("Maximum retry attempts reached for OpenRouter API", exc_info=True)
                raise Exception(f"Failed to communicate with OpenRouter API after {max_retries} attempts: {str(e)}")
            
            # Wait before retrying
            time.sleep(retry_delay)
            
        except Exception as e:
            logger.error(f"Error processing with OpenRouter: {str(e)}", exc_info=True)
            raise Exception(f"Error processing with AI: {str(e)}")
    
    # This should not be reached due to the exception in the loop, but just in case
    raise Exception("Failed to process with OpenRouter API due to unknown error")
