import logging
from io import StringIO
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfminer-six.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        Exception: If text extraction fails
    """
    try:
        logger.debug(f"Starting text extraction from {pdf_path}")
        
        # First, try the simplified approach
        try:
            text = extract_text(pdf_path)
            if text and text.strip():
                logger.debug(f"Successfully extracted text using simplified method from {pdf_path}")
                return text
        except Exception as e:
            logger.warning(f"Simplified extraction failed for {pdf_path}: {str(e)}")
        
        # If the simplified approach fails or returns empty text, try the more detailed approach
        output_string = StringIO()
        with open(pdf_path, 'rb') as in_file:
            resource_manager = PDFResourceManager()
            device = TextConverter(resource_manager, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(resource_manager, device)
            
            for page in PDFPage.get_pages(in_file, check_extractable=False):
                interpreter.process_page(page)
                
            device.close()
            text = output_string.getvalue()
        
        if not text or not text.strip():
            logger.warning(f"Extracted empty text from {pdf_path}")
            
        logger.debug(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text: {str(e)}")

def extract_text_optimized(pdf_path, max_pages=None):
    """
    A more optimized text extraction function for larger PDFs.
    
    Args:
        pdf_path (str): Path to the PDF file
        max_pages (int, optional): Maximum number of pages to process
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        logger.debug(f"Starting optimized extraction from {pdf_path}")
        
        output_string = StringIO()
        with open(pdf_path, 'rb') as in_file:
            resource_manager = PDFResourceManager()
            device = TextConverter(resource_manager, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(resource_manager, device)
            
            # Process pages with a page counter
            page_count = 0
            for page in PDFPage.get_pages(in_file, check_extractable=False):
                interpreter.process_page(page)
                page_count += 1
                
                # If max_pages is specified and we've reached the limit, break
                if max_pages and page_count >= max_pages:
                    break
                    
            device.close()
            text = output_string.getvalue()
        
        logger.debug(f"Successfully extracted {len(text)} characters from {page_count} pages of {pdf_path}")
        return text
        
    except Exception as e:
        logger.error(f"Error in optimized extraction from {pdf_path}: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text: {str(e)}")
