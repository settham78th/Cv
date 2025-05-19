import logging
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

# Configure logging
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PDFMiner.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        Exception: If PDF extraction fails
    """
    try:
        logger.debug(f"Extracting text from PDF: {pdf_path}")
        
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create a PDF parser object
            parser = PDFParser(file)
            
            # Create a PDF document object
            document = PDFDocument(parser)
            
            # Check if the document allows text extraction
            if not document.is_extractable:
                logger.warning(f"PDF {pdf_path} does not allow text extraction")
                return "This PDF does not allow text extraction."
            
            # Create a PDF resource manager object
            rsrcmgr = PDFResourceManager()
            
            # Create a string buffer for the extracted text
            output_string = StringIO()
            
            # Create a PDF device object
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            
            # Create a PDF interpreter object
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            
            # Process each page in the document
            for page in PDFPage.create_pages(document):
                interpreter.process_page(page)
            
            # Get the extracted text
            text = output_string.getvalue()
            
            # Close the device
            device.close()
            
            # Close the string buffer
            output_string.close()
            
            logger.debug(f"Successfully extracted {len(text)} characters from PDF")
            
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")
