import logging
import os
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text or None if extraction failed
    """
    try:
        logger.info(f"Starting text extraction from {pdf_path}")
        
        output_string = StringIO()
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
        
        text = output_string.getvalue()
        
        if not text or text.isspace():
            logger.warning(f"Extraction returned empty text from {pdf_path}")
            return "No text could be extracted from this PDF. The file might be scanned or contain only images."
        
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None