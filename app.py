import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Configure file uploads
app.config["UPLOAD_FOLDER"] = "/tmp/uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload size
app.config["ALLOWED_EXTENSIONS"] = {"pdf"}

# Initialize the app with SQLAlchemy
db.init_app(app)

# Create upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Import utility modules
from utils.pdf_extractor import extract_text_from_pdf
from utils.openrouter_api import (
    analyze_seniority, 
    analyze_industry, 
    analyze_job_type, 
    analyze_specific_role
)

# Import models
with app.app_context():
    import models
    db.create_all()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Check if a file was uploaded
    if "resume" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("index"))
    
    file = request.files["resume"]
    
    # If user does not select file, browser submits an empty file
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(url_for("index"))
    
    if file and allowed_file(file.filename):
        try:
            # Generate a safe filename with UUID to prevent collisions
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            
            # Save the file
            file.save(filepath)
            
            # Extract text from PDF
            text = extract_text_from_pdf(filepath)
            
            if not text or len(text.strip()) == 0:
                flash("Could not extract text from the PDF. Please try a different file.", "danger")
                return redirect(url_for("index"))
            
            # Analyze CV with OpenRouter API
            seniority = analyze_seniority(text)
            industry = analyze_industry(text)
            job_type = analyze_job_type(text)
            specific_role = analyze_specific_role(text)
            
            # Store analysis results in session for display
            session["analysis_results"] = {
                "seniority": seniority,
                "industry": industry,
                "job_type": job_type,
                "specific_role": specific_role,
                "text_preview": text[:500] + "..." if len(text) > 500 else text
            }
            
            # Save to database
            new_analysis = models.CVAnalysis(
                original_filename=filename,
                file_path=filepath,
                extracted_text=text,
                seniority=seniority,
                industry=industry,
                job_type=job_type,
                specific_role=specific_role
            )
            db.session.add(new_analysis)
            db.session.commit()
            
            return redirect(url_for("results"))
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            flash(f"Error processing file: {str(e)}", "danger")
            return redirect(url_for("index"))
    else:
        flash("Allowed file type is PDF", "danger")
        return redirect(url_for("index"))

@app.route("/results", methods=["GET"])
def results():
    # Get analysis results from session
    analysis_results = session.get("analysis_results")
    
    if not analysis_results:
        flash("No analysis results found. Please upload a CV first.", "warning")
        return redirect(url_for("index"))
    
    return render_template("results.html", results=analysis_results)

@app.route("/history", methods=["GET"])
def history():
    # Get last 10 analyses from database
    analyses = models.CVAnalysis.query.order_by(models.CVAnalysis.created_at.desc()).limit(10).all()
    return render_template("results.html", analyses=analyses, is_history=True)

@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum size is 5MB.", "danger")
    return redirect(url_for("index"))

@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html", error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", error="Internal server error. Please try again later."), 500
