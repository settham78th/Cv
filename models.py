from app import db
from datetime import datetime

class CVAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(256), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False)
    seniority = db.Column(db.String(64))
    industry = db.Column(db.String(64))
    job_type = db.Column(db.String(64))
    specific_role = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<CVAnalysis {self.id}: {self.original_filename}>"
