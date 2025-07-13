from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, Dict, Any
import joblib
import re
import uuid
from datetime import datetime

app = FastAPI(
    title="AI Lead Scoring API",
    description="API for lead scoring with ML model and rule-based reranking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for leads
leads_storage: Dict[str, Dict[str, Any]] = {}

# Load the trained model
try:
    model_data = joblib.load("lead_scoring_model.pkl")
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    model = None

class LeadInput(BaseModel):
    """Pydantic model for lead input validation"""
    email: EmailStr
    full_name: str
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    country: str = "India"
    specialization: str = "Select"
    lead_source: str = "Website"
    lead_origin: str = "Landing Page Submission"
    current_occupation: str = "Select"
    total_visits: int = 0
    total_time_spent: int = 0
    page_views_per_visit: float = 0.0
    comments: Optional[str] = ""
    consent: bool = True
    
    @validator('total_visits', 'total_time_spent')
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v
    
    @validator('page_views_per_visit')
    def validate_page_views(cls, v):
        if v < 0:
            raise ValueError('Page views per visit must be non-negative')
        return v
    
    @validator('consent')
    def validate_consent(cls, v):
        if not v:
            raise ValueError('Consent must be given')
        return v

class LeadResponse(BaseModel):
    """Pydantic model for lead response"""
    lead_id: str
    email: str
    initial_score: int
    reranked_score: int
    comments: Optional[str]
    timestamp: str
    status: str = "processed"

class RuleBasedReranker:
    """Rule-based reranker that simulates LLM behavior"""
    
    def __init__(self):
        self.positive_keywords = {
            'urgent': 15,
            'interested': 10,
            'asap': 12,
            'immediately': 10,
            'soon': 8,
            'quick': 8,
            'fast': 8,
            'priority': 12,
            'important': 10,
            'need': 8,
            'want': 6,
            'looking': 6,
            'ready': 10,
            'budget': 8,
            'buy': 12,
            'purchase': 12,
            'invest': 10,
            'decision': 8,
            'approve': 10,
            'manager': 6,
            'director': 8,
            'ceo': 10,
            'founder': 10,
            'recommend': 8,
            'excellent': 6,
            'great': 4,
            'good': 3,
            'perfect': 8,
            'exactly': 6,
            'suitable': 6,
            'fits': 6,
            'match': 6
        }
        
        self.negative_keywords = {
            'not interested': -15,
            'no thanks': -12,
            'unsubscribe': -20,
            'remove': -15,
            'stop': -15,
            'spam': -20,
            'waste': -12,
            'expensive': -8,
            'costly': -8,
            'cheap': -6,
            'later': -6,
            'maybe': -4,
            'think': -3,
            'consider': -3,
            'busy': -6,
            'time': -4,
            'delay': -8,
            'postpone': -10,
            'cancel': -15,
            'decline': -12,
            'reject': -12,
            'refuse': -12,
            'no': -2,
            'not': -2,
            'never': -10,
            'impossible': -10,
            'difficult': -6,
            'hard': -4,
            'problem': -6,
            'issue': -6,
            'concern': -4,
            'worried': -6,
            'doubt': -6,
            'unsure': -4,
            'confused': -4
        }
    
    def analyze_comments(self, comments: str) -> tuple[int, list]:
        """Analyze comments and return score adjustment and matched keywords"""
        if not comments:
            return 0, []
        
        comments_lower = comments.lower()
        adjustment = 0
        matched_keywords = []
        
        # Check positive keywords
        for keyword, score in self.positive_keywords.items():
            if keyword in comments_lower:
                adjustment += score
                matched_keywords.append(f"+{score} ({keyword})")
        
        # Check negative keywords
        for keyword, score in self.negative_keywords.items():
            if keyword in comments_lower:
                adjustment += score  # score is already negative
                matched_keywords.append(f"{score} ({keyword})")
        
        return adjustment, matched_keywords

def preprocess_lead_for_prediction(lead_data: LeadInput) -> dict:
    """Convert lead input to format expected by the model"""
    return {
        'TotalVisits': lead_data.total_visits,
        'Total Time Spent on Website': lead_data.total_time_spent,
        'Page Views Per Visit': lead_data.page_views_per_visit,
        'Lead Origin': lead_data.lead_origin,
        'Lead Source': lead_data.lead_source,
        'Country': lead_data.country,
        'Specialization': lead_data.specialization,
        'What is your current occupation': lead_data.current_occupation,
        'Lead Quality': 'Might be',  # Default value
        'Do Not Email': 'No',
        'Do Not Call': 'No',
        'Asymmetrique Activity Score': 15,  # Default values
        'Asymmetrique Profile Score': 15,
        'Asymmetrique Activity Index': '02.Medium',
        'Asymmetrique Profile Index': '02.Medium',
        'Search': 'No',
        'Magazine': 'No',
        'Digital Advertisement': 'No',
        'Through Recommendations': 'No'
    }

def predict_lead_score(lead_data: dict) -> int:
    """Predict lead score using the trained model"""
    if model is None:
        return 50  # Default score if model is not loaded
    
    try:
        # Preprocess the single lead
        import pandas as pd
        df = pd.DataFrame([lead_data])
        data = df.copy()
        data = data.fillna('Unknown')
        
        # Create features similar to training
        features = []
        
        # Numerical features
        numerical_features = ['TotalVisits', 'Total Time Spent on Website', 
                            'Page Views Per Visit', 'Asymmetrique Activity Score', 
                            'Asymmetrique Profile Score']
        
        for col in numerical_features:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Categorical features
        categorical_features = ['Lead Origin', 'Lead Source', 'Country', 'Specialization', 
                              'What is your current occupation', 'Lead Quality',
                              'Do Not Email', 'Do Not Call']
        
        for col in categorical_features:
            if col in data.columns and col in label_encoders:
                try:
                    data[f'{col}_encoded'] = label_encoders[col].transform(data[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    data[f'{col}_encoded'] = 0
        
        # Binary features
        binary_features = ['Search', 'Magazine', 'Newspaper Article', 'X Education Forums',
                          'Newspaper', 'Digital Advertisement', 'Through Recommendations',
                          'Receive More Updates About Our Courses']
        
        for col in binary_features:
            if col in data.columns:
                data[f'{col}_binary'] = (data[col] == 'Yes').astype(int)
        
        # Activity index features
        activity_features = ['Asymmetrique Activity Index', 'Asymmetrique Profile Index']
        for col in activity_features:
            if col in data.columns:
                activity_map = {'01.High': 3, '02.Medium': 2, '03.Low': 1}
                data[f'{col}_num'] = data[col].map(activity_map).fillna(0)
        
        # Select only the features used in training
        feature_data = []
        for col in feature_columns:
            if col in data.columns:
                feature_data.append(data[col].iloc[0])
            else:
                feature_data.append(0)  # Default value for missing features
        
        # Scale and predict
        X_scaled = scaler.transform([feature_data])
        probability = model.predict_proba(X_scaled)[0, 1]
        score = int(probability * 100)
        
        return min(max(score, 0), 100)  # Ensure score is between 0-100
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 50  # Default score on error

# Initialize the reranker
reranker = RuleBasedReranker()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Lead Scoring API is running", "status": "healthy"}

@app.post("/score", response_model=LeadResponse)
async def score_lead(lead: LeadInput):
    """Score a lead and apply rule-based reranking"""
    try:
        # Generate unique lead ID
        lead_id = str(uuid.uuid4())
        
        # Preprocess lead data for model
        model_input = preprocess_lead_for_prediction(lead)
        
        # Get initial score from ML model
        initial_score = predict_lead_score(model_input)
        
        # Apply rule-based reranking
        adjustment, matched_keywords = reranker.analyze_comments(lead.comments or "")
        reranked_score = min(max(initial_score + adjustment, 0), 100)
        
        # Create response
        response = LeadResponse(
            lead_id=lead_id,
            email=lead.email,
            initial_score=initial_score,
            reranked_score=reranked_score,
            comments=lead.comments,
            timestamp=datetime.now().isoformat()
        )
        
        # Store in memory
        leads_storage[lead_id] = {
            "lead_data": lead.dict(),
            "response": response.dict(),
            "matched_keywords": matched_keywords
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing lead: {str(e)}")

@app.get("/leads")
async def get_all_leads():
    """Get all stored leads"""
    return {"leads": list(leads_storage.values())}

@app.get("/leads/{lead_id}")
async def get_lead(lead_id: str):
    """Get specific lead by ID"""
    if lead_id not in leads_storage:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return leads_storage[lead_id]

@app.delete("/leads/{lead_id}")
async def delete_lead(lead_id: str):
    """Delete a specific lead"""
    if lead_id not in leads_storage:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    del leads_storage[lead_id]
    return {"message": "Lead deleted successfully"}

@app.get("/stats")
async def get_stats():
    """Get basic statistics about stored leads"""
    if not leads_storage:
        return {"total_leads": 0, "average_score": 0, "conversion_rate": 0}
    
    scores = [lead["response"]["reranked_score"] for lead in leads_storage.values()]
    high_quality_leads = sum(1 for score in scores if score >= 70)
    
    return {
        "total_leads": len(leads_storage),
        "average_score": sum(scores) / len(scores),
        "high_quality_leads": high_quality_leads,
        "conversion_rate": (high_quality_leads / len(leads_storage)) * 100
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)