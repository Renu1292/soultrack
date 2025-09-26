import time
from email.policy import default

# INSTALL PACKAGES:
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from sqlalchemy import ForeignKey
from wtforms import StringField, SubmitField, IntegerField, PasswordField, DateField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError, Optional, NumberRange
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import date,datetime,timedelta
import random
from collections import Counter, defaultdict
from flask_migrate import Migrate
from analysis_engine import detect_burnout, suggest_weekly_goal, compute_productivity_score
from weasyprint import HTML
import io
import ctypes.util
from huggingface_hub import InferenceClient
import os
import time
from dotenv import load_dotenv
import traceback

# STEP 1: Load environment variables
load_dotenv()  # Loads from.env file

# STEP 2: Configure Flask
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# DATABASE CONFIGURATION
def get_database_url():
    db_url = os.getenv('DATABASE_URL', '')

    # Production on Render
    if os.getenv('RENDER'):
        if db_url and db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
            print("🚀 Production: Using PostgreSQL")
        return db_url

    # Development
    if not db_url:
        db_url = 'sqlite:///users.db'
    print("💻 Development: Using SQLite")
    return db_url

app.config['SQLALCHEMY_DATABASE_URI']= get_database_url()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disables warning

print(f"✅ Database configured: {app.config['SQLALCHEMY_DATABASE_URI']}")

# app.config['SQLALCHEMY_DATABASE_URI']= os.getenv('DATABASE_URL')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disables warning

# Ensure instance folder exists
try:
    os.makedirs(app.instance_path, exist_ok=True)
    print(f"Instance path: {app.instance_path}")
except Exception as e:
    print(f"Error creating instance folder: {e}")

# STEP 3: Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)

USE_AI = True  # Global setting to control AI usage

# SET UP LOGIN MANAGER
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

MOOD_CHOICES = [
    ('happy', '😄 Happy'),
    ('determined', ' 🥷 Determined'),
    ('beast_mode', '🔥🔥 Beast Mode'),
    ('accomplished', '😎 Accomplished'),
    ('mind_blown', '🤯 Mind Blown'),
    ('calm', '😌 Calm'),
    ('meh', '🙄 Meh!'),
    ('️slow_day', '🙂‍ Slow Day!'),
    ('clueless', ' 😵 Clueless'),
    ('frustrated', '🤬 Frustrated'),
    ('exhausted', '😭 Exhausted'),
]

MOOD_CATEGORY_MAP = {
    'happy': 'positive',
    'determined': 'positive',
    'beast_mode': 'positive',
    'accomplished': 'positive',
    'mind_blown': 'positive',
    'calm': 'positive',
    'meh': 'neutral',
    '️slow_day': 'neutral',
    'clueless': 'negative',
    'frustrated': 'negative',
    'exhausted': 'negative'
}

# CREATE THE USER MODELS:
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String, nullable=False)

    results = db.relationship('Result', backref='user', lazy=True)
    goals = db.relationship('Goal', backref='user', lazy=True)
    weekly_summaries = db.relationship('WeeklySummary', backref='user', lazy=True)

    def __repr__(self):
        return f"User {self.id}: {self.name}>"

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, default=date.today, nullable=False)
    day = db.Column(db.String(50), nullable=False)
    hours = db.Column(db.Integer, nullable=False)
    mood = db.Column(db.String(50), nullable=False)
    mood_category = db.Column(db.String(100), nullable=False)
    obstacles = db.Column(db.String(200), nullable=False)
    lesson = db.Column(db.String(200), nullable=False)
    action = db.Column(db.String(200), nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.Date, default=date.today, nullable=False)
    target_hours = db.Column(db.Integer, nullable=False)
    week_start = db.Column(db.Date, nullable=False)
    week_end = db.Column(db.Date, nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class WeeklySummary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    week_start = db.Column(db.Date, nullable=False)
    week_end = db.Column(db.Date, nullable=False)
    total_hours = db.Column(db.Float, nullable=False)
    session_count = db.Column(db.Integer, nullable=False)
    goal_target = db.Column(db.Integer)
    goal_achieved = db.Column(db.Boolean)
    productivity_score = db.Column(db.Float, nullable=True)
    burnout_flag = db.Column(db.Boolean, nullable=True)
    excuse_count = db.Column(db.Integer, nullable=True)
    ai_feedback = db.Column(db.Text)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# USER LOADER FUNCTION:
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# FUNCTION TO COUNT HOW MAND DAYS WERE LOGGED IN THE WEEK
def get_logged_days_this_week(user_id):
    today = date.today()

    # 1. Get the start of the current week (Sunday)
    start_of_week = today - timedelta(days=(today.weekday() + 1) % 7)

    # 2. Get the of the current week (Saturday)
    end_of_week = start_of_week + timedelta(days=6)

    # 3. Query user sessions within this week
    user_sessions = Result.query.filter(
        Result.user_id == user_id,
        Result.date >= start_of_week,
        Result.date <= end_of_week
    ).all()

    # 4. Extract unique days from these sessions
    logged_days = {s.date for s in user_sessions}

    # 5. Debugging logs
    print(f"Start of week: {start_of_week}")
    print(f"End of week: {end_of_week}")
    print(f"Logged Days This Week: {logged_days}")
    print(f"Stage (Days Logged): {len(logged_days)}")

    # 6. Reset to soil if it's a new week and nothing logged yet
    if today == start_of_week and len(logged_days) == 0:
        return 0  # Show soil image - growing_o.png

    return len(logged_days) # growing_1.png to growing_7.png


# HELPER FUNCTION: Get all sessions for the current calendar week (Sunday to Saturday)
def get_sessions_this_week(user_id, return_range=False):
    today = date.today()

    # 1. Get the start of the current week (Sunday)
    start_of_week = today - timedelta(days=(today.weekday() + 1) % 7)

    # 2. Get the of the current week (Saturday)
    end_of_week = start_of_week + timedelta(days=6)

    # 3. Query user sessions within this week
    sessions = Result.query.filter(
        Result.user_id == user_id,
        Result.date >= start_of_week,
        Result.date <= end_of_week
    ).order_by(Result.date).all()

    if return_range:
        return sessions, start_of_week, end_of_week
    return sessions

# Helper function : get the status of session logged
def get_week_days_status(user_id):
    today = date.today()

    # Get the start of the current week (Sunday)
    start_of_week = today - timedelta(days=(today.weekday() + 1) % 7)

    # Generate all dates in the current week (Sunday to Saturday)
    dates = [start_of_week + timedelta(days=i) for i in range(7)]

    # Query the status logged
    sessions = Result.query.filter(
        Result.user_id == user_id,
        Result.date >= start_of_week,
        Result.date <= start_of_week + timedelta(days=6)
    ).all()

    # Extraxt unique days from these sessions
    session_dates = {s.date for s in sessions}

    # Return status list and actual dates
    return [d in session_dates for d in dates], dates

# CREATE THE DATABASE:
with app.app_context():
    db.create_all()

# CREATE THE FORMS:
class RegistrationForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()] )
    password = PasswordField('Password', validators=[DataRequired()] )
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')] )
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()] )
    password = PasswordField('Password', validators=[DataRequired()] )
    submit = SubmitField('Login')

class HustleForm(FlaskForm):
    date = DateField('Date', validators=[DataRequired()] )
    hours = IntegerField('Focused Hours', validators=[DataRequired()] )
    mood = SelectField('How did you feel today?', choices=MOOD_CHOICES,validators=[DataRequired()] )
    obstacles = TextAreaField('What Challenges Came Up?', validators=[DataRequired()] )
    lesson = TextAreaField('Reflection',validators=[DataRequired()] )
    action = TextAreaField('Your Game Plan for Tomorrow', validators=[DataRequired() ])
    submit = SubmitField('Save')

class GoalForm(FlaskForm):
    target_hours = IntegerField('Target Hours', validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField('Set Goal')

def generate_ai_feedback(obstacles, lesson, action):
   """Generate personalized feedback based using Hugging Face's API on daily reflections."""

   print((f"USE_AI: {USE_AI}, Obstacles: {bool(obstacles)}, Lesson: {bool(lesson)}, Action: {bool(action)}"))

   # MOCK MODE = No API calls
   if not USE_AI:
        print("MOCK MODE: Generating mock feedback...")

        feedback_parts = []
        if obstacles:
            feedback_parts.append("🚧 Obstacle Note: Break challenges into smaller steps.")
        if lesson:
            feedback_parts.append("💡 Lesson Insight: Focus on applying this tomorrow.")
        if action:
            feedback_parts.append("✅ Action Tip: Schedule this first thing in the morning.")

        return  "\n\n".join(feedback_parts) if feedback_parts else "No specific feedback generated."

   else: # AI MODE
        print("AI MODE: Preparing prompt...") # Debug
        reflection = "\n".join([
            f"OBSTACLES: {obstacles}" if obstacles else "",
            f"LESSONS: {lesson}" if lesson else "",
            f"ACTIONS: {action}" if action else ""
        ]).strip()

        if not reflection:
            return "Please provide obstacles, lessons, or actions."

        # Mistral-7B specific prompt format
        prompt = f"""<s>[INST] <<SYS>>      
You are a productivity coach.Provide CONCISE feedback with:
1. One obstacle breakthrough strategy
2. One way to apply the lesson
3. One action improvement suggestion
<</SYS>>

Reflection:
{reflection}[/INST]"""
        print("Prompt:", prompt[:200] + "...")

        # Real API call (only happens if USE_AI = True)
        try:
           # Verify hugging face token is available
           hf_token = os.getenv("HF_TOKEN")
           if not hf_token:
               raise ValueError("Hugging Face token not found in environment variables")

           print("Initializing InferenceClient...")
           client = InferenceClient(
               model="mistralai/Mistral-7B-Instruct-v0.1",
               token=hf_token,
               timeout=30
           )

           print("Making API call...")
           response = client.text_generation(
               prompt,
               max_new_tokens=200,
               temperature=0.7,
               do_sample=True,
               stop=["</s>", "[INST]"]  # Prevent incomplete responses
           )

           # Clean up response
           cleaned_response = response.split("[/INST]")[-1].strip()
           return cleaned_response if cleaned_response else "Received empty response from AI"

        except Exception as e:
            print("❌ Hugging Face API Error:", str(e))
            return f"""🔧 Our AI coach is busy.Quick tips:
1. Try the 2 minute rule for obstacles.
2.Teach someone your lesson
3. Put actions on your calender

(Error: {str(e)})"""

#  FUNCTION TO GENERATE AI FEEDBACK - ANALYSE THE ENTIRE WEEK'S DATA
def generate_weekly_ai_feedback(obstacles_list, lessons_list, actions_list):
    if not USE_AI:
        return "Weekly insights not available in mock mode. Log in later to receive full analysis."

    # Create weekly reflection prompt
    prompt = f"""<s>[INST] <<SYS>>
You are an insightful productivity coach and data analyst. Your task:
1. Identify repeating/limiting obstacles
2. Evaluate lesson/action quality
3. Suggest improvements
4. Give 1 focused suggestion for next week
<</SYS>>

Obstacles they faced:
{', '.join(obstacles_list)}

Lessons they learned:
{', '.join(lessons_list)}

Actions they planned:
{', '.join(actions_list)}

Analysis (be honest, kind and practical):[/INST]"""

    try:
        client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1")
        response = client.text_generation(
            prompt,
            max_new_tokens=300,
            temperature=0.6
        )
        return response.strip()
    except Exception as e:
        return f"Error generating weekly analysis: {str(e)}"

def detect_excuse_with_ai(obstacle_text):
    """
    Classifies obstacles as either 'Excuse' or 'Genuine' with AI assistance.

    args:
        obstacle_text (str): User's description of their obstacle

    Returns:
          str: 'Excuse' or 'Genuine'
    """
    # Early return for empty input
    if not obstacle_text.strip():
        return "Genuine"

    # Mock Mode - optimized keyword checking
    if not USE_AI:
        excuse_keywords = {
            "tired", "scared", "lazy", "don't want", "can't", "no time",
            "boring", "later"
        }
        lower_text = obstacle_text.lower()
        return "Excuse" if any(kw in lower_text for kw in excuse_keywords) else "Genuine"

    try:
        # Define the AI prompt
        prompt = f"""<s>[INST] <<SYS>>
        Classify this obstacles as "Excuse" or "Genuine".Guidelines:
        Genuine" = External factors, verifiable circumstances, or legitimate limitations
        - "Excuses" = Internal resistance, avoidable patterns, or emotional barrier
        when in doubt, choose "Genuine".
        Respond with ONLY one word.
        <</SYS>>       
               
        Reflection : "{obstacle_text}"
        Classification:[/INST]"""
        
        # Make the OpenAI call
        client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1")
        response = client.text_generation(
            prompt,
            max_new_tokens=10,
            temperature=0.3
        )
        return response.strip().split()[0]
    except Exception as e:
        print(f"AI classification error: {e}")
        return "Genuine"

#  CREATE THE ROUTES
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if username already exists
        existing_user = User.query.filter(User.name == form.name.data).first()
        if existing_user:
            flash('Username already exists. Please log in', 'danger')
            return redirect(url_for('login'))

        # Create a new user
        hash_pw = generate_password_hash(form.password.data, salt_length=10)
        user = User(name= form.name.data, password=hash_pw)
        db.session.add(user)
        db.session.commit()
        flash('Account has been registered successfully. Please Log In.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(name= form.name.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Login Successful', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)

@app.route('/hustle', methods=['GET', 'POST'])
@login_required
def hustle():
    form = HustleForm()
    ai_feedback = None

    if form.validate_on_submit():
        selected_date = form.date.data
        day_of_week = selected_date.strftime('%A') # Auto-calculate day

        mood_code = form.mood.data
        mood_category = MOOD_CATEGORY_MAP.get(mood_code,'unknown')

        # Process and save form data to database
        new_result = Result(
            date = form.date.data,
            day = day_of_week,
            hours =form.hours.data,
            mood = mood_code,
            mood_category = mood_category,
            obstacles = form.obstacles.data,
            lesson = form.lesson.data,
            action = form.action.data,
            user_id = current_user.id
        )
        db.session.add(new_result)
        db.session.commit()

        # Call the AI feedback generator
        ai_feedback = generate_ai_feedback(
            obstacles=form.obstacles.data,
            lesson=form.lesson.data,
            action=form.action.data
        )

        flash("Hustle session logged!", "success")
        return redirect(url_for("dashboard"))
    return render_template("hustle.html", form=form, result=None, ai_feedback=ai_feedback)

@app.route('/dashboard')
@login_required
def dashboard():
    # 1. Get filter parameters from query string ((e.g. /dashboard?start=2025-07-01)
    start_date = request.args.get('start')

    # 2. Parse date inputs if provided
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
            entries = Result.query.filter(
                Result.user_id == current_user.id,
                Result.date == start
            ).order_by(Result.date.desc()).all()
        except ValueError:
            # If date parsing fails, show no results
            entries = []
    else:
        # Default: show all entries
        entries = Result.query.filter_by(user_id=current_user.id).order_by(Result.date.desc()).all()

    # 3. Get the days tracked [For Hustle Garden]
    logged_days_count = get_logged_days_this_week(current_user.id)
    stage = logged_days_count

    # 4. Define current week's range (Mon-Sat)
    today = date.today()
    week_start = today - timedelta(days=(today.weekday() + 1) % 7) # Sunday
    week_end = week_start + timedelta(days=6) # Saturday

    # 5. Try to fetch the goal
    goal = Goal.query.filter_by(
        user_id = current_user.id,
        week_start = week_start,
        week_end = week_end
    ).first()

    print(f"Week Range: {week_start} to {week_end}")
    print(f"Found goal: {goal}")

    goal_target = goal.target_hours if goal else None

    # 6. Get all sessions this week
    sessions = get_sessions_this_week(current_user.id)
    total_hours = sum([s.hours for s in sessions])

    # 7. Check goal status
    goal_achieved = total_hours >= goal_target if goal_target else False

    # 8. Compute progress %
    progress = int((total_hours / goal_target) * 100) if goal_target and goal_target > 0 else 0

    return render_template(
        "dashboard.html",
        name=current_user.name,
        entries=entries,
        logged_days_count=logged_days_count,
        stage=stage,
        goal_target=goal_target,
        goal_achieved=goal_achieved,
        progress=progress,
        total_hours=total_hours)

@app.route("/ai_guide/obstacles", methods=["POST"])
@login_required
def get_guide_obstacles():
    data = request.get_json()
    obstacles = data.get("input")

    feedback = generate_ai_feedback(obstacles,"", "")
    return jsonify({"feedback": feedback})

@app.route("/edit/<int:id>", methods=['GET', 'POST'])
@login_required
def edit(id):
    edit_session = Result.query.get_or_404(id) # Get the log_session by ID

    # Ensure the session belongs to the current user
    if edit_session.user_id != current_user.id:
        flash("You are not authorized to edit this log-session", 'danger')
        return redirect(url_for("home"))

    # Pre-fill the form with the current data:
    form = HustleForm(
        date = edit_session.date,
        hours = edit_session.hours,
        mood = edit_session.mood,
        obstacles = edit_session.obstacles,
        lesson = edit_session.lesson,
        action = edit_session.action
    )
    if form.validate_on_submit():
        # Update the session with the new data from the form
        edit_session.date = form.date.data
        edit_session.hours = form.hours.data
        edit_session.mood = form.mood.data
        edit_session.obstacles = form.obstacles.data
        edit_session.lesson = form.lesson.data
        edit_session.action = form.action.data

        # Commit the changes to the database:
        db.session.commit()
        flash("Session updated successfully!", "success")

        # Redirect to dashboard after update.
        return redirect(url_for("dashboard"))

    return render_template("hustle.html", form=form, result=edit_session)

@app.route("/delete/<int:id>", methods=['GET', 'POST'])
@login_required
def delete(id):
    delete_session = Result.query.get(id) # Get the session by its id

    if delete_session:
        db.session.delete(delete_session) # Delete the session
        db.session.commit() # Save the changes
        flash('Session deleted successfully!', "warning")
    else:
        flash("Session not found!", "danger")
    return redirect(url_for("dashboard"))

@app.route("/analysis")
@login_required
def analysis():
    # Get all user sessions for current user
    user_sessions = Result.query.filter_by(user_id =current_user.id).order_by(Result.date).all()

    # Get the date and hours data
    dates = [s.date.strftime('%b %d') for s in user_sessions]
    hours = [s.hours for s in user_sessions]

    # Mood breakdown
    mood_list = [s.mood for s in user_sessions]
    mood_counts = dict(Counter(mood_list))
    top_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "No data"

    total_hours = sum(hours)
    avg_hours = round(total_hours / len(user_sessions),1) if user_sessions else 0


    return render_template("analysis.html",
        dates = dates,
        hours = hours,
        mood_counts = mood_counts,
        top_mood = top_mood,
        total_hours = total_hours,
        avg_hours = avg_hours,
        session_count = len(user_sessions)
    )


# Deletes current summary → redirects to / weekly_summary(which rebuilds it)
@app.route("/weekly_summary")
@login_required
def weekly_summary():
    # --- 1. Fetch Sessions for the current week ---
    recent_sessions, week_start, week_end = get_sessions_this_week(current_user.id, return_range=True)

   #  --- 2. Check in status ---
    days_status, week_dates = get_week_days_status(current_user.id)

    # --- 3. Burnout detection ---
    burnout_flag, burnout_message = detect_burnout(recent_sessions)

    # --- 4. Extract dates and hours for charting ---
    dates = [s.date.strftime('%b %d') for s in recent_sessions]
    hours = [s.hours for s in recent_sessions]

    # --- 5.  Mood Breakdown ---
    mood_list = [s.mood for s in recent_sessions]
    mood_counts = dict(Counter(mood_list))
    top_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "No data"

    # --- 6. Productivity metrics ---
    total_hours = sum(hours)
    avg_hours = round(total_hours / len(recent_sessions), 1) if recent_sessions else 0

    # --- 7. Identify Most & Least Productive Day ---
    if recent_sessions:
        most_productive = max(recent_sessions, key=lambda  s: s.hours)
        least_productive = min(recent_sessions, key=lambda  s: s.hours)

        most_productive_day = most_productive.date.strftime('%A')
        most_productive_hours = float(most_productive.hours)

        least_productive_day = least_productive.date.strftime('%A')
        least_productive_hours = float(least_productive.hours)
    else:
        most_productive_day = least_productive_day = "No data"
        most_productive_hours = least_productive_hours = 0.0

    # --- 8. Weekly AI Feedback Reflection: Obstacles, Lessons, Actions ---
    obstacles = [s.obstacles for s in recent_sessions if s.obstacles]
    lessons = [s.lesson for s in recent_sessions if s.lesson]
    actions = [s.action for s in recent_sessions if s.action]

    ai_feedback = generate_weekly_ai_feedback(obstacles, lessons, actions)

    # ---9. Excuse Detection with AI ---
    excuses = []
    genuines = []

    for o in obstacles:
        label = detect_excuse_with_ai(o)
        if label == "Excuse":
            excuses.append(o)
        elif label == "Genuine":
            genuines.append(o)

    print(excuses)

    excuse_count = len(excuses)

    # ---10. Weekly Goal Tracking ---
    goal = Goal.query.filter_by(
        user_id = current_user.id,
        week_start = week_start,
        week_end = week_end
    ).first()

    goal_target = goal.target_hours if goal else None
    goal_achieved = total_hours >= goal_target if goal_target else False

    # ---11. Productivity Score + Breakdown ---
    session_days = len({ s.date for s in recent_sessions})

    # This returns both score and breakdown dictionary ---
    productivity_score, breakdown = compute_productivity_score(
        total_hours = total_hours,
        session_days = session_days,
        goal_target = goal_target,
        goal_achieved = goal_achieved,
        mood_list = mood_list,
        burnout_flag = burnout_flag,
        baseline_hours= 20
    )

    # ---12.  Fetch past weekly summaries (excluding this week) to suggest a goal
    past_summaries = WeeklySummary.query.filter(
        WeeklySummary.user_id == current_user.id,
        WeeklySummary.week_end < week_end
    ).order_by(WeeklySummary.week_end.desc()).limit(4).all()

    # Extract past total hours
    past_week_hours = [s.total_hours for s in reversed(past_summaries)] # reversed for oldest -> latest

    # Last week goal success
    last_goal_met = past_summaries[-1].goal_achieved if past_summaries else None

    # Add debug prints before the function call
    print("PAST WEEK HOURS:", past_week_hours)
    print("LAST GOAL MET:", last_goal_met)
    print("Recent session dates & hours:",[(s.date, s.hours) for s in recent_sessions])

    # Suggest next goal based on trend
    suggested_goal = suggest_weekly_goal(total_hours, past_week_hours)

    # ---13. WeeklySummary DB Record Check/Create ---
    existing_summary = WeeklySummary.query.filter_by(
            user_id = current_user.id,
            week_start = week_start,
            week_end = week_end
    ).first()

    print("SUMMARY FROM DB:", existing_summary.total_hours if existing_summary else "No summary found.")

    # If no summary OR it's stale (session_count = 0 but now we have sessions), regenerate it
    if not existing_summary or (existing_summary.session_count == 0 and len(recent_sessions) > 0):
        if existing_summary:
            db.session.delete(existing_summary)
            db.session.commit()

        # Create a new WeeklySummary record
        summary = WeeklySummary(
            user_id = current_user.id,
            week_start = week_start,
            week_end =  week_end,
            total_hours = total_hours,
            session_count = len(recent_sessions),
            productivity_score = productivity_score,
            burnout_flag = burnout_flag,
            excuse_count = excuse_count,
            ai_feedback = ai_feedback,
            goal_target = goal_target,
            goal_achieved = goal_achieved
        )
        db.session.add(summary)
        db.session.commit()

    # ---14.  Render the weekly summary page with all metrics
    return render_template("weekly_summary.html",
            dates = dates,
            hours = hours,
            mood_counts = mood_counts,
            top_mood = top_mood,
            total_hours = total_hours,
            avg_hours = avg_hours,
            session_count = len(recent_sessions),
            most_productive_day = most_productive_day,
            most_productive_hours = most_productive_hours,
            least_productive_day = least_productive_day,
            least_productive_hours = least_productive_hours,
            ai_feedback = ai_feedback,
            mock_mode = not USE_AI, # dynamic based on the global USE_AI flag
            burnout_message = burnout_message,
            suggested_goal = suggested_goal,
            productivity_score = productivity_score,
            days_status = days_status,
            week_dates = week_dates,
            excuses = excuses,
            genuines = genuines,
            breakdown = dict(breakdown)

    )

@app.route('/set_goal', methods=['GET', 'POST'])
@login_required
def set_goal():
    form = GoalForm()

    if form.validate_on_submit():
        print("Form validated successfully")
        print("Submitted Goal Hours:", form.target_hours.data)

        # Compute week_start and week_end dynamically
        today = date.today()
        week_start = today - timedelta(days=(today.weekday() + 1) % 7) # Sunday
        week_end = week_start + timedelta(days=6) # Saturday

        # Check for existing goal first
        existing_goal = Goal.query.filter_by(
            user_id = current_user.id,
            week_start = week_start,
            week_end = week_end
        ).first()

        if existing_goal:
            existing_goal.target_hours = form.target_hours.data
        else:
            new_goal = Goal(
                 target_hours = form.target_hours.data,
                 user_id = current_user.id,
                 week_start = week_start,
                 week_end = week_end
            )
            db.session.add(new_goal)

        # Update WeeklySummary if it exits
        existing_summary = WeeklySummary.query.filter_by(
            user_id = current_user.id,
            week_start = week_start,
            week_end = week_end
        ).first()

        if existing_summary:
            existing_summary.goal_target = form.target_hours.data
            existing_summary.goal_achieved = (existing_summary.total_hours >= form.target_hours.data)

            print(f"Updated goal: target={form.target_hours.data}h, achieved={existing_summary}")

        db.session.commit()
        flash("Goals saved successfully!", "success")
        return redirect(url_for('dashboard'))

    return render_template('set_goal.html', form=form)

@app.route("/summary_history")
@login_required
def summary_history():
    summaries = WeeklySummary.query.filter_by(user_id=current_user.id)\
                .order_by(WeeklySummary.week_start.desc()).all()

    return render_template("summary_history.html", summaries=summaries)

@app.route("/refresh_summary")
@login_required
def refresh_summary():
    # Get current week range
    recent_sessions, week_start, week_end = get_sessions_this_week(current_user.id, return_range=True)

    # Find existing summary (if any)
    existing_summary = WeeklySummary.query.filter_by(
        user_id = current_user.id,
        week_start = week_start,
        week_end = week_end
    ).first()

    # Delete old summary
    if existing_summary:
        db.session.delete(existing_summary)
        db.session.commit()

    # Redirect to summary page which will now generate a fresh summary
    return redirect(url_for('weekly_summary'))

@app.route("/export_summary")
@login_required
def export_summary():
    # Fetch existing summary from DB
    summary = WeeklySummary.query.filter_by(
        user_id = current_user.id
    ).order_by(WeeklySummary.week_end.desc()).first()

    if not summary:
        return "No summary available", 404

    # Render the summary into HTML
    rendered_html = render_template("summary_pdf.html", summary=summary)

    # Convert to PDF
    pdf_file = HTML(string=rendered_html).write_pdf()

    return send_file(
        io.BytesIO(pdf_file),
        download_name = "Weekly_Summary.pdf",
        mimetype="application/pdf",
        as_attachment= True
    )

@app.route('/ai_guide/reflection', methods=['POST'])
@login_required
def handle_ai_requests():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSOn data received'}),400

        print(f"Received data: {data}")

        feedback = generate_ai_feedback(
            obstacles=data.get('obstacles', ''),
            lesson=data.get('lesson',''),
            action=data.get('action','')
        )

        return jsonify({
            'status':'success',
            'feedback': feedback
        })

    except Exception as e:
        print("Error traceback:\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/logout')
@login_required
def logout():
    # Clear all session data
    session.clear()
    flash('You have been logged out successfully.', 'info')

    # Redirect to login page
    return redirect(url_for('login'))







if __name__ == '__main__':
    app.run(debug=True)










