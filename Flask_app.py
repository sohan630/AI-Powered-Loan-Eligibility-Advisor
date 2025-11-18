from flask import Flask, render_template, request, redirect, jsonify, session, url_for
import json
import pickle
import numpy as np
import os
import traceback
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# ---------- Load model ----------
try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    print("Model loaded from", MODEL_PATH)
except Exception as e:
    MODEL = None
    print("Warning: model.pkl not loaded:", e)

# ---------- Users helpers ----------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {"users": []}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"users": []}

def save_users(data):
    with open(USERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

# ---------- Auth decorator ----------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "email" not in session:
            # preserve next page
            next_url = request.path
            return redirect(url_for("login", next=next_url))
        return f(*args, **kwargs)
    return wrapper

# ---------- Chatbot/session helpers ----------
def ensure_session():
    if "messages" not in session:
        session["messages"] = []
    if "chat_step" not in session:
        session["chat_step"] = -1
    if "answers" not in session:
        session["answers"] = {}
    session.modified = True

def add_message(role, content):
    ensure_session()
    session["messages"].append({"role": role, "content": content})
    session.modified = True

# Chatbot questions (same order used previously)
QUESTIONS = [
    "What is your Gender? (Male/Female)",
    "Are you Married? (Yes/No)",
    "How many Dependents do you have? (0/1/2/3+)",
    "What is your Education? (Graduate/Not Graduate)",
    "Are you Self Employed? (Yes/No)",
    "Enter your Applicant Income:",
    "Enter your Coapplicant Income:",
    "Enter Loan Amount:",
    "Enter Loan Amount Term (in days):",
    "Enter your Credit History score (300-850):",
    "What is your Property Area? (Rural/Semiurban/Urban)"
]
NUMERIC_STEPS = {5, 6, 7, 8}
CREDIT_STEP = 9

# ---------- Prediction helpers (restored original logic) ----------
def preprocess_data(gender, married, dependents, education, employed, credit, area,
                    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    """Return feature vector matching model expectation."""
    male = 1 if str(gender).strip().lower() == "male" else 0
    married_yes = 1 if str(married).strip().lower() == "yes" else 0

    # dependents one-hot
    if str(dependents) == '1':
        dependents_1, dependents_2, dependents_3 = 1, 0, 0
    elif str(dependents) == '2':
        dependents_1, dependents_2, dependents_3 = 0, 1, 0
    elif str(dependents) in ("3", "3+"):
        dependents_1, dependents_2, dependents_3 = 0, 0, 1
    else:
        dependents_1, dependents_2, dependents_3 = 0, 0, 0

    not_graduate = 1 if str(education).strip().lower() == "not graduate" else 0
    employed_yes = 1 if str(employed).strip().lower() == "yes" else 0
    semiurban = 1 if str(area).strip().lower() == "semiurban" else 0
    urban = 1 if str(area).strip().lower() == "urban" else 0

    # safe numeric conversion
    try:
        ApplicantIncome_f = float(ApplicantIncome)
    except Exception:
        ApplicantIncome_f = 0.0
    try:
        CoapplicantIncome_f = float(CoapplicantIncome)
    except Exception:
        CoapplicantIncome_f = 0.0
    try:
        LoanAmount_f = float(LoanAmount)
    except Exception:
        LoanAmount_f = 0.0
    try:
        Loan_Amount_Term_f = float(Loan_Amount_Term)
    except Exception:
        Loan_Amount_Term_f = 0.0

    ApplicantIncomelog = float(np.log(ApplicantIncome_f)) if ApplicantIncome_f > 0 else 0.0
    totalincomelog = float(np.log(ApplicantIncome_f + CoapplicantIncome_f)) if (ApplicantIncome_f + CoapplicantIncome_f) > 0 else 0.0
    LoanAmountlog = float(np.log(LoanAmount_f)) if LoanAmount_f > 0 else 0.0
    Loan_Amount_Termlog = float(np.log(Loan_Amount_Term_f)) if Loan_Amount_Term_f > 0 else 0.0

    # credit handling: accept binary 0/1 or raw 0-1000 slider
    try:
        credit_val = float(credit)
    except Exception:
        credit_val = 0.0

    if credit_val in (0.0, 1.0):
        credit_flag = int(credit_val)
    else:
        credit_flag = 1 if (credit_val >= 850 and credit_val <= 1000) else 0

    return [
        credit_flag, ApplicantIncomelog, LoanAmountlog, Loan_Amount_Termlog, totalincomelog,
        male, married_yes, dependents_1, dependents_2, dependents_3, not_graduate, employed_yes, semiurban, urban
    ]

def run_prediction_from_answers(answers):
    try:
        gender = answers.get(0, "")
        married = answers.get(1, "")
        dependents = answers.get(2, "0")
        education = answers.get(3, "Graduate")
        self_employed = answers.get(4, "No")
        applicant_income = float(answers.get(5, 0) or 0)
        coapplicant_income = float(answers.get(6, 0) or 0)
        loan_amount = float(answers.get(7, 0) or 0)
        loan_amount_term = float(answers.get(8, 0) or 0)
        credit_history = float(answers.get(9, 0) or 0)
        property_area = answers.get(10, "Urban")

        features = preprocess_data(gender, married, dependents, education,
                                   self_employed, credit_history, property_area,
                                   applicant_income, coapplicant_income, loan_amount, loan_amount_term)

        if MODEL is None:
            return {"error": "Model not loaded on server."}

        pred = MODEL.predict([features])[0]
        if str(pred).upper() in ["Y", "YES", "1"]:
            label = "Eligible"
        else:
            label = "Not eligible"

        return {"prediction": label}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ---------- Signup ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not email or not password:
            return render_template("signup.html", error="Please provide email and password")

        users = load_users()
        for u in users.get("users", []):
            if u.get("email") == email:
                return render_template("signup.html", error="Email already registered")

        users.setdefault("users", [])
        users["users"].append({"name": name, "email": email, "password": password})
        save_users(users)
        return render_template("login.html", info="Signup successful. Please login.")

    return render_template("signup.html")

# ---------- Login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        next_url = request.args.get("next") or url_for("prediction_page")

        users = load_users()
        found = None
        for u in users.get("users", []):
            if u.get("email") == email and u.get("password") == password:
                found = u
                break

        if not found:
            return render_template("login.html", error="Invalid email or password")

        session["email"] = email
        session["name"] = found.get("name") or email.split("@")[0]
        return redirect(next_url)

    return render_template("login.html")

# ---------- Logout ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------- Prediction page & POST ----------
# support GET to show form and POST to compute (keeps forms that post to /prediction working)
@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction_page():
    if request.method == "POST":
        # unify handling of both /prediction and /predict posting forms
        try:
            # robust form field reading (many templates used slightly different names)
            gender = request.form.get("gender") or request.form.get("Gender")
            married = request.form.get("married") or request.form.get("Married")
            dependents = request.form.get("dependents") or request.form.get("Dependents") or "0"
            education = request.form.get("education") or request.form.get("Education")
            employed = request.form.get("employed") or request.form.get("Self_Employed")
            # credit field might be hidden 'credit' or 'Credit_History' or raw slider
            credit_raw = request.form.get("credit") or request.form.get("Credit_History") or request.form.get("creditSlider") or 0
            try:
                credit_val = float(credit_raw)
            except Exception:
                credit_val = 0.0

            area = request.form.get("area") or request.form.get("Property_Area")
            ApplicantIncome = float(request.form.get("ApplicantIncome") or 0)
            CoapplicantIncome = float(request.form.get("CoapplicantIncome") or 0)
            LoanAmount = float(request.form.get("LoanAmount") or 0)
            Loan_Amount_Term = float(request.form.get("Loan_Amount_Term") or 0)

            features = preprocess_data(
                gender, married, dependents, education, employed,
                credit_val, area,
                ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
            )

            if MODEL is None:
                return render_template("prediction.html", prediction_text="Error: Model not loaded on server.")

            raw_pred = MODEL.predict([features])[0]
            try:
                proba = MODEL.predict_proba([features])[0].tolist()
            except Exception:
                proba = None

            rp_str = str(raw_pred).upper()
            if rp_str in ["Y", "YES", "1", "TRUE"]:
                prediction_text = "Loan Status is YES — You are Eligible ✔"
            elif rp_str in ["N", "NO", "0", "FALSE"]:
                prediction_text = "Loan Status is NO — You are Not Eligible ✘"
            else:
                if proba is not None and hasattr(MODEL, "classes_"):
                    try:
                        classes = list(MODEL.classes_)
                        idx = int(np.argmax(proba))
                        chosen = classes[idx]
                        prediction_text = f"Model raw: {raw_pred} -> interpreted as {chosen}"
                    except Exception:
                        prediction_text = f"Model raw: {raw_pred}"
                else:
                    prediction_text = f"Model raw prediction: {raw_pred}"

            return render_template("prediction.html", prediction_text=prediction_text, features_sent=features, raw_pred=raw_pred, proba=proba)

        except Exception as e:
            return render_template("prediction.html", prediction_text="Error during prediction: " + str(e))

    # GET: show form
    return render_template("prediction.html")

# also accept POST to /predict (compatibility)
@app.route("/predict", methods=["POST"])
@login_required
def predict_compat():
    # simply forward form data to prediction_page flow
    return prediction_page()

# ---------- Chatbot pages ----------
@app.route("/chatbot")
@login_required
def chatbot_page():
    # reset chat state (preserve previous behavior)
    session["messages"] = []
    session["chat_step"] = -1
    session["answers"] = {}
    session.modified = True

    add_message("assistant", "Hello! I'm your loan eligibility assistant. Type 'Yes' to begin.")
    return render_template("chatbot.html", messages=session["messages"])

@app.route("/chatbot-response", methods=["POST"])
@login_required
def chatbot_response():
    ensure_session()
    try:
        data = request.get_json() or {}
        user_text = data.get("message", "").strip()
        if not user_text:
            return jsonify({"status": "error", "error": "Empty message"}), 400

        add_message("user", user_text)

        chat_step = session.get("chat_step", -1)
        answers = session.get("answers", {})

        if chat_step == -1:
            if user_text.lower() in ("yes", "y", "ready"):
                session["chat_step"] = 0
                add_message("assistant", QUESTIONS[0])
                return jsonify({"reply": QUESTIONS[0], "messages": session["messages"]})
            else:
                add_message("assistant", "Type 'Yes' when you're ready to begin.")
                return jsonify({"reply": "Type 'Yes' when you're ready to begin.", "messages": session["messages"]})

        current_idx = session.get("chat_step", 0)
        answers[str(current_idx)] = user_text
        session["answers"] = answers
        session.modified = True

        if current_idx in NUMERIC_STEPS:
            try:
                float(user_text)
            except ValueError:
                add_message("assistant", "Please enter a valid number for this question.")
                return jsonify({"reply": "Please enter a valid number for this question.", "messages": session["messages"]})

        if current_idx == CREDIT_STEP:
            try:
                score = float(user_text)
                if not (0 <= score <= 1000):
                    add_message("assistant", "Credit score must be between 0 and 1000.")
                    return jsonify({"reply": "Credit score must be between 0 and 1000.", "messages": session["messages"]})
            except ValueError:
                add_message("assistant", "Please enter a valid number for credit score.")
                return jsonify({"reply": "Please enter a valid number for credit score.", "messages": session["messages"]})

        next_idx = current_idx + 1
        session["chat_step"] = next_idx

        if next_idx < len(QUESTIONS):
            q = QUESTIONS[next_idx]
            add_message("assistant", q)
            return jsonify({"reply": q, "messages": session["messages"]})

        # finished: run prediction from answers
        answers_int_keys = {int(k): v for k, v in answers.items()}
        result = run_prediction_from_answers(answers_int_keys)
        if "error" in result:
            add_message("assistant", "Error processing prediction: " + result.get("error"))
            return jsonify({"reply": "Error processing prediction: " + result.get("error"), "messages": session["messages"]}), 500

        prediction_label = result.get("prediction", "Unknown")
        if prediction_label == "Eligible":
            reply = "🎉 Congratulations! Based on the provided info you appear eligible for the loan."
        else:
            reply = "❌ Based on the provided info you appear NOT eligible. Consider improving credit, decreasing requested amount, or adding a co-applicant."

        add_message("assistant", reply)
        if prediction_label == "Eligible":
            advice = "Next steps: prepare ID, income proofs, property docs if needed, then contact lender."
        else:
            advice = "Recommendations: increase credit score, reduce loan amount, increase income or add co-applicant."
        add_message("assistant", advice)

        # reset chat so user can run again
        session["chat_step"] = -1
        session["answers"] = {}
        session.modified = True

        return jsonify({"reply": reply + "\n\n" + advice, "messages": session["messages"]})

    except Exception as e:
        tb = traceback.format_exc()
        add_message("assistant", "Internal server error: " + str(e))
        return jsonify({"status": "error", "error": str(e), "trace": tb}), 500

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
