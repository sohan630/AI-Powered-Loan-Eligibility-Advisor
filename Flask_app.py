from flask import Flask, request, render_template, jsonify, session
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)
# Set a proper secret key in production (env var)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
except Exception as e:
    print("Error loading model.pkl:", e)
    MODEL = None

# Questions (same order as streamlit)
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

NUMERIC_STEPS = {5, 6, 7, 8}  # indices of numeric questions
CREDIT_STEP = 9

# helpers
def ensure_session():
    if "messages" not in session:
        session["messages"] = []
    if "chat_step" not in session:
        session["chat_step"] = -1   # -1 means not started, else index of question last asked
    if "answers" not in session:
        session["answers"] = {}
    session.modified = True

def add_message(role, content):
    ensure_session()
    session["messages"].append({"role": role, "content": content})
    session.modified = True

def preprocess_data(gender, married, dependents, education, employed, credit, area,
                    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    """
    Build the feature vector expected by the model.

    credit can be:
     - binary 0/1 (already mapped by front-end), OR
     - raw slider score (0-1000). In that case we use threshold 850 to map to 1.
    """

    male = 1 if str(gender).strip().lower() == "male" else 0
    married_yes = 1 if str(married).strip().lower() == "yes" else 0

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

    # safe numeric conversion with fallback to 0
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

    # --- Robust credit handling ---
    # If credit is already 0/1 (binary), use it directly.
    # If it looks like a raw score (>1), use threshold 850 (matches front-end slider).
    try:
        credit_val = float(credit)
    except Exception:
        credit_val = 0.0

    if credit_val in (0.0, 1.0):
        credit_flag = int(credit_val)
    else:
        # assume raw slider 0-1000 -> treat >=850 as good
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

# routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction_page():
    if request.method == "POST":
        try:
            gender = request.form.get("gender")
            married = request.form.get("married")
            dependents = request.form.get("dependents")
            education = request.form.get("education")
            employed = request.form.get("employed")
            applicant_income = float(request.form.get("ApplicantIncome") or 0)
            coapplicant_income = float(request.form.get("CoapplicantIncome") or 0)
            loan_amount = float(request.form.get("LoanAmount") or 0)
            loan_amount_term = float(request.form.get("Loan_Amount_Term") or 0)
            # Ensure credit mapping consistent with training: use 850 threshold (frontend uses 850)
            credit_raw = request.form.get("credit") or 0
            # Try parse int/float - keep as-is (preprocess_data will interpret)
            try:
                credit_val = int(credit_raw)
            except Exception:
                try:
                    credit_val = float(credit_raw)
                except Exception:
                    credit_val = 0

            # area
            area = request.form.get("area")

            # Build features using preprocess_data (preprocess_data is robust to binary or raw score)
            features = preprocess_data(
                gender, married, dependents, education, employed,
                credit_val, area,
                applicant_income, coapplicant_income,
                loan_amount, loan_amount_term
            )

            # Prepare debug variables
            features_sent = features
            raw_pred = None
            proba = None
            prediction_text = "Error: model not loaded."
            if MODEL is None:
                prediction_text = "Error: Model not loaded on server."
            else:
                # raw prediction
                raw_pred = MODEL.predict([features])[0]

                # Try to get prediction probability if supported
                try:
                    proba = MODEL.predict_proba([features])[0].tolist()
                except Exception:
                    proba = None

                # Map raw_pred to user-friendly label robustly
                rp = raw_pred
                rp_str = str(rp).upper()

                if rp_str in ["Y", "YES", "1", "TRUE"]:
                    prediction_text = "Loan Status is YES — You are Eligible ✔"
                elif rp_str in ["N", "NO", "0", "FALSE"]:
                    prediction_text = "Loan Status is NO — You are Not Eligible ✘"
                else:
                    # Fallback: if predict_proba exists, choose the class with highest probability
                    if proba is not None:
                        try:
                            classes = list(MODEL.classes_)
                            idx = int(np.argmax(proba))
                            chosen = classes[idx]
                            prediction_text = f"Model raw: {raw_pred} -> interpreted as {chosen}"
                        except Exception:
                            prediction_text = f"Model raw: {raw_pred}"
                    else:
                        prediction_text = f"Model raw prediction: {raw_pred}"

            # Render template with debug info (template will still show clean prediction_text)
            return render_template(
                "prediction.html",
                prediction_text=prediction_text,
                features_sent=features_sent,
                raw_pred=raw_pred,
                proba=proba
            )

        except Exception as e:
            # Show error message in template
            return render_template("prediction.html", prediction_text="Error during prediction: " + str(e))

    # GET method
    return render_template("prediction.html")


# Chatbot page: pass messages (including initial assistant message)
@app.route("/chatbot")
def chatbot_page():

    # 🔥 CLEAR CHAT EVERY TIME PAGE LOADS
    session["messages"] = []
    session["chat_step"] = -1
    session["answers"] = {}
    session.modified = True

    # Fresh greeting
    add_message("assistant", "Hello! I'm your loan eligibility assistant. Type 'Yes' to begin.")

    return render_template("chatbot.html", messages=session["messages"])


# Chatbot message endpoint
@app.route("/chatbot-response", methods=["POST"])
def chatbot_response():
    ensure_session()
    try:
        data = request.get_json() or {}
        user_text = data.get("message", "").strip()
        if not user_text:
            return jsonify({"status": "error", "error": "Empty message"}), 400

        # Append user message to session messages
        add_message("user", user_text)

        # Determine current step
        # If chat_step == -1 -> not started
        chat_step = session.get("chat_step", -1)
        answers = session.get("answers", {})

        # If not started, expect "yes" to begin
        if chat_step == -1:
            if user_text.lower() in ("yes", "y", "ready"):
                # start at question 0
                session["chat_step"] = 0
                add_message("assistant", QUESTIONS[0])
                return jsonify({"reply": QUESTIONS[0], "messages": session["messages"]})
            else:
                add_message("assistant", "Type 'Yes' when you're ready to begin.")
                return jsonify({"reply": "Type 'Yes' when you're ready to begin.", "messages": session["messages"]})

        # If started: store response for the current question index
        current_idx = session.get("chat_step", 0)
        # store user answer for current_idx
        answers[str(current_idx)] = user_text
        session["answers"] = answers
        session.modified = True

        # Validate numeric fields
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

        # increment step
        next_idx = current_idx + 1
        session["chat_step"] = next_idx

        # If there is a next question, send it
        if next_idx < len(QUESTIONS):
            q = QUESTIONS[next_idx]
            add_message("assistant", q)
            return jsonify({"reply": q, "messages": session["messages"]})

        # Otherwise, we finished all questions — run prediction
        # Convert answers keys to ints
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
        # optionally add structured advice message
        if prediction_label == "Eligible":
            advice = "Next steps: prepare ID, income proofs, property docs if needed, then contact lender."
        else:
            advice = "Recommendations: increase credit score, reduce loan amount, increase income or add co-applicant."
        add_message("assistant", advice)

        # reset chat state so user can run again
        session["chat_step"] = -1
        session["answers"] = {}
        session.modified = True

        return jsonify({"reply": reply + "\n\n" + advice, "messages": session["messages"]})

    except Exception as e:
        tb = traceback.format_exc()
        add_message("assistant", "Internal server error: " + str(e))
        return jsonify({"status": "error", "error": str(e), "trace": tb}), 500

if __name__ == "__main__":
    app.run(debug=True)
