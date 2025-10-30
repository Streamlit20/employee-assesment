import os
import streamlit as st
import pandas as pd
from io import BytesIO
from openai import AzureOpenAI
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
import json

# Function to load employee data from Excel
def load_employee_data():
    # Ensure the file exists, or create an empty one if not (for initial run)
    if not os.path.exists("employee_data_with_assessments.xlsx"):
        st.error("employee_data_with_assessments.xlsx not found. Please create it.")
        st.stop() # Stop the app if crucial file is missing
    employee_df = pd.read_excel("employee_data_with_assessments.xlsx")  # Load the employee data
    return employee_df


# Function to load the OpenAI chat model
client = AzureOpenAI(
    azure_endpoint="https://summarize-gen.openai.azure.com/",
    api_key="1DkyvFGwRKbcWKFYxfGAot2s8Qc9UPM8NmsbJR2OJDWJTBs084usJQQJ99ALACHYHv6XJ3w3AAABACOGvGzF",
    api_version="2025-01-01-preview"
)
    

# AI Model to generate assessment and test cases based on the topic
def generate_assessment(topic):
    SYSTEM_PROMPT = f"""
    You are an AI-powered system that creates coding assessments.

    TASK:
    Create a detailed coding assessment for the following topic: "{topic}"
    - The assessment should contain a problem statement, instructions, and a set of test cases not more than 4.
    - The problem statement should be clear and concise.
    - The test cases should validate the correctness of a potential solution.
    - The output should be a valid JSON object with the following structure:
        {{
            "problem_statement": "A detailed description of the problem.",
            "test_cases": [
                {{"input": "...", "expected_output": "..."}},
                {{"input": "...", "expected_output": "..."}}
            ]
        }}
    """

    # Call OpenAI's model to generate the assessment
    response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
            ],
            max_completion_tokens=100000,
            reasoning_effort="low"
        )

    response_content = response.choices[0].message.content

    try:
        assessment_data = json.loads(response_content)
        return assessment_data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing assessment details from AI: {e}")
        st.code(response_content) # Show the raw response for debugging
        return {"problem_statement": "Error: Could not load assessment.", "test_cases": []}


# Function to evaluate the code using OpenAI
def evaluate_code(problem_statement, code, test_cases):
    
    test_cases_str = json.dumps(test_cases, indent=2)

    SYSTEM_PROMPT = f"""
    You are an AI-powered code evaluator.

    TASK:
    Evaluate the following code for correctness based on the test cases provided for the problem: "{problem_statement}"

    - The code should be assessed based on the test cases:
      - If the code passes all test cases, return a JSON object with "evaluation_result": "PASSED".
      - If the code fails any test case, return a JSON object with "evaluation_result": "FAILED" and indicate the specific test case(s) that failed.
      - The output should be a valid JSON object with the following structure:
        {{
            "evaluation_result": "PASSED" or "FAILED",
            "failed_test_cases": [], // A list of test case names or identifiers that failed (if any).
            "feedback": "Any feedback on the code for improvement."
        }}

    INPUT CODE:
    ```
    {code}
    ```

    TEST CASES:
    ```json
    {test_cases_str}
    ```
    """
    
    # Call OpenAI's model to evaluate the code
    response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
            ],
            max_completion_tokens=100000,
            reasoning_effort="low"
        )
    evaluation_result_json_str = response.choices[0].message.content

    try:
        evaluation_data = json.loads(evaluation_result_json_str)
        return evaluation_data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing evaluation results from AI: {e}")
        st.code(evaluation_result_json_str) # Show the raw response for debugging
        return {"evaluation_result": "FAILED", "failed_test_cases": ["Parsing Error"], "feedback": f"Could not parse AI evaluation: {e}"}

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Employee Assessment Portal", layout="wide")

# Persistent state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "login"

# Hero Card Component
def display_hero_card():
    st.markdown(
        """
        <style>
        .hero-card {
            background-color: #e6f0ff; /* Light blue from the image */
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .hero-card h1 {
            color: #1a1a1a; /* Very dark gray/black for the title */
            font-size: 3em;
            margin-bottom: 10px;
        }
        .hero-card p {
            color: #333333; /* Slightly lighter dark gray for subtitle */
            font-size: 1.2em;
        }
        </style>
        <div class="hero-card">
            <h1>Sonata AI Assessment Handler</h1>
            <p>Your intelligent partner for coding assessments</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Login Page ---
if st.session_state.page == "login":
    display_hero_card()
    st.title("Employee Login")

    email = st.text_input("Enter your Email ID:")

    if st.button("Login"):
        if email:
            employee_df = load_employee_data()
            employee = employee_df[employee_df['Email ID'] == email]

            if not employee.empty:
                st.session_state.email = email
                st.session_state.name = employee.iloc[0]['Name']
                st.session_state.assigned_assessment = employee.iloc[0]['Assigned Assessment']
                st.session_state.page = "assessment"
                st.success("Login Successful! Redirecting to the assessment page.")
                st.rerun()
            else:
                st.error("Email not found. Please check your email ID.")

# --- Assessment Page ---
elif st.session_state.page == "assessment":
    if "email" not in st.session_state:
        st.warning("Please log in first.")
        st.session_state.page = "login" # Redirect to login if not logged in
        st.rerun()
    else:
        display_hero_card()
        st.title(f"Welcome {st.session_state.name} - Assessment Page")

        assigned_assessment = st.session_state.assigned_assessment

        @st.cache_data(show_spinner="Generating assessment details...")
        def get_assessment_details(topic):
            return generate_assessment(topic)

        assessment_details = get_assessment_details(assigned_assessment)
        
        if assessment_details.get("problem_statement") == "Error: Could not load assessment.":
            st.error("There was an error generating the assessment. Please try again or contact support.")
            if st.button("Go to Login"):
                st.session_state.page = "login"
                st.rerun()
        else:
            st.subheader("Problem Statement:")
            st.markdown(assessment_details['problem_statement'])
            
            st.sidebar.subheader("Test Cases:")
            if assessment_details['test_cases']:
                for i, test_case in enumerate(assessment_details['test_cases']):   
                    st.sidebar.markdown(f"**Test Case {i+1}**")
                    st.sidebar.markdown(f"**Input**: `{test_case.get('input', 'N/A')}`")
                    st.sidebar.markdown(f"**Expected Output**: `{test_case.get('expected_output', 'N/A')}`")
            else:
                st.info("No test cases provided for this assessment.")

            c1, c2 = st.columns([2, 1])

            with c1:
                code = st_ace(
                    language=c2.selectbox("Language mode", options=LANGUAGES, index=121),
                    theme=c2.selectbox("Theme", options=THEMES, index=35),
                    keybinding=c2.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
                    font_size=c2.slider("Font size", 5, 24, 14),
                    tab_size=c2.slider("Tab size", 1, 8, 4),
                    show_gutter=c2.checkbox("Show gutter", value=True),
                    show_print_margin=c2.checkbox("Show print margin", value=False),
                    wrap=c2.checkbox("Wrap enabled", value=False),
                    auto_update=c2.checkbox("Auto update", value=False),
                    min_lines=45,
                    key="ace",
                )

            if st.button("Submit Assessment"):
                st.session_state.submitted_code = code
                st.session_state.problem_statement = assessment_details["problem_statement"]
                st.session_state.test_cases = assessment_details["test_cases"]
                st.session_state.page = "submission_processing"
                st.rerun()

# --- Submission Processing Page (Intermediate step for UI transition) ---
elif st.session_state.page == "submission_processing":
    display_hero_card()
    
    # Check if the required data is present
    if "submitted_code" not in st.session_state:
        st.warning("Missing submission data. Returning to assessment page.")
        st.session_state.page = "assessment"
        st.rerun()

    # Use st.status to visually indicate a long-running task
    with st.status("**:blue[Processing your submission and evaluating code with AI...]**", expanded=True) as status:
        st.write("Initializing evaluation process...")
        
        try:
            # 1. Evaluate the code
            st.write("Calling AI model for code evaluation...")
            evaluation_result = evaluate_code(
                st.session_state.problem_statement,
                st.session_state.submitted_code,
                st.session_state.test_cases
            )

            st.write("Evaluation complete. Result received.")
            
            # 2. Load, update, and save the employee data
            st.write("Updating assessment results in employee data file...")
            employee_df = load_employee_data()
            
            # Ensure 'Assessment Result' column exists if it's the first time
            if 'Assessment Result' not in employee_df.columns:
                 employee_df['Assessment Result'] = None

            # Update the result for the logged-in user
            employee_df.loc[employee_df['Email ID'] == st.session_state.email, 'Assessment Result'] = evaluation_result['evaluation_result']
            
            # Save the updated DataFrame
            employee_df.to_excel("employee_data_with_results.xlsx", index=False)
            st.write("Results successfully saved to employee_data_with_results.xlsx.")
            
            # 3. Store results and cleanup
            st.session_state.final_evaluation_result = evaluation_result
            
            # Clean up session state for submission details
            for key in ["submitted_code", "problem_statement", "test_cases"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            status.update(label="**:green[Submission successfully processed!]**", state="complete", expanded=False)
            
            # 4. Transition to the final page
            st.session_state.page = "submission_complete"
            st.rerun()

        except Exception as e:
            status.update(label="**:red[Submission Processing FAILED]**", state="error", expanded=True)
            st.error(f"An error occurred during processing: {e}")
            st.button("Try Submitting Again", on_click=lambda: st.session_state.update(page="assessment"))
            # Keep on this page to show the error.


# --- Submission Complete Page ---
elif st.session_state.page == "submission_complete":
    display_hero_card()
    st.success("AI analysis is completed and the Test results are stored. You will be notified on assessment results soon.")
    st.info("Thank you for completing the assessment!")

    # Optionally display a summary of the result without raw data or download
    if "final_evaluation_result" in st.session_state:
        st.subheader("Your Submission Status:")
        if st.session_state.final_evaluation_result.get("evaluation_result") == "PASSED":
            st.markdown("**:green[Passed]**")
        else:
            st.markdown("**:red[Failed]**")
        
        # Don't show detailed feedback/test cases as per request to keep it simple.
        # st.markdown(f"Further details will be provided via official notification.")

    if st.button("Return to Login"):
        # Clear specific session state variables to ensure a fresh login
        for key in ["email", "name", "assigned_assessment", "final_evaluation_result"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = "login"

        st.rerun()
