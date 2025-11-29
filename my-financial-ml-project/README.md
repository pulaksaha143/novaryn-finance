💰 NOVARYN: Multilingual Financial Advisor Chatbot

NOVARYN is a self-contained financial modeling and advisory tool built using Gradio and powered by the Google Gemini API. It allows users to input their fixed monthly expenses, upload a bank statement CSV for variable expense analysis, and receive personalized financial advice in real-time, all within a responsive web interface.

The application also includes Text-to-Speech (TTS) capabilities to vocalize the advisor's responses, offering a fully accessible experience.

✨ Features

Multilingual Support: The entire UI and the AI's financial advice can be toggled between English, Hindi (हिंदी), and Marathi (मराठी).

Expense Prediction (ML Model): Uses a pre-trained Linear Regression model to predict the next month's variable expenses based on the total amount from an uploaded bank statement CSV.

Comprehensive Financial Snapshot: Calculates total expenses, monthly surplus, and provides advice based on the 30/30/30/10 Allocation Rule (Emergency, Retirement, Goals, Flexible Savings).

AI Financial Advisor: Utilizes the gemini-2.5-flash-preview-09-2025 model with Google Search Grounding to provide up-to-date, practical, and personalized financial strategies.

Text-to-Speech (TTS): Integrates the gemini-2.5-flash-preview-tts model to read out the generated financial advice.

Custom Expense Input: Allows users to input non-standard fixed expenses (e.g., "Gym 1500", "PetCare:500").

🛠️ Setup and Installation

Prerequisites

Python 3.9+

A Gemini API Key (for both text generation and TTS services).

Step 1: Clone the Repository (Simulated)

Assuming this code is saved as financial_model.py:

# In a real environment, you would clone the repo
# git clone <repository-url>
# cd novaryn-finance


Step 2: Create a Virtual Environment and Install Dependencies

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt


requirements.txt content:

gradio
pandas
numpy
scikit-learn
requests
python-dotenv
joblib


Step 3: Configure API Key

Create a file named .env in the same directory as financial_model.py and add your Gemini API Key:

GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"


The script is configured to use the os.getenv function for API key retrieval.

🚀 Usage

1. Run the Application

Execute the Python script to start the Gradio server:

python financial_model.py


The application will launch and typically open in your browser at http://127.0.0.1:7860/.

2. Enter Financial Data

Select Language: Choose your preferred language (English, Hindi, or Marathi).

Input Income and Fixed Expenses: Fill in your average monthly income and fixed costs like Rent, EMI, Debt, Food, and Subscriptions.

Add Custom Expenses: Use the Other Fixed Expenses box for unlisted costs (e.g., Car Insurance 2000, Tuition 5000).

Upload CSV (Optional): Upload a bank statement CSV. The script expects columns like Date, Description, and Amount. The total of the Amount column is used as the basis for the variable expense prediction.

3. Get Initial Overview

Click the "Upload & Get Initial Financial Overview" button. The application will:

Predict your next month's variable expense.

Calculate your surplus/deficit.

Display a comprehensive financial snapshot and allocation advice from the AI advisor.

4. Chat with the Advisor

Use the Financial Chatbot section to ask specific questions about savings, investments, or debt management. The advisor will use your current financial data as context to provide highly relevant and actionable advice.

🧠 Model and Logic

The application uses two primary models:

1. Expense Prediction (ML)

Model: sklearn.linear_model.LinearRegression

Training: A synthetic dataset of 24 months of mock expenses is generated and trained on the prior month's expense to predict the current month's expense. This simulates a basic time-series forecast for variable spending.

Use: Predicts the "Variable Expense" component of the monthly budget.

2. Financial Advice (LLM)

Model: gemini-2.5-flash-preview-09-2025

Function: Used for all chat responses, summaries, and actionable advice.

Grounding: Utilizes Google Search grounding to ensure the investment and saving strategies are based on current, real-world data and best practices.

3. Text-to-Speech (TTS)

Model: gemini-2.5-flash-preview-tts

Function: Converts the advisor's markdown response into PCM audio, which is then wrapped in a WAV file container for playback in the Gradio interface.