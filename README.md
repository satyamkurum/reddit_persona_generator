# Reddit Persona Generator

The **Reddit Persona Generator** analyzes a Reddit user's recent activity (posts & comments) and uses Generative AI to generate a concise psychological and behavioral persona of that user.

---

##  Features

-  Fetches Reddit user data using the PRAW API
-  Cleans and splits data for efficient input to LLMs
-  Uses Google Gemini Pro (via `google-generativeai`) for persona generation
-  Leverages LangChain for structured prompt handling
-  Easy-to-use Streamlit interface (or CLI)
-  Supports secure `.env` secrets handling

---

##  Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```
Create a .env file in the root directory:
-REDDIT_CLIENT_ID=your_reddit_client_id
-REDDIT_CLIENT_SECRET=your_reddit_client_secret
-GOOGLE_API_KEY=your_google_gemini_api_key

## Run
```bash
python reddit_persona.py
```
