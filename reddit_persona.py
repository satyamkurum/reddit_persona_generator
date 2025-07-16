
import json
import praw
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
from IPython.display import FileLink


import os
from dotenv import load_dotenv
load_dotenv()


gemini_api = os.getenv("GEMINI_API_KEY")
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")


reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)



def fetch_reddit_data(username, limit=50):
    redditor = reddit.redditor(username)

    user_texts = []

    # Get submissions (posts)
    for submission in redditor.submissions.new(limit=limit):
        content = f"Title: {submission.title}\n{submission.selftext}"
        user_texts.append({
            "type": "post",
            "content": content,
            "url": f"https://www.reddit.com{submission.permalink}"
        })

    # Get comments
    for comment in redditor.comments.new(limit=limit):
        user_texts.append({
            "type": "comment",
            "content": comment.body,
            "url": f"https://www.reddit.com{comment.permalink}"
        })

    return user_texts



data = fetch_reddit_data("kojied")  # test user

def save_reddit_data_to_txt(user_texts, username):
    output_file = f"{username}_raw_data.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(user_texts, 1):
            f.write(f"{'='*60}\n")
            f.write(f"{i}. Type: {item['type']}\n")
            f.write(f"URL: {item['url']}\n")
            f.write(f"Content:\n{item['content']}\n")
            f.write(f"{'='*60}\n\n")

    print(f"Data saved to: {output_file}")



username = "kojied"
data = fetch_reddit_data(username)
save_reddit_data_to_txt(data, username)



from langchain.text_splitter import CharacterTextSplitter

def preprocess_and_chunk(user_texts, chunk_size=1000, chunk_overlap=100):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    chunks_with_metadata = []

    for item in user_texts:
        content = item["content"].strip()

        if not content or len(content) < 30:
            continue

        chunks = text_splitter.split_text(content)

        for chunk in chunks:
            chunks_with_metadata.append({
                "text": chunk,
                "source": item["url"],
                "type": item["type"]
            })

    print(f" Created {len(chunks_with_metadata)} chunks from {len(user_texts)} posts/comments.")
    return chunks_with_metadata


chunks = preprocess_and_chunk(data)


genai.configure(api_key=gemini_api)

model = genai.GenerativeModel("models/gemini-2.5-flash")



def combine_all_chunks(chunks):
    return "\n\n".join([chunk["text"] for chunk in chunks])



def generate_single_persona(chunks, username):
    all_text = combine_all_chunks(chunks)

    prompt = f"""
You are an AI assistant building a complete Reddit user persona based on public activity.

Below is the user's Reddit content:
--------------------
{all_text}
--------------------

Generate a clean and concise user persona using the following structure:

Reddit Persona: @{username}

Core Traits:
- Trait 1
- Trait 2

Interests:
- Interest 1
- Interest 2

Tone & Communication Style:
- Style 1
- Style 2

Profession Hints:
- Hint 1
- Hint 2

Beliefs:
- Belief 1
- Belief 2

Do not repeat ideas. Do not use emojis or bold text. Do not include any batch-wise phrasing.
Keep the output minimal, well-formatted, and accurate.
End with: 'This persona is based on publicly visible Reddit activity.'
"""

    response = model.generate_content(prompt)
    return response.text.strip()



def save_persona_output(persona_text, username):
    file_name = f"{username}_persona_portfolio.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(persona_text)
    print(f"Final persona saved as: {file_name}")


persona = generate_single_persona(chunks, username="kojied")
save_persona_output(persona, username="kojied")

from IPython.display import FileLink
FileLink("kojied_persona_portfolio.txt")
