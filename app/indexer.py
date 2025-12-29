import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv(find_dotenv())

DB_DIR = "chroma_db"
TARGET_URL = "https://calamitymod.wiki.gg/wiki/Supreme_Calamitas"

def scrape_page(url: str) -> str:
    print(f"Scraping {url}...")
    try:
        response = requests.get(url, headers={"User-Agent": "CalamityBot/1.0"})
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    content_div = soup.find(id = "mw-content-text")
    if not content_div:
        print(f"No content found on {url}")
        return ""

    for element in content_div.select("script, style, .mw-editsection, #toc"):
        element.decompose()

    text = content_div.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join([line for line in lines if line])

    return clean_text

def index_data(text: str, url: str):

    if not text:
        print("No text to index")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    print(f"Generated {len(chunks)} chunks")

    documents = [Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]

    print("Initializing Vector Store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store=Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="calamity_mod",
        persist_directory=DB_DIR,
    )

    print(f"Successfully saved {len(chunks)} chunks to {DB_DIR}")

if __name__ == "__main__":
    raw_text = scrape_page(TARGET_URL)
    index_data(raw_text, TARGET_URL)