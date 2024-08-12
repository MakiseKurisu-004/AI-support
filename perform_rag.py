from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import numpy as np
import tiktoken
import os
import sys
import google.generativeai as genai
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = pinecone_api_key
os.environ['USER_AGENT'] = 'CustomerSupport/1.0'

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedding = genai.embed_content(
    model="models/text-embedding-004",
    content="What is the meaning of life?",
    task_type="retrieval_document",
    title="Embedding of single string")

# 1 input > 1 vector output
# print(str(embedding['embedding'])[:50], '... TRIMMED]')

tokenizer = tiktoken.get_encoding('p50k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
)

def get_embedding(text):
    # Call the OpenAI API to get the embedding for the text
    embedding = genai.embed_content(
    model="models/text-embedding-004",
    content=text,
    task_type="retrieval_document",
    title="Embedding of single string")
    return embedding['embedding']

def cosine_similarity_between_words(sentence1, sentence2):
    # Get embeddings for both words
    embedding1 = np.array(get_embedding(sentence1))
    embedding2 = np.array(get_embedding(sentence2))

    # Reshape embeddings for cosine_similarity function
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # print("Embedding for Sentence 1:", embedding1)
    # print("\nEmbedding for Sentence 2:", embedding2)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


# Example usage
# sentence1 = "I like walking to the park"
# sentence2 = "I like walking to the office"


# similarity = cosine_similarity_between_words(sentence1, sentence2)
# print(f"\n\nCosine similarity between '{sentence1}' and '{sentence2}': {similarity:.4f}")



# # Loading YT video
# loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=DXm6E0Dcel0", add_video_info=True)
# # Analysis: Reacting to Donald Trump's press conference, debate news
# data = loader.load()

# print(data)

# texts = text_splitter.split_documents(data)




# # Loading embedding vector into Pinecone
# gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# vectorstore = PineconeVectorStore(index_name="support", embedding=gemini_embeddings)
index_name = "support"
namespace = "Trump-Press"

# for document in texts:
#     print("\n\n\n\n----")

#     print(document.metadata, document.page_content)

#     print('\n\n\n\n----')

# vectorstore_from_texts = PineconeVectorStore.from_texts([f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}" for t in texts], gemini_embeddings, index_name=index_name, namespace=namespace)


# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key,)

# Connect to your Pinecone index
pinecone_index = pc.Index("support")

def performRAG(query):

    query_embedding = get_embedding(query)

    top_matches = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace=namespace)

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    system_prompt = f"""You are an expert personal assistant. You always answer 
                        questions based only on the context that you have been provided.
                    """

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-001", system_instruction=system_prompt)

    response = model.generate_content(augmented_query)

    if response.candidates and response.candidates[0].content.parts:
        response_text = response.candidates[0].content.parts[0].text
        print(response_text)
        return response_text
    else:
        print("No valid response parts returned. Please check the safety ratings.")
        print("Safety Ratings:", response.candidates[0].safety_ratings)

if __name__ == '__main__':
    query = sys.argv[1]
    print(performRAG(query))
