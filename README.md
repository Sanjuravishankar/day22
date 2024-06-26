Building RAG Applications


for documentation visit lablab.AI
AI toutorials
search
clarifai
https://lablab.ai/t/trulens-google-vertex-ai-tutorial-building-rag-applications
 


Grab Your Access Tokens

Clarifai Personal Access Token
Visit Clarifai: Head over to Clarifai's security settings page.
Get Your Token: Here, you'll find your personal access token. This is like a special password that lets your app talk to Clarifai's services. Copy this token.
OpenAI API Key
Go to OpenAI: Visit the OpenAI website and log into your account.
Retrieve Your Key: Find where they list your API key. This key is what allows your app to interact with OpenAI's powerful AI models.
hugging face: Visit the hugging face website and log into your account.
Retrieve Your Key: Find where they list your API key.


pip install longchain
pip install langchain-openai
pip install clarifai
pip install python-dotenv
pip install streamlit
pip install streamlit streamlit-chat cohere

3.create environment

4.env file

OPENAI_API_KEY=RoW3TACuTTMx8B7vBrFTnByB3YyMT5MJmbRpXscP
GOOGLE_API_KEY=https://console.cloud.google.com/apis/credentials?project=inner-nuance-427503-k1
HUGGINGFACE_API_KEY=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
 
CLARIFAI_PAT=d60537e4fc8242e69ff403a846a4aa86

5. pyhton file

 import os
import requests
import streamlit as st
import weaviate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatVertexAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from trulens_eval import Feedback, Huggingface, Tru, TruChain
from weaviate.embedded import EmbeddedOptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

# Initialize feedback and TruLens
hugs = Huggingface()
tru = Tru()

# Initialize variables
chain_recorder = None
conversation = None

# Function to handle conversation
def handle_conversation(user_input):
    input_dict = {"question": user_input}
    try:
        with chain_recorder as recording:
            response = conversation(input_dict)
            return response.get("answer", "No response generated.")
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit sidebar for configuration
st.sidebar.title("Configuration")
url = st.sidebar.text_input("Enter URL")
submit_button = st.sidebar.button("Submit")

# Initialize session state
if 'initiated' not in st.session_state:
    st.session_state['initiated'] = False
    st.session_state['messages'] = []

# Load and process the document when the button is clicked
if submit_button or st.session_state['initiated']:
    st.session_state['initiated'] = True

    if url and not conversation:
        # Fetch and save the document
        res = requests.get(url)
        with open("state_of_the_union.txt", "w") as f:
            f.write(res.text)

        # Load and split the document
        loader = TextLoader('./state_of_the_union.txt')
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Initialize Weaviate client and vector store
        client = weaviate.Client(embedded_options=EmbeddedOptions())
        vectorstore = Weaviate.from_documents(client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False)
        
        retriever = vectorstore.as_retriever()

        # Initialize LLM and conversation chain
        llm = ChatVertexAI()
        template = """You are an assistant for question-answering tasks..."""
        prompt = ChatPromptTemplate.from_template(template)
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

        # Initialize TruChain with feedbacks
        chain_recorder = TruChain(
            conversation,
            app_id="RAG-System",
            feedbacks=[
                Feedback(hugs.language_match).on_input_output(),
                Feedback(hugs.not_toxic).on_output(),
                Feedback(hugs.pii_detection).on_input(),
                Feedback(hugs.positive_sentiment).on_output(),
            ]
        )

# Streamlit app title
st.title("RAG System powered by TruLens and Vertex AI")

# Display chat messages
for message in st.session_state.messages:
    st.write(f"{message['role']}: {message['content']}")

# User input and response handling
user_prompt = st.text_input("Your question:", key="user_input")
if st.button("Send", key="send_button"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Generate and display assistant response
    assistant_response = handle_conversation(user_prompt)

    # Update and display chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.write(f"Assistant: {assistant_response}")
    tru.run_dashboard()
else:
    st.write("Please enter a URL and click submit to load the application.")


 

7.run --streamlit run day21.py


