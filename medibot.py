import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Cache vector store to improve performance
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Set up a custom prompt
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load Hugging Face model
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

# Streamlit UI
def main():
    st.write("<div style='text-align: center; font-size: 3em; font-weight: bold;'>🤖 MediBot - Your AI Medical Assistant</div>", unsafe_allow_html=True)

    
    # Button to clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input("Type your medical question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the information in the context to answer the user's question.
        If the answer is unknown, say "I don't know" and do not invent an answer.
        Do not provide any information beyond the given context.

        Context: {context}
        Question: {question}

        Answer:
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),  # Reduced k for faster response
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Show typing indicator
            message_placeholder = st.chat_message('assistant').markdown("🤖 *MediBot is typing...*")
            
            # Get response
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            # Format the response
            message_placeholder.markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

            # Show sources in an expandable section
            with st.expander("📚 Source Documents"):
                for doc in source_documents:
                    st.markdown(f"- {doc}")

        except Exception as e:
            st.error("⚠️ An error occurred. Please try again later.")
            st.write(f"Technical details: `{str(e)}`")  # Optional for debugging

if __name__ == "__main__":
    main()
