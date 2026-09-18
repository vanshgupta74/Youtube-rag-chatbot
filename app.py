import streamlit as st
import os

from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎥 YouTube RAG Chatbot")

st.write(
    "Enter a YouTube video ID and ask questions about the video."
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "main_chain" not in st.session_state:
    st.session_state.main_chain = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Video ID Input
# --------------------------------------------------

video_id = st.text_input(
    "Enter YouTube Video ID",
    placeholder="Example: x63HCoDfAhQ"
)


# --------------------------------------------------
# Load Video
# --------------------------------------------------

if st.button("Load Video"):

    if not video_id:
        st.warning("Please enter a YouTube Video ID.")

    elif not HF_TOKEN:
        st.error(
            "Hugging Face API token not found. "
            "Please add HUGGINGFACEHUB_API_TOKEN to your .env file."
        )

    else:

        with st.spinner("Loading video and building RAG pipeline..."):

            try:

                # --------------------------------------------------
                # Step 1 - Fetch Transcript
                # --------------------------------------------------

                api = YouTubeTranscriptApi()

                transcript_list = api.fetch(
                    video_id,
                    languages=["en"]
                )

                transcript = " ".join(
                    chunk.text for chunk in transcript_list
                )


                # --------------------------------------------------
                # Step 1b - Chunking
                # --------------------------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.create_documents(
                    [transcript]
                )


                # --------------------------------------------------
                # Step 1c & 1d - Embeddings + FAISS
                # --------------------------------------------------

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                vector_store = FAISS.from_documents(
                    chunks,
                    embeddings
                )


                # --------------------------------------------------
                # Step 2 - Retrieval
                # --------------------------------------------------

                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )


                # --------------------------------------------------
                # Step 3 - LLM
                # --------------------------------------------------

                llm = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-8B-Instruct",
                    huggingfacehub_api_token=HF_TOKEN,
                    temperature=0.2,
                    max_new_tokens=512
                )

                llm = ChatHuggingFace(
                    llm=llm
                )


                # --------------------------------------------------
                # Prompt
                # --------------------------------------------------

                prompt = PromptTemplate(
                    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
                    input_variables=[
                        "context",
                        "question"
                    ]
                )


                # --------------------------------------------------
                # Format retrieved documents
                # --------------------------------------------------

                def format_docs(retrieved_docs):

                    context_text = "\n\n".join(
                        doc.page_content
                        for doc in retrieved_docs
                    )

                    return context_text


                # --------------------------------------------------
                # Building Chain
                # --------------------------------------------------

                parallel_chain = RunnableParallel({

                    "context": (
                        retriever
                        | RunnableLambda(format_docs)
                    ),

                    "question": RunnablePassthrough()

                })


                parser = StrOutputParser()


                main_chain = (
                    parallel_chain
                    | prompt
                    | llm
                    | parser
                )


                # --------------------------------------------------
                # Store chain in session state
                # --------------------------------------------------

                st.session_state.main_chain = main_chain

                st.session_state.video_id = video_id

                st.session_state.messages = []


                st.success("Video loaded successfully! 🎉")

                st.info(
                    f"Video ID: {video_id}"
                )


            except TranscriptsDisabled:

                st.error(
                    "Captions are disabled for this video."
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )


# --------------------------------------------------
# Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask something about the video..."
)


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

if question:

    if st.session_state.main_chain is None:

        st.warning(
            "Please load a YouTube video first."
        )

    else:

        # Show user question
        with st.chat_message("user"):

            st.markdown(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })


        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = st.session_state.main_chain.invoke(
                        question
                    )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        f"Error generating answer: {str(e)}"
                    )