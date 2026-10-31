
import streamlit as st
from main import run_pipeline
from core.rag_engine import ask_question


# Page configuration
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide"
)

# Session state
if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Header
st.title("🎬 AI Video Assistant")
st.caption("Summarize videos, extract important information, and chat with your videos.")

st.divider()


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    language = st.selectbox(
        "Select Transcription Language",
        ["english", "hinglish"]
    )

    st.divider()

    st.info(
        "Enter a YouTube URL or a local audio/video file path "
        "to get started."
    )


# Video input
st.subheader("📺 Process Your Video")

with st.form("video_form"):
    source = st.text_input(
        "YouTube URL or Local File Path",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    process_button = st.form_submit_button(
        "🚀 Process Video",
        use_container_width=True
    )


# Process video
if process_button:

    if not source.strip():
        st.warning("Please enter a YouTube URL or file path.")

    else:
        try:
            with st.spinner(
                "Processing video... Downloading, transcribing and generating summary."
            ):
                result = run_pipeline(
                    source=source.strip(),
                    language=language
                )

                st.session_state.result = result
                st.session_state.chat_history = []

            st.success("Video processed successfully!")

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")


# Display results
if st.session_state.result:

    result = st.session_state.result

    st.divider()

    st.header(f"📌 {result['title']}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Summary",
        "✅ Action Items",
        "🔑 Decisions",
        "❓ Questions",
        "📄 Transcript"
    ])

    with tab1:
        st.subheader("Video Summary")
        st.write(result["summary"])

    with tab2:
        st.subheader("Action Items")
        st.write(result["action_items"])

    with tab3:
        st.subheader("Key Decisions")
        st.write(result["key_decisions"])

    with tab4:
        st.subheader("Open Questions")
        st.write(result["open_questions"])

    with tab5:
        st.subheader("Full Transcript")
        st.text_area(
            "Transcript",
            value=result["transcript"],
            height=400
        )

    # Chat section
    st.divider()
    st.header("💬 Chat With Your Video")

    st.caption(
        "Ask questions about the video. Answers are generated using your RAG pipeline."
    )

    # Display previous messages
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input("Ask something about this video...")

    if question:

        # Display user question
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:
                    answer = ask_question(
                        result["rag_chain"],
                        question
                    )

                    st.markdown(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
