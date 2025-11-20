import gradio as gr
from rag_pipeline import ingest_cv_build_chain, ask_question


def on_upload(file):
    """
    file: gradio temp file, e.g. <tempfile> with .name
    We'll open it in binary mode and ingest.
    """
    with open(file.name, "rb") as f:
        session = ingest_cv_build_chain(f)

    profile_md = session["profile_md"]
    empty_chat = []
    return session, profile_md, empty_chat


def on_ask(user_question, chat_history, session_state):
    """
    Handles the user's question in the chat interface.
    Checks if a session exists, asks the question using the RAG pipeline,
    and updates the chat history.

    Args:
        user_question (str): The question asked by the user.
        chat_history (list): The current chat history.
        session_state (dict): The session state containing the QA chain.

    Returns:
        tuple: Updated chat history, session state, and an empty string to clear the input box.
    """
    if session_state is None:
        # No CV uploaded yet
        chat_history = chat_history + [("You", user_question),
                                       ("Assistant", "Please upload a CV first.")]
        return chat_history, session_state, ""

    answer = ask_question(session_state, user_question)
    chat_history = chat_history + [("You", user_question), ("Assistant", answer)]
    return chat_history, session_state, ""


with gr.Blocks(title="CV Insight RAG (LangChain)") as demo:
    gr.Markdown("# CV Insight RAG (LangChain)\nUpload a CV, get skills summary, ask targeted questions.")

    session_state = gr.State(value=None)  # holds { qa_chain, profile_md }
    chat_state = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload CV (PDF)", file_types=[".pdf"])
            profile_box = gr.Markdown("**No CV uploaded yet.**")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="CV Q&A", type="tuple")
            user_msg = gr.Textbox(label="Ask something about this CV")
            ask_btn = gr.Button("Ask")

    # When file changes -> build new session_state, show profile, reset chat
    file_input.change(
        fn=on_upload,
        inputs=[file_input],
        outputs=[session_state, profile_box, chat_state],
    )

    # Ask question
    ask_btn.click(
        fn=on_ask,
        inputs=[user_msg, chat_state, session_state],
        outputs=[chatbot, session_state, user_msg],
    )

    # Mirror chat_state into chatbot if needed
    chatbot.change(lambda h: h, inputs=[chat_state], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()
