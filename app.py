import gradio as gr
from rag_pipeline import ingest_cv_build_chain, ask_question, score_candidate_fit, extract_logistics, extract_experience_timeline


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

    with gr.Row():
        with gr.Column(scale=1):
            jd_input = gr.File(label="Upload Job Description (PDF/Text)", file_types=[".pdf", ".txt"])
            match_btn = gr.Button("Analyze Fit")
        
        with gr.Column(scale=2):
            fit_output = gr.JSON(label="Fit Analysis")

    with gr.Row():
        with gr.Column():
            logistics_box = gr.JSON(label="Logistics")
        with gr.Column():
            timeline_box = gr.JSON(label="Experience Timeline")

    # Session state now holds just the ID or we reload?
    # For simplicity, let's keep session_state as the chain for the *active* CV.
    # But we add a dropdown to switch.

    def refresh_doc_list():
        # List local faiss indexes
        import os
        data_dir = os.environ.get("DATA_DIR", "./data")
        if not os.path.exists(data_dir):
            return []
        return [d.replace("faiss_", "") for d in os.listdir(data_dir) if d.startswith("faiss_")]

    with gr.Row():
        doc_selector = gr.Dropdown(label="Select Existing Candidate", choices=refresh_doc_list())
        refresh_btn = gr.Button("Refresh List")

    def on_select_doc(doc_id):
        if not doc_id:
            return None, "**No CV selected**", []
        
        # Load FAISS
        import os
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI
        from rag_pipeline import build_prompt_template, build_profile_summary_markdown # We need raw text for profile... 
        
        # Problem: We didn't save raw text, only embeddings. 
        # We can't rebuild profile summary from FAISS easily without storing metadata.
        # For this prototype, we'll just load the QA chain and say "Profile not available (cached)".
        
        data_dir = os.environ.get("DATA_DIR", "./data")
        idx_path = os.path.join(data_dir, f"faiss_{doc_id}")
        
        if not os.path.exists(idx_path):
            return None, "Error: Index not found", []

        embeddings = OpenAIEmbeddings()
        vs = FAISS.load_local(idx_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, chain_type="stuff",
            chain_type_kwargs={"prompt": build_prompt_template()}, return_source_documents=False
        )
        
        return {"qa_chain": qa_chain}, f"**Loaded cached CV:** {doc_id}\n*(Full profile summary not available in cached mode)*", []

    doc_selector.change(
        fn=on_select_doc,
        inputs=[doc_selector],
        outputs=[session_state, profile_box, chat_state]
    )
    
    refresh_btn.click(
        fn=lambda: gr.update(choices=refresh_doc_list()),
        outputs=[doc_selector]
    )

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

    def on_match(cv_file, jd_file):
        if not cv_file or not jd_file:
            return {"error": "Please upload both CV and JD."}
        
        # Extract text
        from cv_parser import extract_text_from_pdf
        
        with open(cv_file.name, "rb") as f:
            cv_text = extract_text_from_pdf(f)
            
        jd_text = ""
        with open(jd_file.name, "rb") as f:
            if jd_file.name.endswith(".pdf"):
                jd_text = extract_text_from_pdf(f)
            else:
                jd_text = f.read().decode("utf-8", errors="ignore")
                
        return score_candidate_fit(cv_text, jd_text)

    match_btn.click(
        fn=on_match,
        inputs=[file_input, jd_input],
        outputs=[fit_output]
    )

    def update_logistics_timeline(file):
        if not file:
            return {}, []
        
        from cv_parser import extract_text_from_pdf
        with open(file.name, "rb") as f:
            text = extract_text_from_pdf(f)
            
        logistics = extract_logistics(text)
        timeline = extract_experience_timeline(text)
        return logistics, timeline

    file_input.change(
        fn=update_logistics_timeline,
        inputs=[file_input],
        outputs=[logistics_box, timeline_box]
    )

if __name__ == "__main__":
    demo.launch()
