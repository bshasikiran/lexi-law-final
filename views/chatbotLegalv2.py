import json
from views.chat_storage import load_chat, save_chat, create_new_chat, get_chat_list
from views.nvidia_llm import invoke, stream, build_messages

SYSTEM_PROMPT = """\
You are LexiLaw — an expert AI Legal Assistant with comprehensive knowledge of:

- **Indian Law**: Indian Penal Code (IPC) / Bharatiya Nyaya Sanhita (BNS), Code of Criminal Procedure (CrPC), 
  Code of Civil Procedure (CPC), Indian Evidence Act, Indian Constitution, and all major Indian statutes.
- **Legal Procedures**: Filing processes, court procedures, bail applications, FIR processes, 
  appeals, writs, and other procedural aspects.
- **Document Drafting**: Legal agreements, contracts, notices, petitions, and applications.
- **Case Law**: Landmark judgments from the Supreme Court of India and High Courts.
- **General Legal Concepts**: Contract law, property law, family law, corporate law, 
  intellectual property, consumer rights, labor law, and environmental law.

**Your Guidelines:**
1. Provide accurate, clear, and actionable legal information.
2. Always cite relevant sections, acts, and landmark cases when applicable.
3. Use simple language to explain complex legal concepts.
4. If a question is beyond your knowledge, clearly state that and recommend consulting a qualified lawyer.
5. Format responses with markdown for readability — use headings, bullet points, bold text.
6. Provide practical next steps when someone describes a legal situation.
7. Never provide advice that could be construed as incitement to break the law.
8. Clarify that your responses are informational and not a substitute for professional legal counsel.
"""

# Re-export storage functions so app.py doesn't need to change imports
__all__ = ['process_input', 'process_input_stream', 'create_new_chat', 'get_chat_list', 'load_chat']


def _build_conversation_messages(chat_name: str, user_input: str) -> list[dict]:
    """Build the messages array from chat history for the NVIDIA API."""
    current_chat = load_chat(chat_name)
    
    # Build conversation history (last 10 exchanges)
    recent_past = current_chat["past"][-10:]
    recent_generated = current_chat["generated"][-10:]
    
    history = []
    for q, a in zip(recent_past, recent_generated):
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": a})
    
    return build_messages(SYSTEM_PROMPT, history, user_input)


def process_input(chat_name: str, user_input: str) -> str:
    """Process input (non-streaming), update chat, and return the AI's response."""
    messages = _build_conversation_messages(chat_name, user_input)
    
    response = invoke(messages, temperature=0.7, max_tokens=4096)
    
    # Update conversation
    current_chat = load_chat(chat_name)
    current_chat["past"].append(user_input)
    current_chat["generated"].append(response)
    save_chat(chat_name, current_chat)
    
    return response


def process_input_stream(chat_name: str, user_input: str):
    """
    Process input with streaming. Yields SSE-formatted events.
    """
    messages = _build_conversation_messages(chat_name, user_input)
    
    full_response = ""
    
    for event_type, content in stream(messages, temperature=0.7, max_tokens=4096):
        if event_type == "THINKING":
            yield f"data: {json.dumps({'event': 'THINKING', 'content': content})}\n\n"
        elif event_type == "TOKEN":
            full_response += content
            yield f"data: {json.dumps({'event': 'TOKEN', 'content': content})}\n\n"
        elif event_type == "DONE":
            # Save the full response to chat history
            current_chat = load_chat(chat_name)
            current_chat["past"].append(user_input)
            current_chat["generated"].append(full_response)
            save_chat(chat_name, current_chat)
            yield f"data: {json.dumps({'event': 'DONE', 'content': ''})}\n\n"
