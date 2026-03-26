from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from views.chatbotLegalv2 import process_input, process_input_stream, create_new_chat, get_chat_list, load_chat
from views.judgmentPred import extract_text_from_file, predict_verdict, analyze_case_text
from views.docGen import generate_legal_document
import os

app = Flask(__name__)

def _build_chat_list():
    """Helper to build chat list with titles."""
    raw_chat_names = get_chat_list()
    chat_list = []
    for name in raw_chat_names:
        chat_data = load_chat(name)
        first_q = chat_data["past"][0] if chat_data["past"] else "New chat"
        truncated_q = first_q[:30] + '...' if len(first_q) > 30 else first_q
        chat_list.append({
            "name": name,
            "title": truncated_q
        })
    return chat_list

@app.route('/')
def index():
    chat_list = _build_chat_list()
    chat_name = chat_list[0]["name"] if chat_list else create_new_chat()
    chat_data = load_chat(chat_name) if chat_list else {"past": [], "generated": []}
    return render_template('index.html', chat_name=chat_name, chat_list=list(reversed(chat_list)), chat_data=chat_data)

@app.route('/chat_list')
def chat_list_route():
    chat_list = _build_chat_list()
    return jsonify({"chat_list": list(reversed(chat_list))})

@app.route('/chat', methods=['POST'])
def chat():
    """Non-streaming chat endpoint (fallback)."""
    data = request.json
    user_input = data.get('user_input', '')
    chat_name = data.get('chat_name', '')

    if not user_input or not chat_name:
        return jsonify({"error": "Missing input or chat name"}), 400

    try:
        response = process_input(chat_name, user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Failed to process: {str(e)}"}), 500


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """
    SSE streaming chat endpoint.
    Returns Server-Sent Events with TOKEN, THINKING, and DONE events.
    """
    data = request.json
    user_input = data.get('user_input', '')
    chat_name = data.get('chat_name', '')

    if not user_input or not chat_name:
        return jsonify({"error": "Missing input or chat name"}), 400

    def generate():
        try:
            yield from process_input_stream(chat_name, user_input)
        except Exception as e:
            import json
            yield f"data: {json.dumps({'event': 'ERROR', 'content': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@app.route('/new_chat', methods=['POST'])
def new_chat():
    chat_name = create_new_chat()
    return jsonify({"chat_name": chat_name})

@app.route('/load_chat', methods=['POST'])
def load_existing_chat():
    data = request.json
    chat_name = data.get('chat_name')
    if not chat_name:
        return jsonify({"error": "Chat name required"}), 400

    chat_data = load_chat(chat_name)
    return jsonify({"chat_data": chat_data})

@app.route('/predict', methods=['GET', 'POST'])
def predict_judgment():
    if request.method == 'POST':
        if request.content_type and 'multipart/form-data' in request.content_type:
            file = request.files.get('file')
            file_type = request.form.get('file_type')

            if not file or not file_type:
                return jsonify({"error": "File and file type required."}), 400

            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)

            try:
                text = extract_text_from_file(temp_path, file_type)
                result = predict_verdict(text)
                return jsonify({
                    "text": text,
                    "result": result
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
        else:
            data = request.json
            case_text = data.get('case_text', '')
            if not case_text:
                return jsonify({"error": "Case text is required."}), 400
            
            try:
                result = analyze_case_text(case_text)
                return jsonify({
                    "text": case_text,
                    "result": result
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    chat_list = _build_chat_list()
    return render_template('predict.html', chat_list=list(reversed(chat_list)))


@app.route('/generate')
def generate():
    chat_list = _build_chat_list()
    return render_template('generate.html', chat_list=list(reversed(chat_list)))

@app.route('/generate_document', methods=['POST'])
def generate_document():
    data = request.json
    prompt = data.get('doc_prompt', '')
    if not prompt:
        return jsonify({'error': 'Please describe the legal document you need.'}), 400

    try:
        file_path, file_name, preview = generate_legal_document(prompt)
        return jsonify({
            'download_url': f'/download/{file_name}',
            'file_name': file_name,
            'preview': preview
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static/generated_docs', filename, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
