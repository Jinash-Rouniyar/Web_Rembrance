from flask import Flask, render_template, request, send_file, jsonify
import os
import subprocess
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sat-tutor')
def sat_tutor():
    return render_template('sat-tutor.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if audio_file:
        filename = 'recorded_audio.webm'
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(file_path)
        
        # Check if main.py is already running
        if not hasattr(app, 'main_process') or app.main_process.poll() is not None:
            # Start main.py if it's not running
            app.main_process = subprocess.Popen(['python', 'main.py', file_path, UPLOAD_FOLDER])
        
        return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/check_processing')
def check_processing():
    file_path = os.path.join(UPLOAD_FOLDER, "processing_complete.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'status': 'complete'}), 200
    else:
        return jsonify({'status': 'processing'}), 202

@app.route('/get_web_search_results')
def get_web_search_results():
    file_path = os.path.join(UPLOAD_FOLDER, "web_search_results.txt")
    print(f"Checking for file: {file_path}")
    if os.path.exists(file_path):
        print("File found, reading contents...")
        with open(file_path, 'r') as f:
            results = f.read()
        return jsonify({'results': results}), 200
    else:
        print("File not found!")
        return jsonify({'error': 'Web search results not found'}), 404
    
@app.route('/get_processed_audio')
def get_processed_audio():
    filename = 'abc.wav'
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='audio/wav')
    else:
        return jsonify({'error': 'Processed audio file not found'}), 404
    
@app.route('/check_conversation_complete')
def check_conversation_complete():
    file_path = os.path.join(UPLOAD_FOLDER, "conversation_complete.json")
    if os.path.exists(file_path):
        os.remove(file_path)  
        return jsonify({'status': 'complete'}), 200
    else:
        return jsonify({'status': 'incomplete'}), 202
    
@app.route('/get_default_audio')
def get_default_audio():
    filename = 'default.wav'
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype='audio/wav')

@app.route('/delete_audio_files')
def delete_audio_files():
    files_to_delete = ['abc.wav', 'recorded_audio.webm', 'recorded_audio.wav', 'RecordedChats.txt', 'processing_complete.json', 'conversation_complete.json',"web_search_results.txt"]
    for filename in files_to_delete:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Could not delete {file_path}. It may be in use.")
    return jsonify({'message': 'Files deleted successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
