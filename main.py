import os
import sys
from pydub import AudioSegment
import requests
from azure_speech_to_text import SpeechToTextManager
from AIbot import question_chain
import time
import json
from prplxty2 import *


stt = SpeechToTextManager()

def generate_audio_response(text, output_folder):
    subscription_key = os.getenv("AZURE_TTS_KEY")
    region = os.getenv("AZURE_TTS_REGION")
    endpoint_url = f'https://{region}.tts.speech.microsoft.com/cognitiveservices/v1'
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/ssml+xml',
        'X-Microsoft-OutputFormat': 'audio-16khz-32kbitrate-mono-mp3'
    }
    ssml = f"""
    <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
        <voice name='en-US-AvaMultilingualNeural'>
            {text}
        </voice>
    </speak>
    """
    response = requests.post(endpoint_url, headers=headers, data=ssml)

    if response.status_code == 200:
        output_file = os.path.join(output_folder, 'abc.wav')
        with open(output_file, 'wb') as audio_file:
            audio_file.write(response.content)
        return output_file
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def num2words(num):
    numbers = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven"
    }
    return numbers.get(num, str(num))

def perform_web_search(query):
    # Reduce query to under 30 words
    query_words = query.split()
    if len(query_words) > 30:
        query = ' '.join(query_words[:30])
    
    print("Getting search links...")
    links = get_google_search_links(query)
    print(f"Found {len(links)} links")
    
    # print("Scraping and summarizing text from links...")
    # sources = scrape_text_from_links(links)
    
    # results = "\nResults:\n"
    # for source in sources:
    #     results += f"URL: {source['url']}\n"
    #     results += f"Summary: {source['text']}\n\n"
    
    results = "Reference links:\n"
    for i, link in enumerate(links, 1):
        results += f"[{i}] - {link}\n"
    
    return results

def process_audio(input_file, output_folder):
    print(f"Processing audio file: {input_file}")
    
    chat_history = ""
    context = ""
    question_number = None
    chat_file_path = os.path.join(output_folder, "RecordedChats.txt")
    
    while True:
        # Wait for new input
        while not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
            time.sleep(0.5)
        
        # Convert WebM to WAV
        audio = AudioSegment.from_file(input_file, format="webm")
        wav_file = input_file.replace('.webm', '.wav')
        audio.export(wav_file, format="wav")
        
        question = stt.speechtotext_from_file(wav_file)
        
        with open(chat_file_path, "a") as chat_file:
            chat_file.write(f"User: {question}\n")
            chat_history += f"User: {question}\n"
        
        if any(word in question.lower() for word in ["bye", "thank", "goodbye"]):
            final_response = "Happy to help"
            generate_audio_response(final_response,output_folder)
            
            web_search_results = perform_web_search(context)
            with open(os.path.join(output_folder,"web_search_results.txt"),"w") as f:
                f.write(web_search_results)
            # Signal to end the conversation
            with open(os.path.join(output_folder, "conversation_complete.json"), "w") as f:
                json.dump({"status": "complete"}, f)
            with open(os.path.join(output_folder, "processing_complete.json"), "w") as f:
                json.dump({"status": "complete"}, f)
            
            break
        
        if question_number is None:
            for i in range(1, 12):
                if str(i) in question or num2words(i).lower() in question.lower():
                    question_number = i
                    with open("questions.txt", 'r', encoding='utf-8') as f:
                        context = f.read().split('\n\n')[i-1]
                    print(f"Question number {i} selected.")
                    break
            else:
                print("No specific question number detected.")
        
        context_with_history = chat_history + f"\nContext: {context}\n"
        response_text = question_chain.invoke({"context": context_with_history, "question": question})
        
        with open(chat_file_path, "a") as chat_file:
            chat_file.write(f"Tutor: {response_text}\n")
        
        # Generate and save the audio response
        response_file = generate_audio_response(response_text, output_folder)
        if response_file:
            print(f"Audio response generated: {response_file}")
        else:
            print("Failed to generate audio response")
        
        # Signal that processing is complete
        with open(os.path.join(output_folder, "processing_complete.json"), "w") as f:
            json.dump({"status": "complete"}, f)
        
        print("Waiting for new audio input...")
        # Clear the input file to wait for new input
        open(input_file, 'w').close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_audio_file> <output_folder>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_audio(input_file, output_folder)