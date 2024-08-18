import os
from typing import Tuple, Optional
import sys
from pydub import AudioSegment
import requests
from azure_speech_to_text import SpeechToTextManager
import time
import json
from prplxty2 import *
from categorization_chains import (
    gpt_question_chain,
    claude_question_chain,
    groq_question_chain
)
from subcategorization_chains import (
    craft_categorize_chain,
    information_categorize_chain,
    expression_categorize_chain,
)

from task_chains import (
    standard_english_chain,
    vocab_chain,
    purpose_chain,
    connection_chain,
    main_idea_chain,
    detail_chain,
    textual_chain,
    quantitative_chain,
    inference_chain,
    transition_chain,
    synthesis_chain  
)

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

def categorize_question(question: str) -> Tuple[str, Optional[str]]:
    try:
        question_category = gpt_question_chain.invoke({"question": question}).lower()
        print(f"Question Category: {question_category}")

        if "standard" in question_category:
            return question_category, None

        category_chains = {
            "craft and structure": craft_categorize_chain,
            "information and ideas": information_categorize_chain,
            "expression of ideas": expression_categorize_chain,
            "standard english conventions": standard_english_chain
        }
        if question_category == "standard english conventions":
            return question_category, None
        else:
            sub_category_chain = category_chains.get(question_category)
            if sub_category_chain:
                sub_category = sub_category_chain.invoke({"question": question}).lower()
                print(f"Question Sub-Category: {sub_category}")
                return question_category, sub_category
        
        return question_category, None
    except Exception as e:
        print(f"Error in categorizing question: {e}")
        return "Unknown", None

def process_question(category: str, sub_category: Optional[str], chat_history: str, student_input: str, context: str, question: str, options: str, answer_exp: str) -> str:
    task_chains = {
        "vocabulary": vocab_chain,
        "purpose": purpose_chain,
        "connection": connection_chain,
        "main ideas": main_idea_chain,
        "detail": detail_chain,
        "textual evidence": textual_chain,
        "quantitative evidence": quantitative_chain,
        "inference": inference_chain,
        "synthesis": synthesis_chain,
        "transition": transition_chain,
        "standard english conventions": standard_english_chain
    }

    try:
        if category == "standard english conventions":
            return standard_english_chain.invoke({
                "context": context,
                "question": question,
                "options": options,
                "answer_exp": answer_exp,
                "chat_history": chat_history,
                "student_input": student_input
            })
        
        if sub_category:
            for task, chain in task_chains.items():
                if task in sub_category or task==sub_category or sub_category in task:
                    return chain.invoke({
                        "context": context,
                        "question": question,
                        "options": options,
                        "answer_exp": answer_exp,
                        "chat_history": chat_history,
                        "student_input": student_input
                    })
        
        return "No matching task chain found."
    except Exception as e:
        print(f"Error in processing question: {e}")
        return "An error occurred while processing the question."

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
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty"
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
    question_number = None
    chat_file_path = os.path.join(output_folder, "RecordedChats.txt")
    category = None
    sub_category = None
    context = None
    
    with open(chat_file_path, "w") as chat_file:
        while True:
            # Wait for new input
            while not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
                time.sleep(0.5)
            
            # Convert WebM to WAV
            audio = AudioSegment.from_file(input_file, format="webm")
            wav_file = input_file.replace('.webm', '.wav')
            audio.export(wav_file, format="wav")
            
            user_query = stt.speechtotext_from_file(wav_file)
            if any(word in user_query.lower() for word in ["bye", "thank", "goodbye","quit"]):
                final_response = "Happy to help"
                chat_file.write(f"User: {user_query}\nTutor: {final_response}\n")
                generate_audio_response(final_response,output_folder)
                
                if context:
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
                for i in range(1, 21):
                    if str(i) in user_query or num2words(i).lower() in user_query.lower():
                        question_number = i
                        with open("questions.txt", 'r', encoding='utf-8') as f:
                            context = '''
                            In 2014, accusations were made that a global
                            mobile communications carrier had clipped
                            their clients for millions of dollars. The
                            company was adding one-time and recurring
                            service fees to their monthly bills without the
                            clients' knowledge or consent. An investigation
                            by the U.S. Federal Trade Commission
                            resulted in substantial refunds to over 40 percent of
                            their clients.
                            '''
                            question = '''
                            As used in the text, what does the word “clipped”
                            most nearly mean?
                            '''
                            options = '''
                            A) Cut
                            B) Overcharged
                            C) Busted
                            D) Curtailed
                            '''
                            answer_exp = '''
                            For this Words in Context question,
                            use the context of the passage to determine the
                            meaning of 'clipped' Consider the context of the word:
                            clients were clipped for millions of dollars, and then
                            many received refunds. Predict that the company had
                            scammed or overcharged; this matches choice (B) and is
                            correct.
                            Choices(A)and (C) are both incorrect because neither
                            'cut' nor 'busted' make sense in this context. Eliminate
                            (D);'curtail' means to make less, which is the opposite
                            of what occurred.
                        '''
                            # context = f.read().split('\n\n')[i-1] #identify every infomration like question, options,answers and everything
                        print(f"Question number {i} selected.")
                        break
            if question_number is None:
                print("Can you specify the question number?")
                new_response = "Can you specify the question number?"
                response_file = generate_audio_response(new_response, output_folder)
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
                continue
            
            #Once the question number is found, identify and categorize the question
            if category is None: #only categorize the question once
                category, sub_category = categorize_question(user_query)
            
            response_text = process_question(category, sub_category, chat_history, user_query, context, question, options, answer_exp)
            
            chat_file.write(f"User: {user_query}\n")
            chat_file.write(f"Tutor: {response_text}\n")
                    
            chat_history += f"User: {user_query}\nTutor: {response_text}\n"
            
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