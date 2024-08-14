import time
import azure.cognitiveservices.speech as speechsdk
import keyboard
import os

class SpeechToTextManager:
    azure_speechconfig = None
    azure_audioconfig = None
    azure_speechrecognizer = None

    def __init__(self):
        # Creates an instance of a speech config with specified subscription key and service region.
        # Replace with your own subscription key and service region (e.g., "westus").
        try:
            self.azure_speechconfig = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_TTS_KEY'), region=os.getenv('AZURE_TTS_REGION'))
        except TypeError:
            exit("Ooops! You forgot to set AZURE_TTS_KEY or AZURE_TTS_REGION in your environment!")
        
        self.azure_speechconfig.speech_recognition_language="en-US"
        self.stop_listening_flag = False
    
    def stop_listening(self):
        self.stop_listening_flag = True
        
    def speechtotext_from_mic(self):
        
        self.azure_audioconfig = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.azure_speechrecognizer = speechsdk.SpeechRecognizer(speech_config=self.azure_speechconfig, audio_config=self.azure_audioconfig)

        print("Speak into your microphone.")
        speech_recognition_result = self.azure_speechrecognizer.recognize_once_async().get()
        text_result = speech_recognition_result.text

        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(speech_recognition_result.text))
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

        print(f"We got the following text: {text_result}")
        return text_result

    def speechtotext_from_file(self, filename):

        self.azure_audioconfig = speechsdk.AudioConfig(filename=filename)
        self.azure_speechrecognizer = speechsdk.SpeechRecognizer(speech_config=self.azure_speechconfig, audio_config=self.azure_audioconfig)

        print("Listening to the file \n")
        speech_recognition_result = self.azure_speechrecognizer.recognize_once_async().get()

        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: \n {}".format(speech_recognition_result.text))
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

        return speech_recognition_result.text

    def speechtotext_from_file_continuous(self, filename):
        self.azure_audioconfig = speechsdk.audio.AudioConfig(filename=filename)
        self.azure_speechrecognizer = speechsdk.SpeechRecognizer(speech_config=self.azure_speechconfig, audio_config=self.azure_audioconfig)

        done = False
        def stop_cb(evt):
            print('CLOSING on {}'.format(evt))
            nonlocal done
            done = True

        # These are optional event callbacks that just print out when an event happens.
        # Recognized is useful as an update when a full chunk of speech has finished processing
        #self.azure_speechrecognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
        self.azure_speechrecognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
        self.azure_speechrecognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        self.azure_speechrecognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
        self.azure_speechrecognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

        # These functions will stop the program by flipping the "done" boolean when the session is either stopped or canceled
        self.azure_speechrecognizer.session_stopped.connect(stop_cb)
        self.azure_speechrecognizer.canceled.connect(stop_cb)

        # This is where we compile the results we receive from the ongoing "Recognized" events
        all_results = []
        def handle_final_result(evt):
            all_results.append(evt.result.text)
        self.azure_speechrecognizer.recognized.connect(handle_final_result)

        # Start processing the file
        print("Now processing the audio file...")
        self.azure_speechrecognizer.start_continuous_recognition()
        
        # We wait until stop_cb() has been called above, because session either stopped or canceled
        while not done:
            time.sleep(.5)

        # Now that we're done, tell the recognizer to end session
        # NOTE: THIS NEEDS TO BE OUTSIDE OF THE stop_cb FUNCTION. If it's inside that function the program just freezes. Not sure why.
        self.azure_speechrecognizer.stop_continuous_recognition()

        final_result = " ".join(all_results).strip()
        print(f"\n\nHeres the result we got from contiuous file read!\n\n{final_result}\n\n")
        return final_result
    
    #Jinash

    def speechtotext_from_mic_continuous(self):
        self.azure_speechrecognizer = speechsdk.SpeechRecognizer(speech_config=self.azure_speechconfig)
        done = False
        final = ""
        # Optional callback to print out whenever a chunk of speech is finished being recognized.
        
        def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            nonlocal final
            print('RECOGNIZED: {}'.format(evt.result.text))
            final = evt.result.text

        self.azure_speechrecognizer.recognized.connect(recognized_cb)

        # We register this to fire if we get a session_stopped or cancelled event.
        def stop_cb(evt: speechsdk.SessionEventArgs):
            print('CLOSING speech recognition on {}'.format(evt))
            nonlocal done
            done = True

        # Connect callbacks to the events fired by the speech recognizer
        self.azure_speechrecognizer.session_stopped.connect(stop_cb)
        self.azure_speechrecognizer.canceled.connect(stop_cb)

        # This is where we compile the results we receive from the ongoing "Recognized" events
        

        # Perform recognition
        result_future = self.azure_speechrecognizer.start_continuous_recognition_async()
        result_future.get()  # wait for voidfuture, so we know engine initialization is done.
        print('Continuous Speech Recognition is now running, say something.')

        while not done:
            # Method to stop recognition when the stop key is pressed
            if self.stop_listening_flag:
                print("\nEnding azure speech recognition\n")
                self.azure_speechrecognizer.stop_continuous_recognition_async()
                break
        
        return final