let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let silenceDetector;
let isFirstRecording = true;
let recordingStarted = false;

const recordButton = document.getElementById('recordButton');
const listeningText = document.getElementById('listening');
const respondingText = document.getElementById('responding');

const SILENCE_THRESHOLD = 15;
const SILENCE_DURATION = 2000; // 2 seconds
const MIN_RECORD_DURATION = 2000; // 2 seconds
const MAX_RECORD_DURATION = 30000; // 30 seconds

if (typeof navigator !== 'undefined' && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        console.log("Microphone permission granted");
    })
    .catch(err => {
        console.error("Error accessing the microphone", err);
    });
} else {
    console.error("navigator.mediaDevices or getUserMedia is not supported in this browser.");
}

recordButton.onclick = async () => {
recordButton.disabled = true;

if (isFirstRecording) {
    const defaultAudio = new Audio('/get_default_audio');
    defaultAudio.onplay = () => {
        listeningText.classList.add('hidden');
        respondingText.classList.add('hidden');
    };
    defaultAudio.onended = () => {
        listeningText.classList.remove('hidden');
        respondingText.classList.add('hidden');
        startRecording();
    };
    await defaultAudio.play();
    isFirstRecording = false;
} else {
    listeningText.classList.remove('hidden');
    respondingText.classList.add('hidden');
    startRecording();
}
};

async function startRecording() {
audioChunks = [];
recordingStarted = false;
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
        audioChunks.push(event.data);
    }
};

mediaRecorder.onstop = () => {
    if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        uploadAudio(audioBlob);
    } else {
        console.log("No audio recorded. Not uploading.");
        recordButton.disabled = false;
        listeningText.classList.add('hidden');
        respondingText.classList.add('hidden');
    }
};

mediaRecorder.start(100); // Collect data in 100ms chunks
console.log("Listening for audio...");

audioContext = new AudioContext();
analyser = audioContext.createAnalyser();
const source = audioContext.createMediaStreamSource(stream);
source.connect(analyser);

const bufferLength = analyser.frequencyBinCount;
const dataArray = new Uint8Array(bufferLength);

let silenceStart = null;
let recordingStart = null;

silenceDetector = setInterval(() => {
    analyser.getByteFrequencyData(dataArray);
    const average = dataArray.reduce((a, b) => a + b) / bufferLength;
    
    if (average > SILENCE_THRESHOLD) {
        if (!recordingStarted) {
            recordingStarted = true;
            recordingStart = Date.now();
            console.log("Recording started");
        }
        silenceStart = null;
    } else if (recordingStarted) {
        if (!silenceStart) {
            silenceStart = Date.now();
        } else if (Date.now() - silenceStart > SILENCE_DURATION && 
                    Date.now() - recordingStart > MIN_RECORD_DURATION) {
            stopRecording();
        }
    }

    // Stop recording after MAX_RECORD_DURATION
    if (recordingStarted && Date.now() - recordingStart > MAX_RECORD_DURATION) {
        stopRecording();
    }
}, 100);
}

function stopRecording() {
if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    clearInterval(silenceDetector);
    console.log("Recording stopped");
    listeningText.classList.add('hidden');
    respondingText.classList.remove('hidden');
}
}

function uploadAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recorded_audio.webm');

    listeningText.classList.add('hidden');
    respondingText.classList.remove('hidden');

    fetch('/upload_audio', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log(data.message);
        checkProcessing();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('There was an error processing the audio. Please try again.');
        listeningText.classList.add('hidden');
        respondingText.classList.add('hidden');
        recordButton.disabled = false;
    });
}

function checkProcessing() {
    fetch('/check_processing')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'complete') {
                playProcessedAudio();
            } else {
                // If still processing, check again after a short delay
                setTimeout(checkProcessing, 1000);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('There was an error checking the processing status. Please try again.');
            listeningText.classList.add('hidden');
            respondingText.classList.add('hidden');
            recordButton.disabled = false;
        });
}

function playProcessedAudio() {
    fetch('/get_processed_audio')
        .then(response => {
            if (response.ok) {
                return response.blob();
            } else {
                throw new Error('Processed audio file not found');
            }
        })
        .then(blob => {
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audio.onplay = () => {
                listeningText.classList.add('hidden');
                respondingText.classList.add('hidden');
            };
            audio.play();
            audio.onended = () => {
                resetStatusText();
                checkConversationComplete();
            };
        })
        .catch(error => {
            console.error('Error:', error);
            alert('There was an error playing the processed audio. Please try again.');
            listeningText.classList.add('hidden');
            respondingText.classList.add('hidden');
            recordButton.disabled = false;
        });
}

function checkConversationComplete() {
    fetch('/check_conversation_complete')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'complete') {
                console.log("Conversation complete. Fetching web search results.");
                fetchWebSearchResults();
            } else {
                recordButton.disabled = false;
                startRecording();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            recordButton.disabled = false;
            listeningText.classList.add('hidden');
            respondingText.classList.add('hidden');
        });
}

function fetchWebSearchResults() {
    fetch('/get_web_search_results')
        .then(response => response.json())
        .then(data => {
            if (data.results) {
                displayWebSearchResults(data.results);
            } else {
                console.log("No web search results found.");
            }
            deleteAudioFiles();
        })
        .catch(error => {
            console.error('Error fetching web search results:', error);
            deleteAudioFiles();
        });
}

function displayWebSearchResults(results) {
    // Create a new div to display the results
    const resultsDiv = document.createElement('div');
    resultsDiv.innerHTML = `<h2>Web Search Results</h2><pre>${results}</pre>`;
    document.body.appendChild(resultsDiv);
}

function deleteAudioFiles() {
    fetch('/delete_audio_files')
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
            resetStatusText();
        })
        .catch(error => {
            console.error('Error:', error);
            resetStatusText();
        });
}

function resetStatusText() {
    listeningText.classList.add('hidden');
    respondingText.classList.add('hidden');
}