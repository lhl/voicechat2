<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f0f0f0;
        }
        .container {
            text-align: center;
            background-color: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            width: 100%;
        }
        button {
            font-size: 1rem;
            padding: 0.5rem 1rem;
            margin: 0.5rem;
            cursor: pointer;
        }
        #status, #timer, #latency {
            margin-top: 1rem;
            font-weight: bold;
        }
        #logArea {
            width: 100%;
            height: 400px;
            margin-top: 1rem;
            padding: 0.5rem;
            border: 1px solid #ccc;
            overflow-y: auto;
            text-align: left;
            font-family: monospace;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Voice Assistant</h1>
        <div id="timer">00:00:000</div>
        <div id="latency">Response Latency: 0.00ms</div>
        <button id="startButton">Start Session</button>
        <button id="recordButton" disabled>Start Recording</button>
        <div id="status">Ready</div>
        <div id="logArea"></div>
    </div>

    <script>
        const startButton = document.getElementById('startButton');
        const recordButton = document.getElementById('recordButton');
        const status = document.getElementById('status');
        const logArea = document.getElementById('logArea');
        const timerDisplay = document.getElementById('timer');
        const latencyDisplay = document.getElementById('latency');

        let isRecording = false;
        let startTime;
        let timerInterval;
        let peerConnection;
        let dataChannel;
        let audioStream;
        let mediaRecorder;

        let socket;
        let iceCandidatesQueue = [];
        let isWebSocketReady = false;

        function log(message) {
            const timestamp = new Date().toISOString();
            logArea.innerHTML = `${timestamp} - ${message}<br>` + logArea.innerHTML;
        }

        function updateTimer() {
            const elapsed = Date.now() - startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            const milliseconds = elapsed % 1000;
            timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}:${milliseconds.toString().padStart(3, '0')}`;
        }

        function initializeWebSocket() {
            return new Promise((resolve, reject) => {
                socket = new WebSocket('ws://' + window.location.host + '/ws');
                socket.onopen = () => {
                    console.log('WebSocket connected');
                    isWebSocketReady = true;
                    resolve(socket);
                };
                socket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    reject(error);
                };
                socket.onmessage = handleServerMessage;
            });
        }

        function sendToServer(data) {
            if (isWebSocketReady) {
                socket.send(JSON.stringify(data));
            } else {
                console.warn('WebSocket not ready, queuing data');
                iceCandidatesQueue.push(data);
            }
        }

        async function startSession() {
            try {
                await initializeWebSocket();
                
                const configuration = {'iceServers': [{'urls': 'stun:stun.l.google.com:19302'}]};
                peerConnection = new RTCPeerConnection(configuration);
                
                dataChannel = peerConnection.createDataChannel('chat');
                dataChannel.onmessage = handleDataChannelMessage;
                
                peerConnection.onicecandidate = event => {
                    if (event.candidate) {
                        sendToServer({
                            type: 'ice-candidate',
                            candidate: event.candidate
                        });
                    }
                };

                peerConnection.oniceconnectionstatechange = () => {
                    console.log("ICE connection state:", peerConnection.iceConnectionState);
                };
                
                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);
                
                sendToServer({
                    type: 'offer',
                    sdp: peerConnection.localDescription.sdp
                });
                
                // Send any queued ICE candidates
                while (iceCandidatesQueue.length) {
                    const candidate = iceCandidatesQueue.shift();
                    sendToServer(candidate);
                }
                
                startButton.disabled = true;
                recordButton.disabled = false;
                status.textContent = 'Session started';
            } catch (error) {
                console.error('Error starting session:', error);
                status.textContent = 'Error: ' + error.message;
            }
        }

        function handleServerMessage(event) {
            const message = JSON.parse(event.data);
            if (message.type === 'answer') {
                const remoteDesc = new RTCSessionDescription({
                    type: 'answer',
                    sdp: message.sdp
                });
                peerConnection.setRemoteDescription(remoteDesc)
                    .then(() => {
                        console.log("Remote description set successfully");
                    })
                    .catch(error => console.error('Error setting remote description:', error));
            }
        }

        function handleDataChannelMessage(event) {
            const message = JSON.parse(event.data);
            switch (message.type) {
                case 'llm_output':
                    log('LLM: ' + message.content);
                    break;
                case 'tts_complete':
                    log('TTS completed for: ' + message.sentence);
                    const latency = Date.now() - startTime;
                    latencyDisplay.textContent = `Response Latency: ${latency}ms`;
                    break;
                case 'interrupted':
                    log('Processing interrupted');
                    break;
                case 'processing_complete':
                    status.textContent = 'Ready';
                    break;
                case 'error':
                    log('Error: ' + message.message);
                    status.textContent = 'Error';
                    break;
                default:
                    log('Unknown message type: ' + message.type);
            }
        }

        async function startRecording() {
            if (isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                recordButton.textContent = 'Start Recording';

                // After stopping the recording and sending the last chunk of audio data
                dataChannel.send("RECORDING_COMPLETE");

                clearInterval(timerInterval);
            } else {
                try {
                    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioStream.getTracks().forEach(track => peerConnection.addTrack(track, audioStream));
                    
                    const options = { mimeType: 'audio/webm;codecs=opus' };
                    mediaRecorder = new MediaRecorder(audioStream, options);
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            // Convert Blob to ArrayBuffer using a Promise
                            const reader = new FileReader();
                            reader.onload = (e) => {
                                const arrayBuffer = e.target.result;
                                // Send the ArrayBuffer through the data channel
                                dataChannel.send(arrayBuffer);
                            };
                            reader.onerror = (error) => {
                                console.error('Error converting Blob to ArrayBuffer:', error);
                            };
                            reader.readAsArrayBuffer(event.data);
                        }
                    };
                    
                    mediaRecorder.start(1000); // Send data every 1 second
                    isRecording = true;
                    recordButton.textContent = 'Stop Recording';
                    startTime = Date.now();
                    timerInterval = setInterval(updateTimer, 10);
                    status.textContent = 'Recording...';
                } catch (err) {
                    console.error('Error accessing microphone:', err);
                    status.textContent = 'Error: ' + err.message;
                }
            }
        }

        startButton.onclick = startSession;
        recordButton.onclick = startRecording;

        // Initialize when the page loads
        window.onload = () => {
            log('Application initialized');
        };
    </script>
</body>
</html>
