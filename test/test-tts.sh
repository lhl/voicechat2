time curl -X POST "http://localhost:8003/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a test message. Can you reply if you have received this?"}' \
     --output test_output.opus
opusinfo test_output.opus
