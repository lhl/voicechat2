#!/bin/bash

time curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "This is a test message. Can you reply if you have received this?"
      }
    ],
    "temperature": 0.7
  }'
