#!/bin/bash

###
# Run this on your GPU machine if you want to connect it to an Internet accessible jump server for tunneling
###

# Replace these with your actual values
JUMP_SERVER="jump_server"
LOCAL_PORT="8000"  # The port your FastAPI server runs on
REMOTE_PORT="8001"  # An arbitrary port on the jump server

# Create the reverse tunnel
ssh -N -R $REMOTE_PORT:localhost:$LOCAL_PORT $JUMP_SERVER
