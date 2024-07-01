#!/bin/bash

###
# This script is used if you want to connect to your home network GPU via a jump server
###

# Replace these with your actual values
JUMP_SERVER="jump_server"
REMOTE_PORT="8001"  # The same port used in the reverse tunnel
LOCAL_PORT="8000"  # The port you want to use locally

# Create the tunnel through the jump server
ssh -N -L $LOCAL_PORT:localhost:$REMOTE_PORT $JUMP_SERVER

