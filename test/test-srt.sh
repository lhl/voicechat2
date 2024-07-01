time curl 127.0.0.1:8001/inference -H "Content-Type: multipart/form-data" -F file="@test_output.opus" -F temperature="0.0" -F temperature_inc="0.2" -F response_format="json"
