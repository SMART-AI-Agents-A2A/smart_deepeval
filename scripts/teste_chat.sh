Invoke-RestMethod `
  -Uri "http://localhost:8787/v1/ai/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{
    Origin = "http://localhost:5173"
    Cookie = $cookieHeader
  } `
  -Body '{
    "conversationId": "teste-auth-local",
    "messages": [
      {
        "role": "user",
        "content": "Olá, teste de autenticação."
      }
    ]
  }'