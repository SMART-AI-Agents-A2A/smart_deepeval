$promptsPath = "D:\Git\Pesquisas\SMART\DeepEval\prompts_30_postman.json"
$outPath = "D:\Git\Pesquisas\SMART\DeepEval\chat_30_respostas.json"
$uri = "http://127.0.0.1:8787/v1/ai/chat"

$cookie = "__Secure-smart.session_token=vU9uL3ZuBm02pZLuFeNB1MbKKrT1xTPM.wQzfyLMy9Sk%2BLj4cHSbqvYHE1Sg%2B%2F9jDw6i6YD5ph7Q%3D"

$headers = @{
    "Cookie" = $cookie
    "Origin" = "http://localhost:5173"
}

$prompts = Get-Content -Path $promptsPath -Raw | ConvertFrom-Json
$results = @()

foreach ($item in $prompts) {
    Write-Host "Rodando pergunta $($item.numero)..."

    $body = @{
        conversationId = $item.conversationId
        messages = @(
            @{
                role = "user"
                content = $item.prompt
            }
        )
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod `
            -Uri $uri `
            -Method POST `
            -ContentType "application/json" `
            -Headers $headers `
            -Body $body `
            -TimeoutSec 60

        $results += [pscustomobject]@{
            numero = $item.numero
            conversationId = $item.conversationId
            prompt = $item.prompt
            status = 200
            route = $response.route
            selectedAgent = $response.selectedAgent
            response = $response.response
            trace = $response.trace
            agentResults = $response.agentResults
            rag = $response.rag
            error = $null
        }
    }
    catch {
        $results += [pscustomobject]@{
            numero = $item.numero
            conversationId = $item.conversationId
            prompt = $item.prompt
            status = 500
            route = $null
            selectedAgent = $null
            response = $null
            trace = $null
            agentResults = $null
            rag = $null
            error = $_.Exception.Message
        }
    }
}

$results | ConvertTo-Json -Depth 20 | Set-Content -Path $outPath -Encoding UTF8
Write-Host "Concluído: $outPath"