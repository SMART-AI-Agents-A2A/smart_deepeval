$base = "http://localhost:8787"

$body = @{
  email = "ramoncbarbosa@unifesspa.edu.br"
  password = "uqqUaj5uV8HqGA3"
} | ConvertTo-Json

$response = Invoke-WebRequest `
  -Uri "$base/v1/auth/sign-in/email" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body `
  -SessionVariable smartSession

"STATUS:"
$response.StatusCode

"`nSET-COOKIE:"
$response.Headers["Set-Cookie"]

"`nCOOKIES CAPTURADOS:"
$cookies = $smartSession.Cookies.GetCookies($base)
$cookies | Format-Table Name, Value, Domain, Path, Secure

"`nCOOKIE PARA USAR NO .env:"
$cookieHeader = ($cookies | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join "; "
"HONO_COOKIE=$cookieHeader"