$setCookie = [string]$response.Headers["Set-Cookie"]
$cookieHeader = ($setCookie -split ";")[0]

"`nCOOKIE PARA USAR NO .env:"
"HONO_COOKIE=$cookieHeader"