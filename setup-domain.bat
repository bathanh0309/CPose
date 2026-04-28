@echo off
setlocal
title Setup lephongphu.cloud on live zone

set "DOMAIN=lephongphu.cloud"
set "WWW=www.lephongphu.cloud"
set "TUNNEL_ID=f09dc95d-e3e5-4102-99e9-02fdf29edc1a"
set "TUNNEL_TARGET=%TUNNEL_ID%.cfargotunnel.com"

echo ===============================================
echo   Setup live DNS zone for %DOMAIN%
echo ===============================================
echo.
echo Use the currently live Cloudflare zone behind:
echo - alberto.ns.cloudflare.com
echo - elsa.ns.cloudflare.com
echo.
echo Add these DNS records in that live zone:
echo.
echo [1] Apex record
echo - Type: CNAME
echo - Name: @
echo - Target: %TUNNEL_TARGET%
echo - Proxy: Proxied
echo.
echo [2] WWW record
echo - Type: CNAME
echo - Name: www
echo - Target: %TUNNEL_TARGET%
echo - Proxy: Proxied
echo.
echo Notes:
echo - Do not use the john/susan Cloudflare zone for this domain.
echo - Keep the active nameservers as alberto/elsa.
echo - Cloudflare will flatten the apex CNAME automatically.
echo.
echo After saving the records, press any key to verify public DNS.
pause >nul

echo.
echo Checking public nameservers...
powershell -NoProfile -Command "Resolve-DnsName -Type NS %DOMAIN% -Server 1.1.1.1 -DnsOnly -NoHostsFile | Format-Table -AutoSize"

echo.
echo Checking apex resolution...
powershell -NoProfile -Command "Resolve-DnsName %DOMAIN% -Server 1.1.1.1 -DnsOnly -NoHostsFile -ErrorAction SilentlyContinue | Format-Table -AutoSize"

echo.
echo Checking www resolution...
powershell -NoProfile -Command "Resolve-DnsName %WWW% -Server 1.1.1.1 -DnsOnly -NoHostsFile -ErrorAction SilentlyContinue | Format-Table -AutoSize"

echo.
echo If apex/www still do not resolve, the records have not propagated yet
echo or were added in the wrong Cloudflare account.
pause
exit /b 0
