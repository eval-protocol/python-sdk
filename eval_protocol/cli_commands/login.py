import getpass
import logging
import os
import time
import webbrowser
from pathlib import Path
from typing import Dict, Optional, Tuple
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlencode, urlparse, parse_qs

import requests

from eval_protocol.auth import (
    AUTH_INI_FILE,
    FIREWORKS_CONFIG_DIR,
    get_fireworks_api_base,
)

logger = logging.getLogger(__name__)


def _write_auth_file_kv(entries: Dict[str, str]) -> Path:
    """Write key=value entries to ~/.fireworks/auth.ini with 600 perms."""
    FIREWORKS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Merge with any existing keys
    existing: Dict[str, str] = {}
    try:
        with open(AUTH_INI_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    existing.update({k: v for k, v in entries.items() if v is not None})
    AUTH_INI_FILE.write_text("\n".join([f"{k}={v}" for k, v in existing.items()]) + "\n")
    try:
        os.chmod(AUTH_INI_FILE, 0o600)
    except Exception:
        pass
    return AUTH_INI_FILE


def _validate_account(api_key: str, account_id: str, api_base: Optional[str]) -> bool:
    """Validate API key against a specific account id using Fireworks REST API.

    Performs GET /v1/accounts/{account_id}. Returns True on HTTP 200, False otherwise.
    """
    base = (api_base or get_fireworks_api_base()).rstrip("/")
    url = f"{base}/v1/accounts/{account_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            logger.info("Successfully validated credentials against Fireworks API.")
            return True
        else:
            logger.warning(
                f"Validation failed (status {resp.status_code}). Response: {resp.text[:200]}"
            )
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error during validation: {e}")
        return False


def _discover_oidc(issuer: str) -> Dict[str, str]:
    try:
        resp = requests.get(issuer.rstrip("/") + "/.well-known/openid-configuration", timeout=10)
        if resp.ok:
            return resp.json()
    except Exception:
        return {}
    return {}


def _oauth_device_flow(issuer: str, client_id: str, scope: str, open_browser: bool) -> Optional[Dict[str, str]]:
    """Perform OAuth2 Device Authorization Grant and return token dict {access_token, refresh_token, expires_in, token_type, scope} or None."""
    meta = _discover_oidc(issuer)
    device_endpoint = meta.get("device_authorization_endpoint") or issuer.rstrip("/") + "/oauth/device/code"
    token_endpoint = meta.get("token_endpoint") or issuer.rstrip("/") + "/oauth/token"

    # 1) Request device code
    data = {"client_id": client_id, "scope": scope}
    resp = requests.post(device_endpoint, data=data, timeout=15)
    if not resp.ok:
        logger.error(f"Device code request failed: {resp.status_code} {resp.text[:200]}")
        return None
    d = resp.json()
    device_code = d.get("device_code")
    verification_uri = d.get("verification_uri_complete") or d.get("verification_uri")
    user_code = d.get("user_code")
    interval = int(d.get("interval", 5))
    expires_in = int(d.get("expires_in", 600))

    if not device_code or not verification_uri:
        logger.error("Invalid device authorization response; missing device_code or verification_uri.")
        return None

    logger.info("To authorize, visit this URL and enter the code if prompted:")
    logger.info(verification_uri)
    if user_code:
        logger.info(f"User code: {user_code}")
    if open_browser:
        try:
            webbrowser.open(verification_uri)
        except Exception:
            pass

    # 2) Poll token endpoint
    start = time.time()
    while True:
        if time.time() - start > expires_in:
            logger.error("Device code expired before authorization completed.")
            return None
        time.sleep(interval)
        payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
            "client_id": client_id,
        }
        t = requests.post(token_endpoint, data=payload, timeout=15)
        if t.status_code == 200:
            return t.json()
        try:
            err = t.json().get("error")
        except Exception:
            err = None
        if err in ("authorization_pending", "slow_down"):
            if err == "slow_down":
                interval += 5
            continue
        elif err == "access_denied":
            logger.error("Access denied during device authorization.")
            return None
        else:
            logger.warning(f"Unexpected token polling response: {t.status_code} {t.text[:200]}")
            continue


def _oauth_browser_flow(issuer: str, client_id: str, scope: str) -> Optional[Dict[str, str]]:
    """Perform OAuth2 Authorization Code flow using a local redirect server."""
    meta = _discover_oidc(issuer)
    auth_endpoint = meta.get("authorization_endpoint") or issuer.rstrip("/") + "/oauth/authorize"
    token_endpoint = meta.get("token_endpoint") or issuer.rstrip("/") + "/oauth/token"

    # Start temporary local server
    state = secrets.token_urlsafe(24)
    code_holder: Dict[str, Optional[str]] = {"code": None, "error": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # type: ignore
            try:
                parsed = urlparse(self.path)
                if parsed.path != "/":
                    self.send_error(404)
                    return
                params = parse_qs(parsed.query)
                got_state = params.get("state", [""])[0]
                if got_state != state:
                    code_holder["error"] = f"state_mismatch"
                elif "error" in params:
                    code_holder["error"] = params.get("error", [""])[0]
                else:
                    code_holder["code"] = params.get("code", [None])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authenticated</h2>You can close this window.</body></html>"
                )
            except Exception:
                pass

        def log_message(self, format, *args):  # type: ignore
            return

    # Bind to an available port
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/"

    # Launch server in thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    # Build auth URL
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
    }
    auth_url = auth_endpoint + ("?" + urlencode(params))

    try:
        webbrowser.open(auth_url)
    except Exception:
        logger.info("Could not open a browser automatically. Please open this URL manually:")
        logger.info(auth_url)

    # Wait for code up to 180 seconds
    deadline = time.time() + 180
    while time.time() < deadline and code_holder["code"] is None and code_holder["error"] is None:
        time.sleep(0.2)

    try:
        httpd.shutdown()
    except Exception:
        pass

    if code_holder["error"]:
        logger.error(f"OAuth error: {code_holder['error']}")
        return None
    if not code_holder["code"]:
        logger.error("Timed out waiting for OAuth authorization.")
        return None

    data = {
        "grant_type": "authorization_code",
        "code": code_holder["code"],
        "redirect_uri": redirect_uri,
        "client_id": client_id,
    }
    t = requests.post(token_endpoint, data=data, timeout=15)
    if t.status_code == 200:
        return t.json()
    logger.error(f"Token exchange failed: {t.status_code} {t.text[:200]}")
    return None


def _get_oauth_args_via_rest(account_id: Optional[str], api_base: Optional[str]) -> Optional[Dict[str, str]]:
    """Try to fetch OAuth issuer/client args from Fireworks public API.

    Tries several likely endpoints; returns a dict with keys issuerUrl, clientId, cognitoDomain if found.
    """
    base = (api_base or get_fireworks_api_base()).rstrip("/")
    account = account_id or ""
    candidates = []
    if account:
        candidates.extend(
            [
                f"{base}/v1/accounts/{account}:getOAuthArguments",
                f"{base}/v1/accounts/{account}/oauth:arguments",
                f"{base}/v1/accounts/{account}/oauth/arguments",
            ]
        )
    candidates.extend([f"{base}/v1/oauth:arguments", f"{base}/v1/oauth/arguments"])  # global

    for url in candidates:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Normalize keys
                return {
                    "issuerUrl": data.get("issuerUrl") or data.get("issuer_url"),
                    "clientId": data.get("clientId") or data.get("client_id"),
                    "cognitoDomain": data.get("cognitoDomain") or data.get("cognito_domain"),
                }
        except Exception:
            continue
    return None


def login_command(args) -> int:
    """Handle `eval-protocol login` to store Fireworks credentials.

    - Accepts --api-key, --account-id, --api-base
    - If --validate and account id provided, calls REST API to verify
    - Writes ~/.fireworks/auth.ini (key=value) with 600 perms
    """
    # 1) API key flow if explicitly provided
    if getattr(args, "api_key", None):
        api_key: Optional[str] = getattr(args, "api_key", None)
        account_id: Optional[str] = getattr(args, "account_id", None)
        api_base: Optional[str] = getattr(args, "api_base", None)
        validate: bool = bool(getattr(args, "validate", False))
        if validate and account_id and api_key:
            ok = _validate_account(api_key, account_id, api_base)
            if not ok:
                logger.error("Credential validation failed. Use --no-validate to write anyway.")
                return 2
        entries = {"api_key": api_key}
        if account_id:
            entries["account_id"] = account_id
        if api_base:
            entries["api_base"] = api_base
        path = _write_auth_file_kv(entries)
        masked = api_key[:4] + "…" if len(api_key) >= 4 else "***"
        logger.info(f"Saved Fireworks credentials to {path}. API key starts with: {masked}.")
        if not account_id:
            logger.info("No --account-id provided. You can add it later by re-running login.")
        if api_base:
            logger.info(f"Using custom API base: {api_base}")
        return 0

    # 2) OAuth is the default flow (even if --oauth not passed)
    if getattr(args, "oauth", True):
        issuer = getattr(args, "issuer", None) or os.environ.get("FIREWORKS_OIDC_ISSUER")
        client_id = getattr(args, "client_id", None) or os.environ.get("FIREWORKS_OAUTH_CLIENT_ID")
        scope = getattr(args, "scope", None) or os.environ.get("FIREWORKS_OAUTH_SCOPE", "openid offline_access email profile")
        api_base: Optional[str] = getattr(args, "api_base", None)
        account_id: Optional[str] = getattr(args, "account_id", None)
        # If issuer/client not provided, try discovery via public API
        if not issuer or not client_id:
            discovered = _get_oauth_args_via_rest(account_id, api_base)
            if discovered:
                issuer = issuer or discovered.get("issuerUrl")
                client_id = client_id or discovered.get("clientId")
                # cognitoDomain unused here but could be logged
        if not issuer or not client_id:
            logger.error(
                "Unable to discover OAuth issuer/client ID. Provide --issuer and --client-id, or set FIREWORKS_OIDC_ISSUER/FIREWORKS_OAUTH_CLIENT_ID, or use --api-key."
            )
            return 1

        # Try browser flow first; fallback to device flow if it fails
        token = _oauth_browser_flow(issuer, client_id, scope)
        if not token:
            token = _oauth_device_flow(issuer, client_id, scope, open_browser=True)
        if not token:
            return 2
        now = int(time.time())
        expires_in = token.get("expires_in")
        expires_at = str(now + int(expires_in)) if expires_in else ""
        entries = {
            "issuer": issuer,
            "client_id": client_id,
            "access_token": token.get("access_token", ""),
            "refresh_token": token.get("refresh_token", ""),
            "token_type": token.get("token_type", "Bearer"),
            "scope": token.get("scope", scope),
            "expires_at": expires_at,
        }
        if api_base:
            entries["api_base"] = api_base
        if account_id:
            entries["account_id"] = account_id
        path = _write_auth_file_kv(entries)
        logger.info(f"Saved OAuth tokens to {path}.")
        # Inform about API key requirement for LLM/model calls
        has_env_key = bool(os.environ.get("FIREWORKS_API_KEY"))
        has_file_key = False
        try:
            with open(path, "r") as f:
                for line in f:
                    if line.strip().startswith("api_key=") and line.strip().split("=", 1)[1].strip():
                        has_file_key = True
                        break
        except Exception:
            pass
        if not (has_env_key or has_file_key):
            logger.warning(
                "No Fireworks API key detected. Model/LLM calls require FIREWORKS_API_KEY. "
                "You can add it by re-running: eval-protocol login --api-key YOUR_KEY"
            )
        if not account_id:
            logger.info("Tip: pass --account-id to store your account for platform API calls.")
        return 0

    # 3) Fallback: prompt for API key if OAuth not selected/failed above
    api_key = getpass.getpass(prompt="Enter Fireworks API key: ")
    if not api_key:
        logger.error("No credentials provided. Aborting login.")
        return 1
    entries = {"api_key": api_key}
    account_id = getattr(args, "account_id", None)
    api_base = getattr(args, "api_base", None)
    if account_id:
        entries["account_id"] = account_id
    if api_base:
        entries["api_base"] = api_base
    path = _write_auth_file_kv(entries)
    masked = api_key[:4] + "…" if len(api_key) >= 4 else "***"
    logger.info(f"Saved Fireworks credentials to {path}. API key starts with: {masked}.")
    return 0
