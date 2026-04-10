import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass

import jwt
from jwt import InvalidTokenError, PyJWKClient

try:
    from redis import Redis
except Exception:  # pragma: no cover - optional dependency
    Redis = None

logger = logging.getLogger("contexta.auth")


@dataclass
class AuthContext:
    tenant_id: str
    namespace: str
    roles: list[str]
    subject: str
    auth_type: str
    token_kid: str | None = None
    token_jti: str | None = None
    token_exp: int | None = None


class AuthManager:
    """Hybrid auth manager supporting JWT RBAC + legacy key mapping fallback."""

    def __init__(self, *, namespace_signing_key: str, client_namespace_map: dict[str, str]) -> None:
        self.namespace_signing_key = (namespace_signing_key or "contexta-default-signing-key").strip()
        self.client_namespace_map = client_namespace_map

        self.mode = (os.getenv("AUTH_MODE", "hybrid") or "hybrid").strip().lower()
        self.jwt_required = (os.getenv("JWT_REQUIRED", "false").strip().lower() == "true") or self.mode == "jwt"
        self.audit_enabled = os.getenv("AUTH_AUDIT_ENABLED", "true").strip().lower() == "true"

        self.jwt_issuer = (os.getenv("JWT_ISSUER") or "").strip() or None
        self.jwt_audience = (os.getenv("JWT_AUDIENCE") or "").strip() or None
        self.jwt_algorithms = [
            a.strip()
            for a in (os.getenv("JWT_ALGORITHMS", "HS256") or "HS256").split(",")
            if a.strip()
        ]
        self.jwt_namespace_claim = (os.getenv("JWT_NAMESPACE_CLAIM", "namespace") or "namespace").strip()
        self.jwt_tenant_claim = (os.getenv("JWT_TENANT_CLAIM", "tenant_id") or "tenant_id").strip()
        self.jwt_roles_claim = (os.getenv("JWT_ROLES_CLAIM", "roles") or "roles").strip()
        self.deployment_tenant_id = self._sanitize_namespace((os.getenv("DEPLOYMENT_TENANT_ID") or "").strip())
        self.jwt_jwks_url = (os.getenv("JWT_JWKS_URL") or "").strip() or None
        self.jwt_active_kid = (os.getenv("JWT_ACTIVE_KID") or "").strip() or None

        self.jwt_enable_revocation = (os.getenv("JWT_ENABLE_REVOCATION", "true").strip().lower() == "true")
        self.jwt_enable_refresh_service = (os.getenv("JWT_ENABLE_REFRESH_SERVICE", "false").strip().lower() == "true")
        self.access_ttl_seconds = max(60, int(os.getenv("JWT_ACCESS_TTL_SECONDS", "900")))
        self.refresh_ttl_seconds = max(300, int(os.getenv("JWT_REFRESH_TTL_SECONDS", "2592000")))
        self.jwt_refresh_signing_key = (os.getenv("JWT_REFRESH_SIGNING_KEY") or "").strip() or None
        self.jwt_refresh_audience = (os.getenv("JWT_REFRESH_AUDIENCE") or "contexta-refresh").strip()
        self.jwt_refresh_issuer = (os.getenv("JWT_REFRESH_ISSUER") or self.jwt_issuer or "contexta-auth").strip()
        self.revocation_backend = (os.getenv("JWT_REVOCATION_BACKEND", "auto") or "auto").strip().lower()
        self.revocation_prefix = (os.getenv("JWT_REVOCATION_KEY_PREFIX") or "contexta:revoked").strip()

        self._jwks_client = PyJWKClient(self.jwt_jwks_url) if self.jwt_jwks_url else None
        self._revoked_local: dict[str, int] = {}
        self._revoked_redis = None

        redis_url = (os.getenv("REDIS_URL") or "").strip()
        if self.jwt_enable_revocation and redis_url and Redis is not None and self.revocation_backend in {"auto", "redis"}:
            try:
                client = Redis.from_url(redis_url)
                client.ping()
                self._revoked_redis = client
            except Exception:
                self._revoked_redis = None

        raw_signing_keys = (os.getenv("JWT_SIGNING_KEYS_JSON") or "").strip()
        parsed_signing_keys: dict[str, str] = {}
        if raw_signing_keys:
            try:
                loaded = json.loads(raw_signing_keys)
                if isinstance(loaded, dict):
                    parsed_signing_keys = {
                        str(k).strip(): str(v)
                        for k, v in loaded.items()
                        if str(k).strip() and str(v)
                    }
            except json.JSONDecodeError:
                logger.warning("JWT_SIGNING_KEYS_JSON is invalid JSON. Ignoring.")

        self.jwt_signing_keys = parsed_signing_keys
        self.jwt_default_key = (os.getenv("JWT_DEFAULT_SIGNING_KEY") or "").strip() or None
        self._legacy_allowed = self.mode in {"legacy", "hybrid", "auto"}

    def _sanitize_namespace(self, raw: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw).strip()).strip("-_").lower()
        return safe[:64] if safe else ""

    def _audit(self, *, decision: str, reason: str, request_id: str | None, subject: str | None, tenant_id: str | None, namespace: str | None, roles: list[str] | None, auth_type: str) -> None:
        if not self.audit_enabled:
            return
        payload = {
            "event": "auth_decision",
            "decision": decision,
            "reason": reason,
            "request_id": request_id,
            "subject": subject,
            "tenant_id": tenant_id,
            "namespace": namespace,
            "roles": roles or [],
            "auth_type": auth_type,
            "ts": int(time.time()),
        }
        logger.info(json.dumps(payload, ensure_ascii=True))

    def _resolve_jwt_key(self, token: str) -> tuple[str, str | None]:
        if self._jwks_client is not None:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            kid = getattr(signing_key, "key_id", None)
            return signing_key.key, str(kid) if kid else None

        unverified_header = jwt.get_unverified_header(token)
        kid = str(unverified_header.get("kid", "") or "").strip() or None

        if kid and kid in self.jwt_signing_keys:
            return self.jwt_signing_keys[kid], kid

        if self.jwt_active_kid and self.jwt_active_kid in self.jwt_signing_keys:
            return self.jwt_signing_keys[self.jwt_active_kid], self.jwt_active_kid

        if self.jwt_default_key:
            return self.jwt_default_key, kid

        if len(self.jwt_signing_keys) == 1:
            only_kid = next(iter(self.jwt_signing_keys.keys()))
            return self.jwt_signing_keys[only_kid], only_kid

        raise InvalidTokenError("No matching JWT signing key found for token kid")

    def _prune_local_revocation(self) -> None:
        now = int(time.time())
        expired = [jti for jti, exp in self._revoked_local.items() if exp <= now]
        for jti in expired:
            self._revoked_local.pop(jti, None)

    def _revocation_key(self, jti: str) -> str:
        return f"{self.revocation_prefix}:{jti}"

    def revoke_jti(self, jti: str, exp: int | None) -> None:
        if not self.jwt_enable_revocation or not jti:
            return
        now = int(time.time())
        ttl = max(60, int(exp or now + 3600) - now)

        if self._revoked_redis is not None:
            try:
                self._revoked_redis.setex(self._revocation_key(jti), ttl, "1")
                return
            except Exception:
                pass

        self._prune_local_revocation()
        self._revoked_local[jti] = now + ttl

    def _is_jti_revoked(self, jti: str | None) -> bool:
        if not self.jwt_enable_revocation or not jti:
            return False

        if self._revoked_redis is not None:
            try:
                return bool(self._revoked_redis.exists(self._revocation_key(jti)))
            except Exception:
                pass

        self._prune_local_revocation()
        return jti in self._revoked_local

    def _pick_signing_key(self) -> tuple[str, str | None]:
        if self.jwt_active_kid and self.jwt_active_kid in self.jwt_signing_keys:
            return self.jwt_signing_keys[self.jwt_active_kid], self.jwt_active_kid
        if self.jwt_default_key:
            return self.jwt_default_key, self.jwt_active_kid
        if self.jwt_signing_keys:
            first_kid = next(iter(self.jwt_signing_keys.keys()))
            return self.jwt_signing_keys[first_kid], first_kid
        raise RuntimeError("No signing key configured for issuing tokens.")

    def issue_tokens(self, *, subject: str, tenant_id: str, namespace: str, roles: list[str]) -> dict:
        if not self.jwt_enable_refresh_service:
            raise RuntimeError("Refresh token service is disabled.")

        access_key, kid = self._pick_signing_key()
        refresh_key = self.jwt_refresh_signing_key or access_key
        now = int(time.time())
        jti_access = uuid.uuid4().hex
        jti_refresh = uuid.uuid4().hex
        sid = uuid.uuid4().hex

        access_payload = {
            "sub": subject,
            self.jwt_tenant_claim: tenant_id,
            self.jwt_namespace_claim: namespace,
            self.jwt_roles_claim: roles,
            "iat": now,
            "nbf": now,
            "exp": now + self.access_ttl_seconds,
            "iss": self.jwt_issuer or "contexta-auth",
            "aud": self.jwt_audience or "contexta-api",
            "jti": jti_access,
            "sid": sid,
            "typ": "access",
        }
        access_headers = {"typ": "JWT"}
        if kid:
            access_headers["kid"] = kid

        refresh_payload = {
            "sub": subject,
            self.jwt_tenant_claim: tenant_id,
            self.jwt_namespace_claim: namespace,
            self.jwt_roles_claim: roles,
            "iat": now,
            "nbf": now,
            "exp": now + self.refresh_ttl_seconds,
            "iss": self.jwt_refresh_issuer,
            "aud": self.jwt_refresh_audience,
            "jti": jti_refresh,
            "sid": sid,
            "typ": "refresh",
        }

        access_token = jwt.encode(access_payload, access_key, algorithm=self.jwt_algorithms[0], headers=access_headers)
        refresh_token = jwt.encode(refresh_payload, refresh_key, algorithm=self.jwt_algorithms[0])
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_ttl_seconds,
        }

    def refresh_tokens(self, refresh_token: str) -> dict:
        if not self.jwt_enable_refresh_service:
            raise PermissionError("Refresh token service is disabled.")
        if not refresh_token:
            raise PermissionError("Missing refresh token.")

        access_key, _ = self._pick_signing_key()
        refresh_key = self.jwt_refresh_signing_key or access_key
        payload = jwt.decode(
            refresh_token,
            key=refresh_key,
            algorithms=self.jwt_algorithms,
            audience=self.jwt_refresh_audience,
            issuer=self.jwt_refresh_issuer,
            options={
                "require": ["exp", "iat", "nbf", "sub", "jti"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
            },
        )

        if str(payload.get("typ", "")) != "refresh":
            raise PermissionError("Invalid refresh token type.")

        old_jti = str(payload.get("jti", "") or "")
        if self._is_jti_revoked(old_jti):
            raise PermissionError("Refresh token has been revoked.")

        old_exp = int(payload.get("exp", int(time.time()) + 60))
        self.revoke_jti(old_jti, old_exp)

        tenant_id = self._sanitize_namespace(str(payload.get(self.jwt_tenant_claim, "")))
        namespace = self._sanitize_namespace(str(payload.get(self.jwt_namespace_claim, "")))
        roles = self._extract_roles(payload)
        subject = str(payload.get("sub", "")).strip()
        return self.issue_tokens(
            subject=subject,
            tenant_id=tenant_id,
            namespace=namespace,
            roles=roles,
        )

    def _decode_jwt(self, token: str) -> tuple[dict, str | None]:
        key, resolved_kid = self._resolve_jwt_key(token)
        options = {
            "require": ["exp", "iat", "nbf", "sub"],
            "verify_signature": True,
            "verify_exp": True,
            "verify_nbf": True,
            "verify_iat": True,
        }
        payload = jwt.decode(
            token,
            key=key,
            algorithms=self.jwt_algorithms,
            issuer=self.jwt_issuer,
            audience=self.jwt_audience,
            options=options,
        )
        return payload, resolved_kid

    def _extract_roles(self, payload: dict) -> list[str]:
        raw_roles = payload.get(self.jwt_roles_claim, [])
        if isinstance(raw_roles, str):
            tokens = [r.strip().lower() for r in raw_roles.split(",") if r.strip()]
            return list(dict.fromkeys(tokens))
        if isinstance(raw_roles, list):
            tokens = [str(r).strip().lower() for r in raw_roles if str(r).strip()]
            return list(dict.fromkeys(tokens))
        return []

    def _resolve_legacy_namespace(self, request) -> tuple[str, str, list[str], str]:
        client_key = (request.headers.get("X-Client-Key") or "").strip()

        if self.client_namespace_map:
            if not client_key:
                raise PermissionError("Missing X-Client-Key for namespace-bound access.")
            mapped = self.client_namespace_map.get(client_key)
            if not mapped:
                raise PermissionError("Invalid client key.")
            tenant = self._sanitize_namespace(mapped)
            return tenant, mapped, ["legacy"], f"client:{client_key}"

        # When no static map is configured, still honor a caller-provided client key
        # so browsers/proxies resolve to a stable anonymous namespace.
        if client_key:
            digest = hmac.new(
                self.namespace_signing_key.encode("utf-8"),
                f"client:{client_key}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:24]
            resolved = f"anon-{digest}"
            return resolved, resolved, ["anonymous"], f"client:{client_key}"

        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        identity = f"{client_host}|{user_agent}"
        digest = hmac.new(
            self.namespace_signing_key.encode("utf-8"),
            identity.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]
        resolved = f"anon-{digest}"
        return resolved, resolved, ["anonymous"], f"anon:{digest}"

    def _enforce_deployment_tenant(self, tenant_id: str) -> None:
        if not self.deployment_tenant_id:
            return
        if self.deployment_tenant_id != tenant_id:
            raise PermissionError("Tenant is not allowed on this deployment.")

    def authenticate(self, request) -> AuthContext:
        request_id = getattr(getattr(request, "state", object()), "request_id", None)
        auth_header = (request.headers.get("Authorization") or "").strip()

        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if not token:
                self._audit(
                    decision="deny",
                    reason="empty_bearer_token",
                    request_id=request_id,
                    subject=None,
                    tenant_id=None,
                    namespace=None,
                    roles=[],
                    auth_type="jwt",
                )
                raise PermissionError("Authorization bearer token is empty.")

            try:
                payload, kid = self._decode_jwt(token)
            except InvalidTokenError as e:
                self._audit(
                    decision="deny",
                    reason=f"invalid_jwt:{str(e)}",
                    request_id=request_id,
                    subject=None,
                    tenant_id=None,
                    namespace=None,
                    roles=[],
                    auth_type="jwt",
                )
                raise PermissionError("Invalid JWT token.")

            namespace = self._sanitize_namespace(str(payload.get(self.jwt_namespace_claim, "")))
            tenant_id = self._sanitize_namespace(str(payload.get(self.jwt_tenant_claim, namespace)))
            if not namespace:
                self._audit(
                    decision="deny",
                    reason="missing_namespace_claim",
                    request_id=request_id,
                    subject=str(payload.get("sub", "")) or None,
                    tenant_id=tenant_id or None,
                    namespace=None,
                    roles=self._extract_roles(payload),
                    auth_type="jwt",
                )
                raise PermissionError("JWT is missing namespace claim.")
            if not tenant_id:
                raise PermissionError("JWT is missing tenant claim.")

            self._enforce_deployment_tenant(tenant_id)

            roles = self._extract_roles(payload)
            subject = str(payload.get("sub", "")).strip()
            if not subject:
                raise PermissionError("JWT subject is missing.")

            jti = str(payload.get("jti", "") or "") or None
            exp = int(payload.get("exp", 0)) if payload.get("exp") is not None else None
            if self._is_jti_revoked(jti):
                raise PermissionError("Token has been revoked.")

            ctx = AuthContext(
                tenant_id=tenant_id,
                namespace=namespace,
                roles=roles,
                subject=subject,
                auth_type="jwt",
                token_kid=kid,
                token_jti=jti,
                token_exp=exp,
            )
            self._audit(
                decision="allow",
                reason="jwt_ok",
                request_id=request_id,
                subject=ctx.subject,
                tenant_id=ctx.tenant_id,
                namespace=ctx.namespace,
                roles=ctx.roles,
                auth_type="jwt",
            )
            return ctx

        if self.jwt_required:
            self._audit(
                decision="deny",
                reason="jwt_required",
                request_id=request_id,
                subject=None,
                tenant_id=None,
                namespace=None,
                roles=[],
                auth_type="jwt",
            )
            raise PermissionError("JWT authentication is required.")

        if not self._legacy_allowed:
            self._audit(
                decision="deny",
                reason="legacy_auth_disabled",
                request_id=request_id,
                subject=None,
                tenant_id=None,
                namespace=None,
                roles=[],
                auth_type="legacy",
            )
            raise PermissionError("Legacy authentication is disabled.")

        tenant_id, namespace, roles, subject = self._resolve_legacy_namespace(request)
        self._enforce_deployment_tenant(tenant_id)
        ctx = AuthContext(tenant_id=tenant_id, namespace=namespace, roles=roles, subject=subject, auth_type="legacy")
        self._audit(
            decision="allow",
            reason="legacy_ok",
            request_id=request_id,
            subject=ctx.subject,
            tenant_id=ctx.tenant_id,
            namespace=ctx.namespace,
            roles=ctx.roles,
            auth_type="legacy",
        )
        return ctx


def require_roles(context: AuthContext, allowed_roles: list[str]) -> bool:
    """RBAC check. Legacy mode keeps backward compatibility; JWT requires explicit role match."""
    if context.auth_type != "jwt":
        return True
    if not allowed_roles:
        return True
    role_set = {r.strip().lower() for r in context.roles}
    return any(role in role_set for role in allowed_roles)
