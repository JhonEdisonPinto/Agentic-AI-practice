"""Cliente GraphDB basado en RDFLib.

Este modulo expone operaciones de:
- Carga de ontologia TTL al repositorio GraphDB.
- Ejecucion de consultas SPARQL SELECT.
- Ejecucion de actualizaciones SPARQL UPDATE.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urlparse

from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore, SPARQLUpdateStore


class GraphDBClient:
    """Conexion a GraphDB usando RDFLib SPARQLStore y SPARQLUpdateStore."""

    def __init__(self, repository_id: str | None = None) -> None:
        self.base_url = os.getenv("GRAPHDB_BASE_URL", "http://localhost:7200").rstrip("/")
        self.repository_id = repository_id or os.getenv("GRAPHDB_REPOSITORY", "ontologia-laboral")
        self.username = os.getenv("GRAPHDB_USERNAME")
        self.password = os.getenv("GRAPHDB_PASSWORD")
        self.login_url = os.getenv("GRAPHDB_LOGIN_URL")
        self.auth_mode = os.getenv("GRAPHDB_AUTH_MODE", "auto").lower()
        self.enabled = os.getenv("GRAPHDB_ENABLED", "false").lower() in {"1", "true", "yes"}
        self._token: str | None = os.getenv("GRAPHDB_TOKEN")

        if self.auth_mode == "auto":
            if self.login_url and self.username and self.password:
                self.auth_mode = "token"
            elif self.username and self.password:
                self.auth_mode = "basic"
            else:
                self.auth_mode = "none"

        # En sandbox Graphwise normalmente login y SPARQL comparten host.
        # Si auth token esta activo y base_url apunta a localhost, inferimos host desde login_url.
        if self.auth_mode == "token" and self.login_url and "localhost" in self.base_url:
            parsed = urlparse(self.login_url)
            if parsed.scheme and parsed.netloc:
                self.base_url = f"{parsed.scheme}://{parsed.netloc}"

    @property
    def query_endpoint(self) -> str:
        return f"{self.base_url}/repositories/{self.repository_id}"

    @property
    def update_endpoint(self) -> str:
        return f"{self.base_url}/repositories/{self.repository_id}/statements"

    def _set_credentials(self, store: SPARQLStore | SPARQLUpdateStore) -> None:
        if self.username and self.password and hasattr(store, "setCredentials"):
            store.setCredentials(self.username, self.password)

    def _auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}

        if self.auth_mode == "token":
            token = self.obtain_token()
            if token:
                # Graphwise puede retornar encabezado completo "GDB ...".
                # Si el token ya trae esquema, se usa tal cual.
                if token.startswith("GDB ") or token.startswith("Bearer "):
                    headers["Authorization"] = token
                else:
                    headers["Authorization"] = f"Bearer {token}"
        elif self.auth_mode == "basic" and self.username and self.password:
            import base64

            credentials = f"{self.username}:{self.password}".encode("utf-8")
            headers["Authorization"] = f"Basic {base64.b64encode(credentials).decode('ascii')}"

        return headers

    def obtain_token(self, force_refresh: bool = False) -> str | None:
        """Obtiene token GDB via /rest/login para Graphwise sandbox."""
        if self.auth_mode != "token":
            return None

        if self._token and not force_refresh:
            return self._token

        if not self.login_url:
            raise RuntimeError("GRAPHDB_LOGIN_URL no configurado para auth token")
        if not self.username or not self.password:
            raise RuntimeError("GRAPHDB_USERNAME y GRAPHDB_PASSWORD son requeridos para auth token")

        payload = json.dumps({
            "username": self.username,
            "password": self.password,
        }).encode("utf-8")

        req = request.Request(
            self.login_url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        with request.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8")
            header_auth = response.headers.get("Authorization")

        token: str | None = None
        try:
            data = json.loads(body) if body else {}
            token = data.get("token") or data.get("gdbToken") or data.get("access_token")
        except json.JSONDecodeError:
            token = body.strip() if body.strip() else None

        if not token and header_auth:
            token = header_auth.strip()

        if not token:
            raise RuntimeError("No se pudo obtener token desde /rest/login")

        self._token = token
        return token

    def ping(self) -> bool:
        """Valida conectividad basica contra GraphDB."""
        try:
            if self.auth_mode == "token":
                # Validacion real del endpoint SPARQL en sandbox/token mode.
                probe = "ASK { ?s ?p ?o }"
                req = request.Request(
                    self.query_endpoint,
                    data=probe.encode("utf-8"),
                    method="POST",
                    headers={
                        "Content-Type": "application/sparql-query",
                        "Accept": "application/sparql-results+json",
                        **self._auth_headers(),
                    },
                )
                with request.urlopen(req, timeout=10) as response:
                    return response.status in (200, 204)

            req = request.Request(self.query_endpoint, method="GET", headers=self._auth_headers())
            with request.urlopen(req, timeout=5):
                return True
        except HTTPError as exc:
            # Algunos endpoints responden 400 por falta de parametro query al usar GET.
            # Eso indica endpoint alcanzable (no es error de conectividad).
            if exc.code == 400:
                return True
            return False
        except Exception:
            return False

    def upload_ttl_file(self, ttl_path: str | Path) -> None:
        """Sube un archivo TTL al repositorio via endpoint statements."""
        path = Path(ttl_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo TTL: {path}")

        data = path.read_bytes()
        req = request.Request(
            self.update_endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "text/turtle",
                **self._auth_headers(),
            },
        )

        with request.urlopen(req, timeout=60) as response:
            if response.status not in (200, 201, 204):
                raise RuntimeError(f"Error subiendo ontologia. HTTP {response.status}")

    def select(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Ejecuta un SELECT sobre GraphDB y retorna lista de diccionarios."""
        if self.auth_mode == "token":
            return self._select_http(sparql_query)

        store = SPARQLStore(self.query_endpoint)
        self._set_credentials(store)
        graph = Graph(store=store)

        results = graph.query(sparql_query)
        out: List[Dict[str, Any]] = []
        for row in results:
            item: Dict[str, Any] = {}
            for var in results.vars:
                value = getattr(row, str(var), None)
                item[str(var)] = str(value) if value is not None else None
            out.append(item)
        return out

    def _select_http(self, sparql_query: str) -> List[Dict[str, Any]]:
        """SELECT por HTTP directo (util cuando se usa Bearer token)."""
        req = request.Request(
            self.query_endpoint,
            data=sparql_query.encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/sparql-results+json",
                **self._auth_headers(),
            },
        )

        try:
            with request.urlopen(req, timeout=60) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"SELECT HTTP {exc.code}: {details[:300]}") from exc

        data = json.loads(payload)
        vars_ = data.get("head", {}).get("vars", [])
        bindings = data.get("results", {}).get("bindings", [])

        out: List[Dict[str, Any]] = []
        for row in bindings:
            item: Dict[str, Any] = {}
            for var in vars_:
                item[var] = row.get(var, {}).get("value")
            out.append(item)
        return out

    def update(self, sparql_update: str) -> None:
        """Ejecuta UPDATE SPARQL sobre GraphDB usando RDFLib."""
        if self.auth_mode == "token":
            self._update_http(sparql_update)
            return

        store = SPARQLUpdateStore(self.query_endpoint, self.update_endpoint)
        self._set_credentials(store)
        graph = Graph(store=store)
        graph.update(sparql_update)

    def _update_http(self, sparql_update: str) -> None:
        """UPDATE por HTTP directo para entornos con Bearer token."""
        req = request.Request(
            self.update_endpoint,
            data=sparql_update.encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/sparql-update",
                "Accept": "application/json",
                **self._auth_headers(),
            },
        )

        try:
            with request.urlopen(req, timeout=60) as response:
                if response.status not in (200, 201, 204):
                    raise RuntimeError(f"UPDATE fallo HTTP {response.status}")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"UPDATE HTTP {exc.code}: {details[:300]}") from exc
