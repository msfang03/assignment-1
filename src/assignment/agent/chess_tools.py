"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    # TODO(Part 3.3.b): Parse the arguments, call the provided
    # /api/simulate endpoint with fen and optional move, and return its JSON.
    # Catch any errors raised by the tool and return an error message between
    # `<chess_error></chess_error>` for the agent to address. Cover malformed
    # JSON arguments, arguments that are not an object, a missing or
    # non-string fen, a non-string move, a position or move the server rejects,
    # and a transport failure.
    try:
        args = json.loads(arguments)
    except Exception as e:
        return f"<chess_error>Malformed JSON arguments, got {e}</chess_error>"

    if not isinstance(args, dict):
        return (
            f"<chess_error>Tool arguments must decode to a JSON object, "
            f"got {type(args).__name__}</chess_error>"
        )

    url = "/api/simulate"

    fen = args.get("fen")
    move = args.get("move")

    if move is not None and not isinstance(args.get("move"), str):
            return f"<chess_error>Given a non-string move. Move needs to be a string</chess_error>"
    
    if not isinstance(fen, str):
        return f"<chess_error>Missing fen or non-string fen</chess_error>"

    try:
        if move is not None:
            state = _request_state(client, "POST", url, json={"fen": fen, "move": move})
        else:
            state = _request_state(client, "POST", url, json={"fen": fen})
        return json.dumps(state)
    except httpx.TransportError as e:
        return f"<chess_error>Transport failure communicating with chess server: {e}</chess_error>"
    except (ValueError, RuntimeError) as e:
        return f"<chess_error>{e}</chess_error>"
    except Exception as e:
        return f"<chess_error>Unexpected Error: {e}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    # TODO(3.1.b): Parse the arguments and POST {"move": <uci move>} to
    # /api/move. Return its JSON object. Catch any errors raised by the
    # tool and return an error message between `<chess_error></chess_error>`
    # for the agent to address. Cover malformed JSON arguments, arguments
    # that are not an object, a missing or non-string fen, a non-string move,
    # a position or move the server rejects, and a transport failure.
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise TypeError(f"Tool Call arguments must decode to a JSON object, got {type(args).__name__}")
    except Exception as e:
        return f"<chess_error>Malformed JSON arguments, got {e}</chess_error>"

    if "move" not in args:
        return f"<chess_error> Missing argument move</chess_error>"
    move = args.get("move")

    if not isinstance(move, str):
        return f"<chess_error>Given a non-string move. Move needs to be a string</chess_error>"

    if "fen" in args and not isinstance(args.get("fen"), str):
        return f"<chess_error>Missing fen or non-string fen</chess_error>"

    url = "/api/move"

    try:
        state = _request_state(client, "POST", url, json={"move": move})
        return json.dumps(state)
    except httpx.TransportError as e:
        return f"<chess_error>Transport failure communicating with chess server: {e}</chess_error>"
    except (ValueError, RuntimeError) as e:
        return f"<chess_error>{e}</chess_error>"
    except Exception as e:
        return f"<chess_error>Unexpected Error: {e}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    # TODO(3.4): parse the arguments and run the code in the
    # sandbox with the registered tools available by name.
    #
    # `/opt/assignment/sandbox_python.py` is a script on the `env` sandbox
    # that has access to the same tool definitions in this file. Use it to run
    # the code that the model produced as an argument to the run_python tool.
    # The script accepts two positional arguments -- `port` and a base64-encoded
    # string of code (to prevent issues with quoting). Implement this tool
    # call.
    #
    # The script prints one JSON object with `stdout`, `stderr`, and `error`
    # from running the code -- return that string as it is.
    #
    # A non-zero returncode means the sandbox itself failed, not the model's
    # code. Report `exception_info` or `stderr` as a <chess_error>.
    #
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"<chess_error>Malformed JSON arguments: {exc}</chess_error>"

    if not isinstance(args, dict):
        return (
            f"<chess_error>Tool arguments must decode to a JSON object, "
            f"got {type(args).__name__}</chess_error>"
        )

    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return "<chess_error>Missing or non-string 'code' argument</chess_error>"

    b64_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    cmd = f"python /opt/assignment/sandbox_python.py {port} {b64_code}"

    try:
        result = env.execute(cmd)
    except Exception as exc:
        return f"<chess_error>Sandbox execution error: {exc}</chess_error>"

    returncode = getattr(result, "returncode", 0)
    if returncode != 0:
        err_msg = (
            getattr(result, "exception_info", None)
            or getattr(result, "stderr", None)
            or f"Sandbox exited with code {returncode}"
        )
        return f"<chess_error>{str(err_msg).strip()}</chess_error>"

    stdout = getattr(result, "stdout", "")
    return stdout.strip()


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    # TODO(3.5): parse the arguments and return the named skill's content.
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"<chess_error>Malformed JSON arguments: {exc}</chess_error>"

    if not isinstance(args, dict):
        return (
            f"<chess_error>Tool arguments must decode to a JSON object, "
            f"got {type(args).__name__}</chess_error>"
        )

    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return "<chess_error>Missing or non-string 'name' argument</chess_error>"

    if name not in skills:
        available = ", ".join(repr(k) for k in sorted(skills.keys())) or "none"
        return f"<chess_error>Unknown skill '{name}'. Available skills: {available}</chess_error>"

    skill = skills[name]
    if isinstance(skill, dict):
        return skill.get("content", "")
    return str(skill)


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
