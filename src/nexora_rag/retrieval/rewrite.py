"""
Conversational query rewriting and query routing.

STATUS: DEFERRED until Step S (conversation memory) exists.

Query rewriting needs chat history to rewrite FROM (e.g. turning "and for
G4?" into "What is the per diem for a G4 employee?" requires knowing what
the previous question was). Since Postgres-backed session/message storage
(Step S) doesn't exist yet, this file stays a stub -- filled in once
multi-turn conversations have somewhere to read history from.
"""

# TODO(Step O, after Step S): implement query rewriting + routing here