"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import json

class RateLimitError(Exception):
    """Exception raised when the rate limit is exceeded."""

    def __init__(self, message='Rate limit exceeded. Please try again later.'):
        self.message = message
        super().__init__(self.message)


class RefusalError(Exception):
    """Exception raised when the LLM refuses to generate a response."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class EmptyResponseError(Exception):
    """Exception raised when the LLM returns an empty response."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

# public recovery information shared by response adapters, Graphiti and ingest owners
class LLMResponseError(Exception):
    """An unresolved response and the next recovery action its caller can safely take."""

    def __init__(self, message: str, *, recovery_action: str = "correct", diagnostics: dict | None = None):

        super().__init__(message)
        self.recovery_action = recovery_action
        try:
            self.diagnostics = json.loads(json.dumps(diagnostics or {}, default = str, allow_nan = False))
            if not isinstance(self.diagnostics, dict):
                raise ValueError('Response diagnostics must be an object')
        except Exception as error:
            self.diagnostics = {
                'diagnostic_error': f'{type(error).__name__}: {error}',
                'diagnostic_excerpt': repr(diagnostics)[:60_000],
            } # diagnostics
