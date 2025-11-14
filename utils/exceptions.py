"""
Custom exceptions for YouTube Thumbnail Generator
"""


class InsufficientCharactersError(Exception):
    """
    Raised when the system cannot find enough different people in the video.

    This exception signals that we should try extracting more frames
    and retrying the character selection.
    """

    def __init__(self, found: int, required: int, message: str = None):
        self.found = found
        self.required = required
        self.message = message or f"Found only {found}/{required} different people"
        super().__init__(self.message)
