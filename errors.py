# https://stackoverflow.com/a/49224771/4294919
class CustomException(Exception):
    default_message = ""
    def __init__(self, *args, **kwargs):
        if not args: args = (self.default_message,)
        super().__init__(*args, **kwargs)        

class ScaleMisMatchError(CustomException):
    default_message = "Scales of Ctxts don't match"

class DepthExhaustionError(Exception):
    default_message = """No more multiplication is possible. Bootstrap needed"""

class ErrorOutOfBoudError(Exception):
    default_message = """Error has grown too large"""

class InvalidParamError(Exception):
    default_message = "Invalid Ciphertext parameter"