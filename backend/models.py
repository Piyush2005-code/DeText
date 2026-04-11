from pydantic import BaseModel


class LanguageRequest(BaseModel):
    detection_algo : str
    prompt : str



