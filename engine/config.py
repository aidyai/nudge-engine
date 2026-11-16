"""Settings. Read from the environment, or from a .env file next to where you run things."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    api_key: str = "change-me"  # the website sends this. The API refuses to run with the placeholder.

    # the models we run on Modal (deploy/llm.py, deploy/tts.py) and Deepgram
    models_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "phonellm-mini-alpha-2"
    tts_url: str = ""
    # Modal proxy auth token for the brain (only needed when it is deployed with LLM_SNAPSHOT=1, see deploy/llm.py)
    llm_proxy_key: str = ""  # looks like wk-...
    llm_proxy_secret: str = ""  # looks like ws-...
    deepgram_api_key: str = ""
    hf_token: str = ""  # only used by scripts/check_services.py

    # phone calls (Twilio). Only numbers listed in allowed_numbers can ever be called.
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    allowed_numbers: str = ""  # comma separated, with country code: +2348012345678,+2348098765432

    @property
    def llm_extra_headers(self) -> dict[str, str]:
        """Headers that get past Modal's proxy to the brain. Empty when the brain is public (the default)."""
        if self.llm_proxy_key and self.llm_proxy_secret:
            return {"Modal-Key": self.llm_proxy_key, "Modal-Secret": self.llm_proxy_secret}
        return {}

    @property
    def allowed_list(self) -> list[str]:
        return [n.strip() for n in self.allowed_numbers.split(",") if n.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
