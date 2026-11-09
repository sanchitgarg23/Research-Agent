"""LLM factory for instantiating model providers."""

import logging
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import config

logger = logging.getLogger(__name__)


def get_llm(
    temperature: float = 0.7, 
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None
) -> BaseChatModel:
    """Get LLM instance based on configuration.
    
    Args:
        temperature: Temperature for the LLM
        model_override: Optional model name to override config.model_name
        provider_override: Optional provider to override config.model_provider
        
    Returns:
        LLM instance (ChatOllama, ChatGoogleGenerativeAI, or ChatOpenAI)
    """
    model_name = model_override or config.model_name
    provider = provider_override or config.model_provider
    
    if provider == "ollama":
        logger.info(f"Using Ollama model: {model_name}")
        return ChatOllama(
            model=model_name,
            base_url=config.ollama_base_url,
            temperature=temperature,
            num_ctx=8192,
            num_predict=2048,
        )
    elif provider == "openai":
        logger.info(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            base_url=f"{config.openai_base_url}/v1",
            api_key=config.openai_api_key,
            temperature=temperature
        )
    elif provider == "llamacpp":
        logger.info(f"Using llama.cpp server model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            base_url=f"{config.llamacpp_base_url}/v1",
            api_key="not-needed",
            temperature=temperature
        )
    else:  # gemini
        logger.info(f"Using Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.google_api_key,
            temperature=temperature
        )
