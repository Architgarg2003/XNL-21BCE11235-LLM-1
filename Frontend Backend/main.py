from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import logging
import asyncio

# Setup Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# API Keys (Ensure they're set in the environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set these in env
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not OPENAI_API_KEY or not TOGETHER_API_KEY:
    logging.error("API keys are missing! Set them as environment variables.")
    raise RuntimeError("Missing API keys!")

# LLM Endpoints
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

# Models
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Request Model
class PromptRequest(BaseModel):
    prompt: str

async def query_openai(prompt: str):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": PRIMARY_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    async with httpx.AsyncClient(timeout=0.01) as client:  # 10ms timeout
        try:
            response = await client.post(OPENAI_URL, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logging.warning(f"OpenAI API Error: {e.response.status_code} - {e.response.text}")
        except httpx.TimeoutException:
            logging.warning("OpenAI request timed out (>10ms)")
        except Exception as e:
            logging.warning(f"OpenAI request failed: {str(e)}")
        
        return None  # Failover to Together API

async def query_together_ai(prompt: str):
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": FALLBACK_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.post(TOGETHER_URL, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logging.error(f"Together API Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logging.error(f"Together request failed: {str(e)}")
        
        raise HTTPException(status_code=500, detail="Both models failed.")

@app.post("/generate")
async def generate(request: PromptRequest):
    # Try OpenAI first with 10ms timeout
    response = await query_openai(request.prompt)
    
    # Fallback to Together AI if OpenAI fails or times out
    if response is None:
        response = await query_together_ai(request.prompt)
    
    return {"response": response}