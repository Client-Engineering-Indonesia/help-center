import warnings
warnings.filterwarnings('ignore')

import io
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
import pandas as pd

from app.helpers.helper import *
from app.helpers.wxwd_function import *
from app.helpers.cos_public_image import *

app = FastAPI(
    title='Sample-app FastAPI and Docker',
    version = '1.0.0',
)


@app.get("/")
async def root():
    return {"message": "Hello World with BNI"}

@app.get("/ping")
async def ping():
    return "Hello, I am alive..."

# @app.post("/process_dict")
# async def process_dict(input_dict: dict):
#     # You can perform any processing on the input dictionary here
#     # For example, let's just return the received dictionary as is
#     return input_dict

@app.post("/process_dict_req")
async def process_dict_req(request: Request):
    try:
        passage = await request.json()
        answer = passage['answer']
        modified_answer = await replace_urls(answer)
        return {"modified_answer": modified_answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/bni_helps")
async def get_watsondiscovery_answer(request: Request):

    system_prompt="""Understand the 'passage' and answer the question based on the information provided. Include any links or URLs present in the passage. If a URL is mentioned, it must be maintained in the response without any additional notes or comments. Respond concisely, clearly, and avoid redundant information. Please do not generate clarifying questions. Provide a direct response or answer based on the given context."""
    
    try:
        user_question = await request.json()
        question = user_question['question']
        watson_qa_instance = WatsonQA()
        modified_answer = await watson_qa_instance.watsonxai(question, system_prompt)
        return modified_answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="Custom title",
#         version="3.0.2",
#         summary="This is a very custom OpenAPI schema",
#         description="Here's a longer description of the custom **OpenAPI** schema",
#         routes=app.routes,
#     )
#     openapi_schema["info"]["x-logo"] = {
#         "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
#     }
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom title",
        version="3.0.2",
        description="Here's a longer description of the custom **OpenAPI** schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    openapi_schema["servers"] = [{"url": "http://localhost:8000"}]  # Add your server URL
    app.openapi_schema = openapi_schema
    return app.openapi_schema



app.openapi = custom_openapi