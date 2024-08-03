import uvicorn
from fastapi import FastAPI
from openai import AsyncOpenAI
from schemas.schemas import SearchQuery, MultiSearchQueryResponse, ImageRequest


client = AsyncOpenAI()


app = FastAPI(
    title="Ecommerce Vision API",
    description="""A FastAPI application to extract products from images and describe them as an array of queries""",
    version="0.1.0",
)

@app.post("/api/extract_products", response_model=MultiSearchQueryResponse) #(2)!
async def extract_products(image_request: ImageRequest) -> MultiSearchQueryResponse: #(3)!
    completion = await client.chat.completions.create(
        model="gpt-4-vision-preview", #(4)!
        max_tokens=image_request.max_tokens,
        temperature=image_request.temperature,
        stop=["```"],
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an expert system designed to extract products from images for
                an ecommerce application. Please provide the product name and a
                descriptive query to search for the product. Accurately identify every
                product in an image and provide a descriptive query to search for the
                product. You just return a correctly formatted JSON object with the
                product name and query for each product in the image and follows the
                schema below:

                JSON Schema:
                {MultiSearchQueryResponse.model_json_schema()}""", #(5)!
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract the products from the image,
                        and describe them in a query in JSON format""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_request.url},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "```json", #(6)!
            },
        ],
    )
    return MultiSearchQueryResponse.model_validate_json(completion.choices[0].message.content)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"hello": "world"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)