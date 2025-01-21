import os
from openai import OpenAI
import json

# GPT API Key
openai_api_key = os.getenv('OPENAI_API_KEY')

def runGptPrediction(values):

    client = OpenAI(
        api_key=openai_api_key,
    )

    # Prepare the prompt for GPT
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that processes a list of items. "
                "Each item is a list containing three elements:\n"
                "1. Item name or text (which may be misspelled)\n"
                "2. Price\n"
                "3. Tag (e.g., ##PRICE:, ##SUBTOTAL:, ##TOTAL:)\n\n"
                "For each item, please:\n"
                "- Correct any misspellings in the item name or text.\n"
                "- Swap the order if necessary so that the item name comes first, "
                "followed by the tag and price.\n"
                "- If the tag corresponds to a priced line item, output it as a JSON object with:\n"
                "    {\n"
                "      \"type\": \"receipt item\",\n"
                "      \"name\": \"<Corrected Item Name>\",\n"
                "      \"price\": <Price>\n"
                "    }\n"
                "- If the tag corresponds to a subtotal, output it as:\n"
                "    {\n"
                "      \"type\": \"subtotal\",\n"
                "      \"price\": <Price>\n"
                "    }\n"
                "- If the tag corresponds to a total, output it as:\n"
                "    {\n"
                "      \"type\": \"total\",\n"
                "      \"price\": <Price>\n"
                "    }\n"
                "Collect all these objects into a JSON array—nothing else. "
                "Return only that valid JSON array, for example:\n\n"
                "[\n"
                "  {\"type\": \"receipt item\", \"name\": \"Apple\", \"price\": 1.5},\n"
                "  {\"type\": \"subtotal\", \"price\": 10},\n"
                "  {\"type\": \"total\", \"price\": 10}\n"
                "]\n\n"
                "Do not include any additional text, explanation, or code fences."
            )
        },
        {
            "role": "user",
            "content": f"Here is the list of items:\n{values}"
        }
    ]

    # Call the GPT API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    # Extract the assistant's reply
    assistant_reply = response.choices[0].message.content
    print("GPT raw output:", assistant_reply)

    # Convert the JSON string to a Python list (or array of objects)
    try:
        corrected_list = json.loads(assistant_reply)
    except json.JSONDecodeError:
        raise ValueError("GPT did not return valid JSON.")

    return corrected_list
    


def runRecieptPredictionGpt(image):
    # takes base64 encoded receipt and makes api call to chatGPT to decipher contents.
    client = OpenAI(
        api_key=openai_api_key,
    )
    

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that processes an image of a receipt containing user items. "
                "Do not include items that have a cost of 0."
                "Do not include tax as an element in the json object."
                "Remeber that the total value of the purchase might be under other names other then just total."
                "ONLY IF the name of the store the receipt is from is deceipherable, add the name of the store to the final JSON Object."
                "Return a valid JSON object, for example:\n\n"
                "{\n"
                " \"store\": Walmart,\n"
                "  \"items\": [\n"
                "    {\n"
                "      \"name\": \"Apple\",\n"
                "      \"price\": 1.5\n"
                "    }\n"
                "  ],\n"
                "  \"subtotal\": {\n"
                "    \"price\": 10\n"
                "  },\n"
                "  \"total\": {\n"
                "    \"price\": 10\n"
                "  }\n"
                "}\n\n"
                "Do not include any additional text, explanation, or code fences."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ],
        }
    ],
    )

    assistant_reply = response.choices[0].message.content
    # Convert the JSON string to a Python list (or array of objects)
    try:
        corrected_list = json.loads(assistant_reply)
    except json.JSONDecodeError:
        return [500, "GPT Error"]
    return [200, corrected_list]

'''
Example return object
{
  "items": [
    {
      "name": "Deboned Chicken Thigh",
      "price": 14.99
    },
    {
      "name": "4 pcs",
      "price": 8.49
    }
  ],
  "subtotal": {
    "price": 23.48
  },
  "total": {
    "price": 26.53
  }
}
'''