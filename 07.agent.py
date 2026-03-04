import json
from openai import AzureOpenAI
import sys
import io
from dotenv import load_dotenv
import os

load_dotenv()

ENDPOINT = os.getenv("ENDPOINT")
API_KEY = os.getenv("API_KEY")
API_TYPE = os.getenv("API_TYPE")
API_VERSION = os.getenv("API_VERSION")

# 1. Azure OpenAI 설정
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)

deployment_name = "gpt-4.1" # 실제 Azure 배포명을 입력하세요

# 2. 실제 실행될 함수(Tools) 정의
def get_weather(city):
    weather_map = {"도쿄": "흐림, 18도", "파리": "맑음, 22도", "뉴욕": "눈, -2도"}
    result = weather_map.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")
    print(f"[시스템 로그] 날씨 조회 중: {city} -> {result}")
    return result

def get_exchange_rate(currency_code):
    rates = {"JPY": 9.2, "USD": 1340.5, "EUR": 1460.0}
    result = rates.get(currency_code.upper(), 1.0)
    print(f"[시스템 로그] 환율 조회 중: {currency_code} -> {result}")
    return result

# 3. 모델에게 알려줄 도구 정보(Metadata)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨와 온도를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름 (예: 도쿄, 파리)"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "원화(KRW) 대비 해당 통화의 환율을 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency_code": {"type": "string", "description": "통화 코드 (예: JPY, USD, EUR)"}
                },
                "required": ["currency_code"],
            },
        },
    }
]

# 4. 에이전트 실행 루프
def run_travel_agent(user_prompt):
    messages = [
        {"role": "system", "content": "너는 유능한 여행사 직원이야. 도구를 사용해 정확한 정보를 제공해줘."},
        {"role": "user", "content": user_prompt}
    ]

    # 첫 번째 호출: 모델이 도구를 사용할지 판단
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 모델이 도구 사용을 결정했다면
    if tool_calls:
        messages.append(response_message)
        
        # 각 도구 호출 실행
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "get_weather":
                function_response = get_weather(function_args.get("city"))
            elif function_name == "get_exchange_rate":
                function_response = str(get_exchange_rate(function_args.get("currency_code")))
            
            # 도구 실행 결과를 메시지에 추가
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

            # 두 번째 호출: 도구 결과를 바탕으로 최종 답변 생성
        second_response = client.chat.completions.create(
            model=deployment_name,
            messages=messages
        )
        return second_response.choices[0].message.content

    return response_message.content


# 5. 테스트 실행
test_prompt = "도쿄 날씨 알려주고, 10,000엔을 환전하면 한화로 얼마인지 계산해줘."
print(f"최종 답변: {run_travel_agent(test_prompt)}")