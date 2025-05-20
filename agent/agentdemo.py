import torch._dynamo
torch._dynamo.config.suppress_errors = True
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 去除尾部重复 emoji（例如 👌 连续重复）
import re

promptdemo = """
    Ingredient Preparation
    
   🥔 Main Ingredients
    Potatoes – 2 (approx. 400g)
    
    Green or Chili Peppers – 2–3 (approx. 100g)
    
    🥩 Optional Add-ins
    Shredded Pork – 50g (can be omitted for a vegetarian version)
    
    🌱 Seasonings
    Cooking Oil – appropriate amount
    
    Green Onion & Garlic – a small handful each, finely chopped
    
    Cooking Wine – 1 teaspoon
    
    Light Soy Sauce – 1 tablespoon
    
    White Sugar – ½ teaspoon (optional)
    
    Salt – to taste
    
    Cooking Instructions
    🔪 Julienne and Soak
    Peel and finely julienne the potatoes. Soak in clean water for 5 minutes to remove excess starch. Once the water runs clear, drain and set aside.
    
    Remove stems from the peppers and cut into thin strips. Chop the green onion and garlic.
    
    🍳 Stir-Fry in Oil or Dry
    Add an appropriate amount of oil to a wok and heat to medium (around 150°C / 300°F).
    
    Add the potato strips and stir-fry on high heat for 1–2 minutes until just cooked (still crisp), then remove and drain excess oil.
    
    🍳 Aromatics and Flavoring
    Leave a little oil in the wok. Add chopped green onion and garlic, stir-fry for about 10 seconds until fragrant.
    
    Add shredded pork and stir-fry until it changes color. Pour in cooking wine and soy sauce, stir quickly to combine.
    
    🔄 Combine and Finish
    Return the drained potato strips to the wok, stir-fry on high heat until well-seasoned.
    
    Add pepper strips and stir-fry for another 30 seconds.
    
    Sprinkle in salt and sugar, stir-fry for 10–15 more seconds, then remove from heat and serve.
    
    Tips
    💧 Drain thoroughly: Make sure the potato strips are well-drained after soaking to avoid oil splatter or sogginess.
    🔥 Use high heat: The whole process should be done over high heat to maintain the crisp texture of the potatoes.
    🕒 Keep it quick: Both the potatoes and peppers should be cooked briefly to stay crunchy.
    🥄 Adjust seasonings to taste: Feel free to add sugar, vinegar, or use different types of chili based on your preference.
    
    With these simple steps, you can quickly make a delicious and appetite-boosting stir-fried potato and pepper dish at home. Enjoy it with rice or steamed buns for a true taste of home cooking!
    """

promptdemo2 = """
    Ingredient Preparation

    Main Ingredients
    Potatoes – 2 (approx. 400g)

    Green or Chili Peppers – 2–3 (approx. 100g)

    Optional Add-ins
    Shredded Pork – 50g (can be omitted for a vegetarian version)

    Seasonings
    Cooking Oil – appropriate amount

    Green Onion & Garlic – a small handful each, finely chopped

    Cooking Wine – 1 teaspoon

    Light Soy Sauce – 1 tablespoon

    White Sugar – ½ teaspoon (optional)

    Salt – to taste

    Cooking Instructions
    Julienne and Soak
    Peel and finely julienne the potatoes. Soak in clean water for 5 minutes to remove excess starch. Once the water runs clear, drain and set aside.

    Remove stems from the peppers and cut into thin strips. Chop the green onion and garlic.

    Stir-Fry in Oil or Dry
    Add an appropriate amount of oil to a wok and heat to medium (around 150°C / 300°F).

    Add the potato strips and stir-fry on high heat for 1–2 minutes until just cooked (still crisp), then remove and drain excess oil.

    Aromatics and Flavoring
    Leave a little oil in the wok. Add chopped green onion and garlic, stir-fry for about 10 seconds until fragrant.

    Add shredded pork and stir-fry until it changes color. Pour in cooking wine and soy sauce, stir quickly to combine.

    Combine and Finish
    Return the drained potato strips to the wok, stir-fry on high heat until well-seasoned.

    Add pepper strips and stir-fry for another 30 seconds.

    Sprinkle in salt and sugar, stir-fry for 10–15 more seconds, then remove from heat and serve.

    Tips
       Drain thoroughly: Make sure the potato strips are well-drained after soaking to avoid oil splatter or sogginess.
       Use high heat: The whole process should be done over high heat to maintain the crisp texture of the potatoes.
       Keep it quick: Both the potatoes and peppers should be cooked briefly to stay crunchy.
       Adjust seasonings to taste: Feel free to add sugar, vinegar, or use different types of chili based on your preference.

    With these simple steps, you can quickly make a delicious and appetite-boosting stir-fried potato and pepper dish at home. Enjoy it with rice or steamed buns for a true taste of home cooking!
    """

model_id = "D:\\Yolo_BitNet_Food\\agent\\BitNet\\models\\BitNet-b1.58-2B-4T"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   # 或改为 torch.float16 看你 GPU 支持情况
    device_map="cuda"
)


prompt = (
    f"System: You are a professional AI chef assistant.\n "
    f"User: I have a tomato and some noodles. Can you suggest a dish for me with the type in the following information:{promptdemo}?\n"
    f"Assistant:"
)

# prompt = (
#     "System: You are a professional AI chef assistant. "
#     "When a user gives you ingredients, respond with a structured recipe in English, using the following format:\n\n"
#     "Title of the Dish (you name it based on the ingredients)\n\n"
#     "Ingredient Preparation\n"
#     "🥔 Main Ingredients\n"
#     "- List the main ingredients with quantities\n\n"
#     "🥩 Optional Add-ins\n"
#     "- List optional ingredients\n\n"
#     "🌱 Seasonings\n"
#     "- List common seasonings with quantities\n\n"
#     "Cooking Instructions\n"
#     "🔪 Step Name\n"
#     "Step description.\n"
#     "🍳 Step Name\n"
#     "Step description.\n"
#     "🔄 Step Name\n"
#     "Step description.\n\n"
#     "Tips\n"
#     "💧 Tip 1\n"
#     "🔥 Tip 2\n"
#     "🕒 Tip 3\n"
#     "🥄 Tip 4\n\n"
#     "End with a warm comment encouraging the user to enjoy the meal.\n\n"
#     "User: I have a tomato  and some noodles. Can you suggest a dish for me?\n"
#     "Assistant:"
# )


chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
# chat_outputs = model.generate(**chat_input, max_new_tokens=600)
chat_outputs = model.generate(
    **chat_input,
    max_new_tokens=500,
    repetition_penalty=1.2,  # 1.1~1.5，数值越大越抑制重复
    no_repeat_ngram_size=3,  # 禁止重复3-gram短语
    temperature=0.8,         # 适当调低温度控制输出多样性
    top_p=0.9                # 使用 nucleus sampling
)
response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)




# response = re.sub(r'(?:👌|👍|🍽️|🍜){3,}', lambda m: m.group(0)[:3], response)

print("\nAssistant Response:", response)
# print("\nAssistant Response:", response)
