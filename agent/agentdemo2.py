import torch._dynamo
torch._dynamo.config.suppress_errors = True
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_id = "D:\\Yolo_BitNet_Food\\agent\\BitNet\\models\\BitNet-b1.58-2B-4T"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # 或改为 torch.float16 看你 GPU 支持情况
    device_map="cuda"
)

# 定义生成食谱的函数
def generate_recipe(user_ingredients: str, prompt_style: str) -> str:
    prompt = (
        f"System: You are a professional AI chef assistant.\n"
        f"User: I have {user_ingredients}. Can you suggest a dish for me with the type in the following information:{prompt_style}?\n"
        f"Assistant:"
    )
    # 编码输入文本
    chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成输出
    chat_outputs = model.generate(
        **chat_input,
        max_new_tokens=500,
        repetition_penalty=1.2,  # 1.1~1.5，数值越大越抑制重复
        no_repeat_ngram_size=3,  # 禁止重复3-gram短语
        temperature=0.8,         # 适当调低温度控制输出多样性
        top_p=0.9                # 使用 nucleus sampling
    )

    # 解码输出并返回生成的食谱
    response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response

# 示例 prompt 内容，可以根据需求添加不同风格
promptdemo1 = """
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

    Enjoy a quick and tasty stir-fried potato and pepper dish with rice or steamed buns!
"""

promptdemo2 = """
    Ingredient Preparation

    🍝 Main Ingredients
    Tomato – 1 large (approx. 150g)
    Dried Noodles – 100g

    🌿 Seasonings
    Cooking Oil – 1 tablespoon
    Garlic – 2 cloves, minced
    Salt – to taste
    Sugar – 1/2 teaspoon (optional)
    Basil or Oregano – a pinch
    Water – as needed

    Cooking Instructions
    🔪 Prep Ingredients
    Dice the tomato. Mince garlic.

    🍳 Make the Sauce
    Heat oil in a pan, sauté garlic until fragrant. Add diced tomatoes, cook until soft. Add sugar, salt, herbs. Simmer to form a sauce.

    🍝 Boil Noodles
    Boil noodles in salted water until al dente. Drain and add to the sauce.

    🔄 Combine and Serve
    Toss noodles in the sauce until well coated. Serve hot with optional cheese or herbs on top.

    Tips
    💧 Add a bit of pasta water to adjust the sauce consistency.
    🔥 Use ripe tomatoes for better flavor.
    🕒 Don't overcook noodles – keep them firm.
    🧀 Add parmesan or chili flakes for extra taste.

    Enjoy a quick, comforting plate of tomato noodles!
"""

# 输入用户的食材
user_ingredients = "a tomato and some noodles"

# 使用新的 promptdemo2 生成食谱
result = generate_recipe(user_ingredients, promptdemo2)
print("\nAssistant Response:", result)
print("hello")
result2 = generate_recipe(user_ingredients, promptdemo1)
print("\nAssistant Response1:", result)
