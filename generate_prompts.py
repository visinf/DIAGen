import argparse
import ast
import random
import warnings

from transformers import AutoTokenizer, pipeline
import transformers
import torch
from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.custom_coco import CustomCOCO
from semantic_aug.datasets.focus import FOCUS
from typing import Dict
import os
import csv
import re
from openai import OpenAI
from dotenv import load_dotenv

DEFAULT_PROMPT = "a photo of a {name}"

# classes = ["mouse", "remote"]  # this can be used to create prompts for manual classes

# Start of Llama part
DEFAULT_PROMPT_W_SETTING = "a photo of a {name} in a {setting}"
DEFAULT_PROMPT_W_ADJECTIVE = "a photo of a {adjective} {name}"
DEFAULT_PROMPT_W_SETTING_AND_ADJECTIVE = "a photo of a {adjective} {name} in a {setting}"

SYS_PROMPT = "You are a helpful, respectful and precise assistant. \
You will be asked to generate {num_prompts} words. Only respond with those {num_prompts} words. \
Wrap those words as strings in a python list."

USER_PROMPTS = {
    "setting": "In which different settings can a {name} occur?",
    "adjective": "What are different visual and descriptive adjectives for {name}?",
}

PROMPT_TEMPLATE = f"""<s>[INST] <<SYS>>
{SYS_PROMPT}
<</SYS>>

[user_prompt] [/INST]
"""
# End of Llama part

# Start of GPT part
user_content_normal = "Create prompts for me that have the following structure:\n" \
                      "a photo of a [adjective] <classname> [location and/or weather preposition] [weather] [location] [time of day with preposition]\n\n" \
                      "The <classname> is replaced with the actual classname, e.g. 'car'\n" \
                      "All the attributes in [] are optionals. This means example prompts for car could be:\n" \
                      "'a photo of a red car' (adjective optional)\n" \
                      "'a photo of a car on a road' (location optional)\n" \
                      "'a photo of a car in snow' (weather optional)\n" \
                      "'a photo of a car at night' (time of day optional)\n" \
                      "'a photo of a huge car in a tunnel' (adjective and location optionals)\n" \
                      "'a photo of a green car on a foggy bridge at daytime' (all optionals)\n" \
                      "'a photo of a car' (no optional)\n\n" \
                      "If you use adjectives, they should be visual. So don't use something like 'interesting'.\n" \
                      "Also vary the number of optionals that you use.\n\n" \
                      "Can you give me {num_prompts} prompts of this structure for class {name} please."

user_content_reduced = "Create prompts for me that have the following structure:\n" \
                      "a photo of a [adjective] <classname> [location and/or weather preposition] [weather] [location] [time of day with preposition]\n\n" \
                      "The <classname> is replaced with the actual classname, e.g. 'car'\n" \
                      "All the attributes in [] are optionals.\n" \
                      "Also vary the number of optionals that you use.\n\n" \
                      "Can you give me {num_prompts} prompts of this structure for class {name} please."

user_content_long_1 = "Create prompts for me that have the following structure:\n" \
                      "a photo of a [adjective] <classname> [location and/or weather preposition] [weather] [location] [time of day with preposition]\n\n" \
                      "The <classname> is replaced with the actual classname, e.g. 'car'\n" \
                      "All the attributes in [] are optionals.\n" \
                      "Can you give me {num_prompts} prompts of this structure for class {name} please."

user_content_edge_cases = "Create prompts for me that have the following structure:\n" \
                          "a photo of a <classname> [uncommon location]\n\n" \
                          "The <classname> is replaced with the actual classname, e.g. 'car'\n" \
                          "You have to replace the [] with an uncommon location. This means example prompts for car could be:\n" \
                          "'a photo of a car underwater'\n" \
                          "'a photo of a car in snow'\n" \
                          "'a photo of a car in a desert'\n\n" \
                          "In my application I need prompts that cover edge cases.\n" \
                          "Can you give me {num_prompts} prompts of this structure for class {name} please."

user_content_temp = {
    "normal": user_content_normal,
    "edge_cases": user_content_edge_cases
}

GPT_PROMP_TEMPLATE = [{"role": "user", "content": user_content_temp}]
# End of GPT part

DATASETS = {
    "coco": COCODataset,
    "custom_coco": CustomCOCO,
    "focus": FOCUS,
}


def call_gpt_api(prompt, client, model):
    completion = client.chat.completions.create(
        model=model,
        messages=prompt,
    )
    return completion.choices[0].message.content


def extract_list_from_string(s):
    # Finds all substrings that matches Python list syntax
    matches = re.findall(r"\[.*?\]", s, re.DOTALL)
    for match in matches:
        print(f"Found list in LLM response")
        if "INST" in match:
            continue
        # Safely evaluate the string as a Python list
        return ast.literal_eval(match)
    raise Exception()


def extract_enum_as_list_from_string(s):
    cleaned_prompts = []
    lines = s.split('\n')

    # Define a regex pattern for lines starting with a number, period, and space
    pattern = r'^\d+\.\s*'

    for line in lines:
        # Check if the line matches the enumeration pattern
        if re.match(pattern, line):
            # remove enumeration
            prompt = line.split('.')[1]
            # remove the leading space
            prompt = prompt[1:]
            cleaned_prompts.append(prompt)

    return cleaned_prompts


def clean_single_prompt(p, c):
    # remove the leading space
    prompt_item = p[1:]
    # only replace the first occurrence of class name with {} -> the other one might be a descriptive word
    prompt_item = prompt_item.replace(f'[{c}]', '{name}', 1)
    # remove duplicate class mentioning in the prompt
    prompt_item = prompt_item.replace(f'[{c}]', '')

    # replace with default prompt if prompt is not as desired
    if not prompt_item.count('{name}') == 1:
        prompt_item = DEFAULT_PROMPT.format(name=c)

    return prompt_item


def clean_response_llama(res: str, num_prompts: int, class_name: str):
    try:
        lst = extract_list_from_string(res)
    except Exception as e:
        print(Warning(f"No list was found in the LLM response for class: {class_name}"))
        # Search for num_prompts words (if response was not a proper list)
        split = res.split(", ")
        if len(split) == num_prompts:
            lst = split
        else:
            # Search for enumeration
            lst = extract_enum_as_list_from_string(res)
            if len(lst) <= num_prompts:
                print(Warning(f"No enum was found in the LLM response for class: {class_name}"))
                raise Exception()

    # Remove _, classname and unnecessary spaces in response
    for i in range(len(lst)):
        lst[i] = lst[i].replace(class_name, "")
        lst[i] = lst[i].replace("_", " ")
        lst[i] = lst[i].strip()
    return lst


def clean_response_gpt(res: str, num_prompts: int, class_name: str):
    prompts = re.findall(r'\d+\.\s"?(.*?)"?(?=\n|$)', res)
    final_class_prompts = []
    for p in prompts:
        c_name_no_under_score = class_name.replace("_", " ")
        if c_name_no_under_score in p.lower():  # use .lower() to find "tv" in "a photo of a TV" etc.
            # final_class_prompts.append(p.replace(c_name_no_under_score, "{name}"))
            prompt_with_keyword = re.sub(r'\b{}\b'.format(c_name_no_under_score), "{name}", p, flags=re.IGNORECASE)
            final_class_prompts.append(prompt_with_keyword)
        else:
            print(f"No string '{class_name}' in prompt: {p}")
    if len(final_class_prompts) > num_prompts:
        random.shuffle(final_class_prompts)
        final_class_prompts = final_class_prompts[:num_prompts]
    if len(final_class_prompts) < num_prompts:
        warnings.warn(
            f"{num_prompts} prompts requested for class {class_name} but only found {len(final_class_prompts)}",
            UserWarning)
    return final_class_prompts


def construct_prompts(prompt_dict: dict, num_prompts: int, mode: str, classes: list):
    prompts = {}
    prompt_skeleton = DEFAULT_PROMPT
    kws = [mode]
    if mode == "setting_adjective":
        prompt_skeleton = DEFAULT_PROMPT_W_SETTING_AND_ADJECTIVE
        kws = ["setting", "adjective"]
    elif mode == "setting" or "uncommonSetting":
        prompt_skeleton = DEFAULT_PROMPT_W_SETTING
    elif mode == "adjective":
        prompt_skeleton = DEFAULT_PROMPT_W_ADJECTIVE

    for c in classes:
        class_prompt_list = []
        for i in range(num_prompts):
            temp = prompt_skeleton
            for kw in kws:
                word_list_for_kw = prompt_dict[kw][c]
                if len(word_list_for_kw) <= i:
                    # set standard prompt if no content word (for setting, ...) was found
                    temp = DEFAULT_PROMPT
                    print(Warning(f'Something went wrong during the prompt construction for {c}'))
                    break
                else:
                    if mode == "uncommonSetting":
                        temp = temp.replace("{setting}", prompt_dict[kw][c][i])
                    temp = temp.replace("{" + f"{kw}" + "}", prompt_dict[kw][c][i])
            class_prompt_list.append(temp)
        prompts[c] = class_prompt_list

    return prompts


def write_prompts_to_csv(all_prompts: Dict):
    # all_prompts contains a key for each class and the value are a list containing all prompts
    rows = []
    for class_name, class_prompts in all_prompts.items():
        for prompt_idx, single_prompt in enumerate(class_prompts, start=1):
            row = {'class_name': class_name, 'class_idx': prompt_idx, 'prompt': single_prompt}
            rows.append(row)

    # Writing to CSV
    out_dir = args.outdir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = "prompts.csv"
    if ".csv" in args.out_filename:
        filename = args.out_filename
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['class_name', 'class_idx', 'prompt'], delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


def process_gpt_api(client, model: str, c_names: list, num_prompts: int, content: str):
    prompt_dict = {}
    for c_name in c_names:
        # Create the input prompt for GPT
        c_name_no_under_score = c_name.replace("_", " ")
        user_content = user_content_temp[content].replace("{name}", c_name_no_under_score)
        num_margin = 5  # additional prompts if some prompts are bad
        user_content = user_content.replace("{num_prompts}", str(num_prompts + num_margin))
        input_prompt = GPT_PROMP_TEMPLATE
        input_prompt[0]['content'] = user_content

        # Call GPT and clean the response -> try multiple times if no prompt was found in the response
        output = call_gpt_api(input_prompt, client, model)
        prompt_list = []
        for i in range(5):
            prompt_list = clean_response_gpt(output, num_prompts, c_name)
            if len(prompt_list) == num_prompts:
                prompt_dict[c_name] = prompt_list
                break
        print(f"{c_name} -----> final prompts (count={len(prompt_list)}):\n{prompt_list}\n")
    return prompt_dict


def process_llama_api(model_path: str, content: str):
    prompt_words_key_word = {}  # stores all words for the final prompt for only settings or only adjective

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)
    pipe = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    content = content.split("_")
    for key_word in content:
        user_prompt = USER_PROMPTS[key_word]
        model_pr = PROMPT_TEMPLATE.replace("[user_prompt]", user_prompt)

        prompt_words = {}  # stores all words for the final prompt

        for idx in range(len(class_names)):
            name = class_names[idx]
            name_w_spaces = name.replace("_", " ")
            model_prompt = model_pr.format(num_prompts=str(args.prompts_per_class), name=name_w_spaces)

            # MR: sometimes the response of the LLM is not as required, hence try more often.
            prompt_okay = False
            trys_prompt = 0
            max_trys_prompt = 10
            while not prompt_okay and trys_prompt < max_trys_prompt:

                # MR: sometimes our Llama 2 configs lead to instabilities (tensor goes inf, nan, or negative).
                #   Then just do another call
                #   Yet I don't know why that happens...
                response = []  # just to declare it...
                response_okay = False
                trys_response = 0
                while not response_okay and trys_response < 10:
                    try:
                        # Call of LLM
                        response = pipe(
                            model_prompt,
                            do_sample=True,
                            top_k=10,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id,
                            max_length=1024,
                        )
                        response_okay = True
                    except RuntimeError as e:
                        print(Warning(f"Exception thrown while piping Llama 2: {e}"))
                        trys_response += 1

                print(f"\n{name} , try: {trys_prompt} -----> LLM response:\n{response[0]['generated_text']}")
                try:
                    prompt_words[name] = clean_response_llama(response[0]['generated_text'], args.prompts_per_class,
                                                              name)
                    prompt_okay = True
                except Exception as e:
                    if trys_prompt >= max_trys_prompt - 1:
                        print(f"After {max_trys_prompt} LLM calls no proper prompt was found for {name}")
                    else:
                        print(f"Doing another call of Llama2 to get a better response")
                    trys_prompt += 1

            if name in prompt_words.keys():
                print(f"\n{name} -----> final prompt words for {key_word}:\n{prompt_words[name]}")
            else:
                print(f"\n{name} -----> no words found for {key_word}:\n{prompt_words[name]}")

        prompt_words_key_word[key_word] = prompt_words

    return construct_prompts(prompt_words_key_word, args.prompts_per_class, args.content, class_names)


def init_gpt_api():
    return OpenAI(api_key=os.getenv('api_key'))


def configure():
    load_dotenv()


if __name__ == '__main__':

    '''
    example call from terminal:
    
    python generate_prompts.py --dataset "coco" --prompts-per-class 5 --content "setting_adjective"
    '''

    # Load .env
    configure()

    parser = argparse.ArgumentParser("LLM Prompt Generation")

    parser.add_argument("--outdir", type=str, default="prompts")
    parser.add_argument("--out-filename", type=str, default="prompts.csv")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        choices=["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "gpt-3.5-turbo",
                                 "gpt-4-turbo", "gpt-4o"])
    parser.add_argument("--prompts-per-class", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "custom_coco", "focus"])
    # --content is only active for llama models
    parser.add_argument("--content", type=str, default="setting_adjective",
                        choices=["setting", "adjective", "setting_adjective", "uncommonSetting"])
    # --content is only active for gpt models
    parser.add_argument("--content-gpt", type=str, default="normal", choices=["normal", "edge_cases"])

    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    classes = dataset.class_names
    class_names = classes

    # Initialize class_prompts
    class_prompts = {c: [] for c in classes}

    if "gpt" in args.model:
        client_ = init_gpt_api()
        class_prompts = process_gpt_api(client_, args.model, class_names, args.prompts_per_class, args.content_gpt)

    elif "llama" in args.model:
        class_prompts = process_llama_api(args.model, args.content)

    write_prompts_to_csv(class_prompts)
