import pandas as pd


def load_df(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    else:
        raise ValueError("Unsupported file format")

def format_messages(
    file_path,
    prompt_field,
    completion_positive_field,
    completion_negative_field,
    prompt_template_positive,
    prompt_template_negative,
):
    df = load_df(file_path)
    #### for rebutall ####
    if completion_negative_field in df.columns:
        df["domestication"] = df[completion_negative_field]
    if completion_positive_field in df.columns:
        df["foreignization"] = df[completion_positive_field]
    completion_negative_field = "domestication"
    completion_positive_field = "foreignization"

    messages = {
        "prompt_template_label": [],
        "completion_label": [],
        "messages": []
    }
    for i, row in df.iterrows():
        for prompt_template_label, prompt_template in enumerate([prompt_template_negative, prompt_template_positive]):
            try:
                messages["messages"].append(
                    [
                        {"role": "user", "content": prompt_template.format(input=row[prompt_field])},
                        {"role": "assistant", "content": row[completion_positive_field]},
                    ]
                )
                messages["prompt_template_label"].append(prompt_template_label)
                messages["completion_label"].append(1)
                messages["messages"].append(
                    [
                        {"role": "user", "content": prompt_template.format(input=row[prompt_field])},
                        {"role": "assistant", "content": row[completion_negative_field]},
                    ]
                )
                messages["prompt_template_label"].append(prompt_template_label)
                messages["completion_label"].append(0)
            except KeyError:
                messages["messages"].append(
                    [
                        {"role": "user", "content": prompt_template.format(input=row[f"{prompt_field}_{completion_positive_field}"])},
                        {"role": "assistant", "content": row[completion_positive_field]},
                    ]
                )
                messages["prompt_template_label"].append(prompt_template_label)
                messages["completion_label"].append(1)
                messages["messages"].append(
                    [
                        {"role": "user", "content": prompt_template.format(input=row[f"{prompt_field}_{completion_negative_field}"])},
                        {"role": "assistant", "content": row[completion_negative_field]},
                    ]
                )
                messages["prompt_template_label"].append(prompt_template_label)
                messages["completion_label"].append(0)

    df = pd.DataFrame(messages)
    df_for_model_positive = df[df["prompt_template_label"] == 1]
    df_for_model_negative = df[df["prompt_template_label"] == 0]
    return df_for_model_positive, df_for_model_negative